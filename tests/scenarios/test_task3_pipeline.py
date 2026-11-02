"""Task 3 scenarios: duplicates, failed updates, outages, unauthorised access, maintenance windows."""
import json
import shutil
from datetime import timedelta

import pytest

from a_real_time_gen_ai_customer_service_bot.core.auth import PermissionDenied
from a_real_time_gen_ai_customer_service_bot.core.config import PROJECT_ROOT
from a_real_time_gen_ai_customer_service_bot.kb_pipeline.pipeline import KBPipeline
from a_real_time_gen_ai_customer_service_bot.kb_pipeline.scheduler import UpdateScheduler
from a_real_time_gen_ai_customer_service_bot.kb_pipeline.store import RollbackError, VersionedStore
from a_real_time_gen_ai_customer_service_bot.monitoring.metrics import MetricsCollector
from conftest import ist

SRC = PROJECT_ROOT / "data" / "kb_docs"
EVAL = json.loads((PROJECT_ROOT / "data" / "eval" / "eval_cases.json").read_text())
IN_WINDOW = lambda: ist(13, 2, 0)            # Tuesday 02:00 IST, inside 01:00-04:00 (Tue 13 Oct 2026)


@pytest.fixture
def env(tmp_path, config, clock, auth):
    docs = tmp_path / "docs"
    shutil.copytree(SRC, docs)
    health = {"ok": True, "raise": False}

    def check():
        if health["raise"]:
            raise ConnectionError("service down")
        return health["ok"]

    metrics = MetricsCollector(config, clock)
    pipe = KBPipeline(config, clock, auth, VersionedStore(clock), docs, tmp_path / "quarantine",
                      EVAL, metrics=metrics, health_check=check)
    pipe.health = health
    pipe.docs = docs
    pipe.metrics_ = metrics
    return pipe


def activate_initial(pipe, clock):
    clock.set(IN_WINDOW())
    result = pipe.run_update("kb_manager")
    assert result.status == "activated", result
    clock.advance(minutes=6)
    assert pipe.health_tick().status == "confirmed"
    return result


def write_doc(path, doc_id="X1", version=1, body="Some body text here.", title="T", region="global", access="public"):
    path.write_text(f"---\ndoc_id: {doc_id}\ntitle: {title}\ndoc_type: faq\nversion: {version}\nproduct: all\n"
                    f"region: {region}\naccess_level: {access}\neffective_date: 2026-01-01\n---\n{body}\n")


# --- ingest: only new / modified, duplicates, quarantine, versions --------------------------------
def test_first_update_loads_all_valid_documents(env, clock):
    r = activate_initial(env, clock)
    assert len(r.ingest.new) == 10 and not r.ingest.quarantined
    assert r.quality.passed and r.quality.accuracy >= 0.8 and r.version_id == 1
    assert env.store.active.version_id == 1


def test_only_new_or_modified_files_are_processed(env, clock):
    activate_initial(env, clock)
    calls = env.parse_calls
    clock.set(IN_WINDOW() + timedelta(days=1))
    assert env.run_update("kb_manager").status == "no_changes" and env.parse_calls == calls
    write_doc(env.docs / "new-faq.md", "NEW-1", body="Gift cards never expire.")
    r = env.run_update("kb_manager")
    assert r.ingest.new == ["new-faq.md"] and len(r.ingest.unchanged) == 10 and env.parse_calls == calls + 1


def test_modified_file_is_detected_and_creates_new_version(env, clock):
    activate_initial(env, clock)
    clock.set(IN_WINDOW() + timedelta(days=1))
    f = env.docs / "warranty-faq.md"
    f.write_text(f.read_text().replace("12 month", "12 month (extended)"))
    r = env.run_update("kb_manager")
    assert r.ingest.modified == ["warranty-faq.md"] and r.status == "activated" and r.version_id == 2


def test_duplicate_document_is_detected_and_skipped(env, clock):
    activate_initial(env, clock)
    shutil.copy(env.docs / "warranty-faq.md", env.docs / "warranty-faq-copy.md")
    clock.set(IN_WINDOW() + timedelta(days=1))
    r = env.run_update("kb_manager")
    assert r.ingest.duplicates == {"warranty-faq-copy.md": "warranty-faq.md"} and r.status == "no_changes"


def test_duplicates_within_the_first_batch(env):
    shutil.copy(env.docs / "warranty-faq.md", env.docs / "zz-copy.md")
    report = env.ingest()
    assert "zz-copy.md" in report.duplicates and len(report.documents) == 10


def test_renamed_file_is_not_lost(env, clock):
    activate_initial(env, clock)
    (env.docs / "warranty-faq.md").rename(env.docs / "warranty-renamed.md")
    clock.set(IN_WINDOW() + timedelta(days=1))
    r = env.run_update("kb_manager")
    assert "warranty-renamed.md" in env.store.active.documents and "warranty-faq.md" not in env.store.active.documents


def test_invalid_files_are_quarantined_and_valid_ones_still_load(env):
    (env.docs / "broken.md").write_text("no metadata here")
    (env.docs / "malware.exe").write_bytes(b"MZ\x90\x00")
    (env.docs / "binary.md").write_bytes(b"\xff\xfe\x00\x01")
    (env.docs / "empty.md").write_text("---\ndoc_id: E\ntitle: t\ndoc_type: faq\nversion: 1\nproduct: all\nregion: global\n"
                                       "access_level: public\neffective_date: 2026-01-01\n---\n")
    report = env.ingest()
    assert set(report.quarantined) == {"broken.md", "malware.exe", "binary.md", "empty.md"}
    assert len(report.documents) == 10
    for name in report.quarantined:
        assert (env.quarantine_dir / name).exists() and (env.quarantine_dir / f"{name}.reason.txt").read_text()
        assert not (env.docs / name).exists()


def test_same_doc_id_and_version_with_different_content_is_quarantined(env, clock):
    activate_initial(env, clock)
    write_doc(env.docs / "clash.md", doc_id="RET-POL", version=2, body="Returns within 3 days.")
    report = env.ingest()
    assert "clash.md" in report.quarantined and "return-policy-v2.md" not in report.quarantined


def test_unknown_access_level_is_quarantined(env):
    write_doc(env.docs / "weird.md", access="top-secret")
    assert "weird.md" in env.ingest().quarantined


def test_injection_in_document_is_sanitised_and_flagged(env):
    report = env.ingest()
    assert "shipping-faq.md" in report.injection_flagged
    assert "password" not in report.documents["shipping-faq.md"].content.lower()
    assert env.metrics_.snapshot()["failures"]["by_kind"]["prompt_injection_in_document"] == 1


def test_pii_is_masked_during_ingest(env):
    write_doc(env.docs / "pii.md", "PII-1", body="Call our manager at 9876543210 or mail boss@corp.com for help.")
    doc = env.ingest().documents["pii.md"]
    assert "9876543210" not in doc.content and "boss@corp.com" not in doc.content


# --- quality gate ----------------------------------------------------------------------------------
def test_update_that_reduces_accuracy_is_rejected_and_live_kb_unchanged(env, clock):
    activate_initial(env, clock)
    clock.set(IN_WINDOW() + timedelta(days=1))
    f = env.docs / "return-policy-v2.md"
    f.write_text(f.read_text().replace("30 days", "90 days"))
    r = env.run_update("kb_manager")
    assert r.status == "rejected" and any("accuracy" in x for x in r.quality.reasons)
    assert env.store.active.version_id == 1
    assert "30 days" in env.store.active.documents["return-policy-v2.md"].content


def test_deleting_an_important_document_is_rejected(env, clock):
    activate_initial(env, clock)
    clock.set(IN_WINDOW() + timedelta(days=1))
    (env.docs / "router-troubleshooting.md").unlink()
    assert env.run_update("kb_manager").status == "rejected"


def test_rejected_candidate_is_not_re_evaluated_until_files_change(env, clock):
    activate_initial(env, clock)
    clock.set(IN_WINDOW() + timedelta(days=1))
    (env.docs / "router-troubleshooting.md").unlink()
    env.run_update("kb_manager")
    again = env.run_update("kb_manager")
    assert again.status == "rejected" and "previously rejected" in again.detail


def test_grounding_drop_is_rejected(env, clock, monkeypatch):
    activate_initial(env, clock)
    clock.set(IN_WINDOW() + timedelta(days=1))
    write_doc(env.docs / "extra.md", "EXTRA", body="Extra information about gifts.")
    real = env._score
    calls = {"n": 0}

    def fake(docs):
        calls["n"] += 1
        acc, grd, fails = real(docs)
        return (acc, grd - 0.2 if calls["n"] == 1 else grd, fails)      # candidate scored first
    monkeypatch.setattr(env, "_score", fake)
    r = env.run_update("kb_manager")
    assert r.status == "rejected" and any("grounding" in x for x in r.quality.reasons)


def test_quality_minimums_are_configurable(env, config, clock):
    env.eval_cases = env.eval_cases + [{"question": "What is the price of the moon?", "role": "customer",
                                        "region": "EU", "expect_doc": "NOPE", "expect_contains": "nothing"}]
    config.set("kb_pipeline.min_accuracy", 0.95)                 # 9/10 = 0.90 < 0.95
    clock.set(IN_WINDOW())
    r = env.run_update("kb_manager")
    assert r.status == "rejected" and "below minimum" in r.detail
    config.set("kb_pipeline.min_accuracy", 0.80)
    clock.advance(minutes=1)
    assert env.run_update("kb_manager").status == "activated"    # new thresholds -> candidate is judged again


# --- maintenance window ------------------------------------------------------------------------------
def test_approved_update_outside_window_waits_then_activates_inside(env, clock):
    clock.set(ist(12, 14, 0))                                    # 14:00 IST: outside
    r = env.run_update("kb_manager")
    assert r.status == "approved_pending_window" and env.store.active is None and env.pending
    assert not env.activate_pending()
    clock.set(ist(13, 1, 0))                                     # window opens
    assert env.activate_pending() and env.store.active.version_id == 1 and env.pending is None


@pytest.mark.parametrize("h,m,inside", [(0, 59, False), (1, 0, True), (3, 59, True), (4, 0, False), (14, 0, False)])
def test_window_boundaries(env, clock, h, m, inside):
    assert env.in_maintenance_window(ist(13, h, m)) is inside


def test_window_is_configurable_and_may_cross_midnight(env, config):
    config.set("kb_pipeline.maintenance_window.start", "22:00")
    config.set("kb_pipeline.maintenance_window.end", "02:00")
    assert env.in_maintenance_window(ist(13, 23, 30)) and env.in_maintenance_window(ist(13, 1, 0))
    assert not env.in_maintenance_window(ist(13, 3, 0)) and not env.in_maintenance_window(ist(13, 21, 59))


# --- health check and automatic rollback ------------------------------------------------------------------
def test_failed_health_check_within_five_minutes_rolls_back_automatically(env, clock):
    activate_initial(env, clock)
    clock.set(IN_WINDOW() + timedelta(days=1))
    write_doc(env.docs / "new-faq.md", "NEW-1", body="Gift cards never expire.")
    assert env.run_update("kb_manager").status == "activated" and env.store.active.version_id == 2
    clock.advance(minutes=2)
    env.health["ok"] = False
    result = env.health_tick()
    assert result.status == "rolled_back" and env.store.active.version_id == 1
    assert env.store.versions[2].status == "rolled_back" and env.watch is None


def test_service_outage_during_health_check_triggers_rollback(env, clock):
    activate_initial(env, clock)
    clock.set(IN_WINDOW() + timedelta(days=1))
    write_doc(env.docs / "new-faq.md", "NEW-1", body="Gift cards never expire.")
    env.run_update("kb_manager")
    env.health["raise"] = True
    assert env.health_tick().status == "rolled_back" and env.store.active.version_id == 1


def test_healthy_update_is_confirmed_after_the_observation_period(env, clock):
    activate_initial(env, clock)
    clock.set(IN_WINDOW() + timedelta(days=1))
    write_doc(env.docs / "new-faq.md", "NEW-1", body="Gift cards never expire.")
    env.run_update("kb_manager")
    clock.advance(minutes=4, seconds=59)
    assert env.health_tick().status == "healthy"
    clock.advance(seconds=1)
    assert env.health_tick().status == "confirmed" and env.store.active.version_id == 2


def test_health_window_length_is_configurable(env, config, clock):
    config.set("kb_pipeline.health_check_minutes", 1)
    clock.set(IN_WINDOW())
    env.run_update("kb_manager")
    clock.advance(minutes=1)
    assert env.health_tick().status == "confirmed"


def test_unhealthy_first_version_has_nothing_to_roll_back_to(env, clock):
    clock.set(IN_WINDOW())
    env.run_update("kb_manager")
    env.health["ok"] = False
    assert env.health_tick().status == "unhealthy_no_rollback"


def test_manual_rollback_needs_permission_and_a_previous_version(env, clock):
    activate_initial(env, clock)
    with pytest.raises(RollbackError):
        env.rollback("admin")
    clock.set(IN_WINDOW() + timedelta(days=1))
    write_doc(env.docs / "new-faq.md", "NEW-1", body="Gift cards never expire.")
    env.run_update("kb_manager")
    with pytest.raises(PermissionDenied):
        env.rollback("customer")
    assert env.rollback("admin").version_id == 1


def test_versions_are_kept_for_history(env, clock):
    activate_initial(env, clock)
    clock.set(IN_WINDOW() + timedelta(days=1))
    write_doc(env.docs / "new-faq.md", "NEW-1", body="Gift cards never expire.")
    env.run_update("kb_manager")
    assert set(env.store.versions) == {1, 2} and env.store.versions[1].status == "retired"


# --- access control -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("role", ["customer", "agent", None, "hacker"])
def test_unauthorised_roles_cannot_update(env, role):
    with pytest.raises(PermissionDenied):
        env.run_update(role)
    assert env.store.active is None


def test_permissions_can_be_changed_at_runtime(env, config, clock):
    config.set("auth.permissions", {"kb_update": ["agent"], "kb_rollback": [], "view_escalations": []})
    clock.set(IN_WINDOW())
    assert env.run_update("agent").status == "activated"


# --- scheduler: configurable time, retries 15/30/60, outages -----------------------------------------------------------
@pytest.fixture
def sched(env, clock):
    clock.set(ist(12, 12, 0))                                    # Monday noon
    return UpdateScheduler(env._config, clock, env)


def kinds(events):
    return [e.type for e in events]


def test_scheduler_does_not_run_before_update_time_and_runs_at_it(sched, clock):
    assert sched.tick() == []
    clock.set(ist(13, 1, 59))
    assert sched.tick() == []
    clock.set(ist(13, 2, 0))
    ev = sched.tick()
    assert kinds(ev) == ["update_ok"] and sched._pipeline.store.active.version_id == 1


def test_scheduler_runs_once_per_day(sched, clock):
    clock.set(ist(13, 2, 0))
    sched.tick()
    clock.set(ist(13, 2, 30))
    assert sched.tick() == [] or kinds(sched.tick()) == []
    clock.set(ist(14, 2, 0))
    assert kinds(sched.tick()) == ["update_ok"]


def test_update_time_is_configurable_at_runtime(sched, clock, config):
    config.set("kb_pipeline.update_time", "03:30")
    clock.set(ist(13, 2, 30))
    assert sched.tick() == []
    clock.set(ist(13, 3, 30))
    assert kinds(sched.tick()) == ["update_ok"]


def break_source(env):
    env.docs.rename(env.docs.parent / "docs_offline")


def restore_source(env):
    (env.docs.parent / "docs_offline").rename(env.docs)


def test_failed_update_retries_after_15_30_60_minutes_then_gives_up(sched, clock, env):
    break_source(env)
    clock.set(ist(13, 2, 0))
    assert kinds(sched.tick()) == ["update_failed", "retry_scheduled"]
    clock.set(ist(13, 2, 14))
    assert sched.tick() == []                                        # not yet: needs 15 min
    clock.set(ist(13, 2, 15))
    assert kinds(sched.tick()) == ["update_failed", "retry_scheduled"] and "30" in sched.events[-1].detail
    clock.set(ist(13, 2, 44))
    assert sched.tick() == []
    clock.set(ist(13, 2, 45))                                        # +30
    assert kinds(sched.tick()) == ["update_failed", "retry_scheduled"] and "60" in sched.events[-1].detail
    clock.set(ist(13, 3, 44))
    assert sched.tick() == []
    clock.set(ist(13, 3, 45))                                        # +60
    assert kinds(sched.tick()) == ["update_failed", "gave_up"]
    clock.set(ist(13, 6, 0))
    assert sched.tick() == []                                        # nothing more today
    restore_source(env)
    clock.set(ist(14, 2, 0))
    assert kinds(sched.tick()) == ["update_ok"]                      # next day works again


def test_retry_succeeds_when_service_recovers(sched, clock, env):
    break_source(env)
    clock.set(ist(13, 2, 0))
    sched.tick()
    restore_source(env)
    clock.set(ist(13, 2, 15))
    assert kinds(sched.tick()) == ["update_ok"] and env.store.active.version_id == 1
    assert sched.retry_at is None


def test_retry_delays_are_configurable(sched, clock, env, config):
    config.set("kb_pipeline.retry_delays_minutes", [1])
    break_source(env)
    clock.set(ist(13, 2, 0))
    sched.tick()
    clock.set(ist(13, 2, 1))
    assert kinds(sched.tick()) == ["update_failed", "gave_up"]


def test_unauthorised_scheduler_identity_fails_and_never_updates(env, clock, config):
    s = UpdateScheduler(config, clock, env, role="customer")
    clock.set(ist(13, 2, 0))
    ev = s.tick()
    assert ev[0].type == "update_failed" and "PermissionDenied" in ev[0].detail and env.store.active is None


def test_scheduler_activates_pending_update_when_window_opens(env, clock, config):
    config.set("kb_pipeline.update_time", "12:00")               # update runs outside the window
    clock.set(ist(12, 11, 0))
    s = UpdateScheduler(config, clock, env)
    clock.set(ist(12, 12, 0))
    assert kinds(s.tick()) == ["update_ok"] and env.pending
    clock.set(ist(13, 1, 0))
    assert "activated_pending" in kinds(s.tick()) and env.store.active.version_id == 1


def test_scheduler_reports_auto_rollback_event(sched, clock, env):
    clock.set(ist(13, 2, 0))
    sched.tick()                                                  # v1 active
    clock.set(ist(13, 2, 6))
    sched.tick()                                                  # confirmed
    clock.set(ist(14, 2, 0))
    write_doc(env.docs / "new-faq.md", "NEW-1", body="Gift cards never expire.")
    sched.tick()                                                  # v2 active
    env.health["ok"] = False
    clock.set(ist(14, 2, 2))
    assert any(e.type == "health" and "rolled_back" in e.detail for e in sched.tick())
    assert env.store.active.version_id == 1


# --- monitoring ---------------------------------------------------------------------------------------------------------
def test_metrics_snapshot_and_alerts(config, clock):
    m = MetricsCollector(config, clock)
    for ms in (100, 200, 7000):
        m.record_latency("chat", ms)
    m.record_confidence(0.2)
    m.record_confidence(0.3)
    for _ in range(6):
        m.record_failure("timeout")
    m.record_escalation("high_risk:duplicate_payment")
    s = m.snapshot()
    assert s["latency"]["count"] == 3 and s["failures"]["count"] == 6 and s["escalations"]["count"] == 1
    assert {a.name for a in m.alerts()} == {"high_latency", "low_confidence", "many_failures"}


def test_metrics_window_slides_with_time(config, clock):
    m = MetricsCollector(config, clock)
    m.record_failure("x")
    clock.advance(minutes=16)
    assert m.snapshot()["failures"]["count"] == 0
    assert m.snapshot(window_minutes=60)["failures"]["count"] == 1
