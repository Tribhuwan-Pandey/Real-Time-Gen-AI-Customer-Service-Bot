"""Regression tests for problems found in the full-project audit."""
import os
import shutil
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from a_real_time_gen_ai_customer_service_bot.core.config import Config, ConfigError, PROJECT_ROOT
from a_real_time_gen_ai_customer_service_bot.main import load_env_file, resolve_role
from a_real_time_gen_ai_customer_service_bot.monitoring.metrics import MetricsCollector
from a_real_time_gen_ai_customer_service_bot.multimodal.ocr import OCRResult
from a_real_time_gen_ai_customer_service_bot.multimodal.storage import FileStore
from a_real_time_gen_ai_customer_service_bot.orchestrator import MAX_STORED_MESSAGES, Orchestrator
from a_real_time_gen_ai_customer_service_bot.tickets.sla import SLAEngine

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def make_bot(tmp_path, config, clock, name="run", ocr=None):
    kb = tmp_path / "kb"
    if not kb.exists():
        shutil.copytree(PROJECT_ROOT / "data" / "kb_docs", kb)
    return Orchestrator(config, clock, kb_dir=kb, workdir=tmp_path / name, ocr=ocr)


# --- concurrency ---------------------------------------------------------------------------------------------
def test_simultaneous_customers_get_unique_ids_and_isolated_sessions(tmp_path, config, clock):
    bot = make_bot(tmp_path, config, clock)

    def work(i):
        return [bot.handle_message(f"cust{i}", f"My router is broken, order ORD-{100000 + i}, mail u{i}@x.com", region="EU")
                for _ in range(4)]
    with ThreadPoolExecutor(24) as ex:
        results = [r for rs in ex.map(work, range(60)) for r in rs]
    assert not [r for r in results if r.status == "error"]
    ids = [t.ticket_id for t in bot.repo.tickets.values()]
    assert len(ids) == len(set(ids)) == 60                                  # one ticket per customer, no id collisions
    for i in range(60):
        assert bot.sessions.get_active(f"cust{i}").entities["order_ids"] == [f"ORD-{100000 + i}"]
    assert len({r.session_id for r in results}) == 60


def test_escalation_ids_are_unique_under_concurrency(tmp_path, config, clock):
    bot = make_bot(tmp_path, config, clock)
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda i: bot.handle_message(f"c{i}", "I was charged twice for my order"), range(80)))
    ids = [e.id for e in bot.escalation.records]
    assert len(ids) == len(set(ids)) == 80


def test_a_slow_upload_does_not_block_other_customers(tmp_path, config, clock):
    class SlowOcr:
        def extract(self, data, filename):
            time.sleep(1.0)
            return OCRResult("Order ORD-777888 total Rs 100 on 2026-10-05", 0.95)
    config.set("multimodal.processing_timeout_seconds", 5)
    bot = make_bot(tmp_path, config, clock, ocr=SlowOcr())
    slow = threading.Thread(target=lambda: bot.handle_message("slow", "see my invoice", attachments=[("a.png", PNG)]))
    slow.start()
    time.sleep(0.2)                                                         # the slow upload is now being processed
    start = time.perf_counter()
    bot.handle_message("fast", "How many days do I have to return products?", region="EU")
    assert time.perf_counter() - start < 0.7                                 # not stuck behind the 1 s OCR
    slow.join()


def test_background_checks_and_chat_can_run_together(tmp_path, config, clock):
    bot = make_bot(tmp_path, config, clock)
    stop = threading.Event()

    def background():
        while not stop.is_set():
            bot.run_background_checks()
    t = threading.Thread(target=background)
    t.start()
    try:
        with ThreadPoolExecutor(8) as ex:
            results = list(ex.map(lambda i: bot.handle_message(f"c{i}", "This is terrible, my router is broken"), range(40)))
    finally:
        stop.set()
        t.join()
    assert not [r for r in results if r.status == "error"]


# --- uploaded files must obey retention even after a restart ----------------------------------------------------
def test_files_left_by_a_previous_run_are_still_deleted_after_retention(tmp_path, config, clock):
    bot1 = make_bot(tmp_path, config, clock)
    bot1.handle_message("c1", "see invoice", attachments=[("a.txt", b"Order ORD-123456 total Rs 500 on 2026-10-05")])
    uploads = tmp_path / "run" / "uploads"
    assert len(list(uploads.iterdir())) == 1
    for f in uploads.iterdir():                                              # pretend the file is 2 days old on disk
        old = (datetime.now(timezone.utc) - timedelta(days=2)).timestamp()
        os.utime(f, (old, old))
    store = FileStore(uploads, config, type("C", (), {"now": staticmethod(lambda: datetime.now(timezone.utc))})())
    assert len(store.ids()) == 1                                             # adopted after the "restart"
    assert len(store.purge_expired()) == 1 and list(uploads.iterdir()) == []


def test_recent_leftover_files_are_kept_until_their_time(tmp_path, config):
    uploads = tmp_path / "up"
    uploads.mkdir()
    (uploads / "F-abc123.png").write_bytes(b"x")
    store = FileStore(uploads, config, type("C", (), {"now": staticmethod(lambda: datetime.now(timezone.utc))})())
    assert store.purge_expired() == [] and (uploads / "F-abc123.png").exists()


def test_upload_folder_and_files_are_private(tmp_path, config, clock):
    bot = make_bot(tmp_path, config, clock)
    bot.handle_message("c1", "see invoice", attachments=[("a.txt", b"Order ORD-123456 total Rs 500 on 2026-10-05")])
    uploads = tmp_path / "run" / "uploads"
    assert stat.S_IMODE(os.stat(uploads).st_mode) == 0o700
    assert all(stat.S_IMODE(os.stat(f).st_mode) == 0o600 for f in uploads.iterdir())


# --- configuration safety ---------------------------------------------------------------------------------------------
def test_sla_config_must_define_every_tier(config):
    with pytest.raises(ConfigError):
        config.set("sla.response_minutes", {"high": 60})
    assert config.get("sla.response_minutes")["critical"] == 240              # unchanged
    config.set("sla.response_minutes", {"critical": 10, "high": 20, "medium": 30, "low": 40})


def test_sla_engine_survives_a_config_object_without_every_tier(config, calendar, clock):
    data = config.to_dict()
    data["sla"]["response_minutes"] = {"high": 60}
    cfg = Config.__new__(Config)                                              # built WITHOUT validation, on purpose
    cfg._data, cfg._config_path, cfg._holidays_path = data, None, None
    engine = SLAEngine(cfg, calendar, clock, None)
    assert engine.allowed_minutes("low") == 60 and engine.allowed_minutes("critical") == 60


def test_supported_languages_must_be_implemented(config):
    with pytest.raises(ConfigError):
        config.set("languages.supported", ["en", "ja"])
    config.set("languages.supported", ["en", "es"])
    assert config.get("languages.supported") == ["en", "es"]


# --- memory bounds / input limits --------------------------------------------------------------------------------------
def test_metrics_do_not_grow_without_bound(config, clock):
    m = MetricsCollector(config, clock)
    for _ in range(MetricsCollector.MAX_EVENTS + 500):
        m.record_failure("x")
        clock.advance(seconds=1)
    assert len(m._events) <= MetricsCollector.MAX_EVENTS


def test_extremely_long_message_is_handled_quickly(tmp_path, config, clock):
    bot = make_bot(tmp_path, config, clock)
    start = time.perf_counter()
    r = bot.handle_message("c1", "My router is broken and terrible. " * 5000)
    assert r.status != "error" and time.perf_counter() - start < 2.0


def test_very_long_conversations_are_bounded(tmp_path, config, clock):
    bot = make_bot(tmp_path, config, clock)
    for i in range(MAX_STORED_MESSAGES // 2 + 150):
        bot.handle_message("c1", f"hello number {i}")
    conv = bot.sessions.get_active("c1")
    assert len(conv.messages) <= MAX_STORED_MESSAGES + 100 and len(conv.recent(10)) == 10


def test_close_is_safe_to_call_twice(tmp_path, config, clock):
    bot = make_bot(tmp_path, config, clock)
    bot.close()
    bot.close()


# --- API key handling and .env -----------------------------------------------------------------------------------------
def test_resolve_role_matches_exactly():
    keys = {"abc123": "admin", "abc12": "agent"}
    assert resolve_role("abc123", keys) == "admin" and resolve_role("abc12", keys) == "agent"
    for wrong in (None, "", "abc", "abc1234", "ABC123", " abc123"):
        assert resolve_role(wrong, keys) == "customer"


def test_dotenv_file_is_actually_loaded(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text('BOT_API_KEYS={"k1": "admin"}\nBOT_STATE_FILE=/tmp/x.json\n')
    monkeypatch.delenv("BOT_API_KEYS", raising=False)
    monkeypatch.delenv("BOT_STATE_FILE", raising=False)
    assert load_env_file(str(tmp_path / ".env")) is True
    assert os.environ["BOT_API_KEYS"] == '{"k1": "admin"}' and os.environ["BOT_STATE_FILE"] == "/tmp/x.json"
    monkeypatch.delenv("BOT_API_KEYS")
    monkeypatch.delenv("BOT_STATE_FILE")
