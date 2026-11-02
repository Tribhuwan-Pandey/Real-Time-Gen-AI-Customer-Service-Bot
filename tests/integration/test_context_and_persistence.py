"""Task 6 extras: follow-ups that use the conversation, trivial first messages, and surviving a restart."""
import json
import os
import shutil
import stat

import pytest

from a_real_time_gen_ai_customer_service_bot.core.config import PROJECT_ROOT
from a_real_time_gen_ai_customer_service_bot.orchestrator import Orchestrator


def make_bot(tmp_path, config, clock, name="run"):
    kb = tmp_path / "kb"
    if not kb.exists():
        shutil.copytree(PROJECT_ROOT / "data" / "kb_docs", kb)
    return Orchestrator(config, clock, kb_dir=kb, workdir=tmp_path / name)


@pytest.fixture
def bot(tmp_path, config, clock):
    return make_bot(tmp_path, config, clock)


# --- follow-ups that use context ------------------------------------------------------------------------------
def test_region_followup_reuses_the_previous_question(bot):
    first = bot.handle_message("c1", "How many days do I have to return products?", region="EU")
    assert "30 days" in first.text
    second = bot.handle_message("c1", "And in India?")
    assert second.answer.status == "answered" and "10 days" in second.text and second.answer.citations[0]["doc_id"] == "RET-POL-IN"


def test_in_france_does_not_become_the_indian_region(bot):
    bot.handle_message("c1", "How many days do I have to return products?", region="EU")
    second = bot.handle_message("c1", "And in France?")
    assert "10 days" not in second.text


def test_topic_followup_combines_with_the_previous_question(bot):
    bot.handle_message("c1", "How many days do I have to return products?", region="EU")
    r = bot.handle_message("c1", "What about the refund after a return?", region="EU")
    assert r.answer.status == "answered" and "7 business days" in r.text


def test_followup_with_no_new_content_asks_for_clarification(bot):
    bot.handle_message("c1", "How many days do I have to return products?", region="EU")
    r = bot.handle_message("c1", "What about the other one?")
    assert r.status == "clarification" and r.answer is None


def test_followup_without_any_context_asks_for_clarification(bot):
    assert bot.handle_message("c9", "And in India?").status == "clarification"


def test_followups_do_not_leak_between_customers(bot):
    bot.handle_message("a", "How many days do I have to return products?", region="EU")
    assert bot.handle_message("b", "And in India?").status == "clarification"


def test_trivial_first_message_does_not_ask_for_a_language(bot):
    for text in ("ok", "hmm", "???"):
        r = bot.handle_message(f"t-{text}", text)
        assert r.status == "clarification" and "Which language" not in r.text


def test_long_unrecognised_text_still_asks_which_language(bot):
    r = bot.handle_message("t9", "Xyzzy plugh frobnicate quuxwhatever blorpity")
    assert r.status == "clarification" and "language" in r.text.lower()


# --- persistence across a restart ----------------------------------------------------------------------------------
def test_restart_keeps_tickets_escalations_and_restores_returning_customer(tmp_path, config, clock):
    bot1 = make_bot(tmp_path, config, clock)
    r = bot1.handle_message("c1", "Charged twice on card 4111 1111 1111 1111, ORD-123456, mail jane@example.com, phone 9876543210")
    tid, esc_id = r.ticket.ticket_id, r.escalations[0].id
    state = tmp_path / "state.json"
    bot1.save_state(state)

    clock.advance(minutes=45)                                   # the process was down for 45 minutes
    bot2 = make_bot(tmp_path, config, clock, name="run2")
    restored = bot2.load_state(state)
    assert restored["tickets"] == 1 and restored["escalations"] == 1 and restored["sessions"] == 1

    assert [t.ticket_id for t in bot2.list_tickets("agent")] == [tid]
    assert bot2.escalation.records[0].id == esc_id and bot2.escalation.records[0].rule == "high_risk:duplicate_payment"
    back = bot2.handle_message("c1", "Hello again")
    assert back.session_event == "restored" and "Welcome back" in back.text and "duplicate_payment" in back.text


def test_ids_continue_after_restart_and_duplicates_are_still_detected(tmp_path, config, clock):
    bot1 = make_bot(tmp_path, config, clock)
    bot1.handle_message("c1", "My router keeps disconnecting every hour. ORD-111222 jane@example.com")
    state = tmp_path / "state.json"
    bot1.save_state(state)
    bot2 = make_bot(tmp_path, config, clock, name="run2")
    bot2.load_state(state)
    dup = bot2.handle_message("c1", "My router keeps disconnecting every hour. ORD-111222 jane@example.com")
    assert dup.ticket.ticket_id == "TCK-0001" and dup.ticket.duplicate_count == 1
    other = bot2.handle_message("c2", "Where is my parcel? ORD-333444 bob@example.com")
    assert other.ticket.ticket_id == "TCK-0002"                         # no id collision


def test_sla_status_survives_a_restart(tmp_path, config, clock):
    bot1 = make_bot(tmp_path, config, clock)
    bot1.handle_message("c1", "My router keeps disconnecting. ORD-111222 jane@example.com")
    t1 = next(iter(bot1.repo.tickets.values()))
    deadline = bot1.sla.deadline(t1)
    state = tmp_path / "state.json"
    bot1.save_state(state)
    bot2 = make_bot(tmp_path, config, clock, name="run2")
    bot2.load_state(state)
    t2 = bot2.repo.get(t1.ticket_id)
    assert bot2.sla.deadline(t2) == deadline and t2.created_at == t1.created_at and t2.sla_priority == t1.sla_priority


def test_restored_session_after_24_hours_starts_fresh(tmp_path, config, clock):
    bot1 = make_bot(tmp_path, config, clock)
    bot1.handle_message("c1", "My router keeps disconnecting. ORD-111222 jane@example.com")
    state = tmp_path / "state.json"
    bot1.save_state(state)
    clock.advance(hours=25)
    bot2 = make_bot(tmp_path, config, clock, name="run2")
    bot2.load_state(state)
    assert bot2.handle_message("c1", "Hello").session_event == "new_after_restore_window"


def test_state_file_contains_no_raw_personal_or_payment_data_in_sessions_and_has_private_permissions(tmp_path, config, clock):
    bot = make_bot(tmp_path, config, clock)
    bot.handle_message("c1", "Charged twice on card 4111 1111 1111 1111, ORD-123456, mail jane@example.com phone 9876543210")
    state = tmp_path / "state.json"
    bot.save_state(state)
    sessions = json.dumps(json.loads(state.read_text())["sessions"])
    for secret in ("4111 1111", "jane@example.com", "9876543210"):
        assert secret not in sessions
    assert "ORD-123456" in sessions
    assert stat.S_IMODE(os.stat(state).st_mode) == 0o600


def test_corrupt_or_unknown_version_state_is_ignored_not_fatal(tmp_path, config, clock):
    bot = make_bot(tmp_path, config, clock)
    bad = tmp_path / "state.json"
    bad.write_text("{ this is not json")
    assert bot.load_state(bad) is None
    bad.write_text(json.dumps({"version": 99}))
    assert bot.load_state(bad) is None
    assert bot.load_state(tmp_path / "missing.json") is None
    assert bot.handle_message("c1", "Hello").status == "ok"


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path, config, clock):
    bot = make_bot(tmp_path, config, clock)
    bot.handle_message("c1", "Where is my parcel? ORD-333444 bob@example.com")
    state = tmp_path / "out" / "state.json"
    bot.save_state(state)
    bot.save_state(state)
    assert [p.name for p in state.parent.iterdir()] == ["state.json"]


def test_api_saves_state_on_shutdown_and_loads_it_on_start(tmp_path, config, clock):
    from fastapi.testclient import TestClient
    from a_real_time_gen_ai_customer_service_bot.main import create_app
    state = str(tmp_path / "api_state.json")
    bot1 = make_bot(tmp_path, config, clock)
    with TestClient(create_app(bot1, {"k": "agent"}, background_interval_seconds=None, state_path=state)) as c:
        c.post("/chat", json={"customer_id": "c1", "message": "Where is my parcel? ORD-333444 bob@example.com"})
    assert os.path.exists(state)                                            # saved when the app shut down
    bot2 = make_bot(tmp_path, config, clock, name="run2")
    with TestClient(create_app(bot2, {"k": "agent"}, background_interval_seconds=None, state_path=state)) as c:
        tickets = c.get("/tickets", headers={"X-API-Key": "k"}).json()["tickets"]
    assert len(tickets) == 1 and tickets[0]["order_id"] == "ORD-333444"
