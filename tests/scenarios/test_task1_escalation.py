"""Task 1 hidden-test style scenarios: sarcasm, calm high-risk, after-hours, simulated time."""
import pytest

from a_real_time_gen_ai_customer_service_bot.core.business_calendar import BusinessCalendar
from a_real_time_gen_ai_customer_service_bot.escalation.engine import EscalationEngine
from a_real_time_gen_ai_customer_service_bot.language.intents import detect_intents
from a_real_time_gen_ai_customer_service_bot.sentiment.analyzer import SentimentAnalyzer
from a_real_time_gen_ai_customer_service_bot.session.manager import SessionManager
from a_real_time_gen_ai_customer_service_bot.session.models import Message
from conftest import ist


class Harness:
    def __init__(self, config, clock):
        self.clock, self.config = clock, config
        self.sessions = SessionManager(config, clock)
        self.analyzer = SentimentAnalyzer(config)
        self.engine = EscalationEngine(config, BusinessCalendar(config), clock)
        self.conv, _ = self.sessions.get_or_create("cust-1")

    def send(self, text):
        sentiment = self.analyzer.analyze(text, self.conv.sentiment_history)
        intents = detect_intents(text)
        self.conv.sentiment_history.append(sentiment)
        self.conv.messages.append(Message("user", text, self.clock.now(), "en", sentiment.label,
                                          sentiment.negative_like, [i.name for i in intents]))
        self.sessions.touch(self.conv)
        return self.engine.evaluate(self.conv, intents, sentiment)


@pytest.fixture
def h(config, clock):
    return Harness(config, clock)           # Monday 11:00 IST = open


# --- calm high-risk -------------------------------------------------------------
@pytest.mark.parametrize("text,rule", [
    ("Hello, I noticed I was charged twice for my order. Please look into it when you can. Thank you!", "high_risk:duplicate_payment"),
    ("Hi, it seems someone accessed my account and changed the password. Kindly help.", "high_risk:account_compromise"),
    ("Good afternoon. I would like to inform you that I will be speaking to my lawyer about this.", "high_risk:legal_threat"),
    ("Hola, me cobraron dos veces, gracias", "high_risk:duplicate_payment"),
    ("Bonjour, mon compte a été piraté, merci", "high_risk:account_compromise"),
])
def test_calm_high_risk_escalates(h, text, rule):
    out = h.send(text)
    assert out.escalated and out.records[0].rule == rule and out.records[0].queue == "agent_queue"


def test_high_risk_escalates_once_per_conversation(h):
    assert h.send("I was charged twice").escalated
    assert not h.send("I was charged twice, again, please check").escalated


# --- sarcasm ---------------------------------------------------------------------
def test_three_sarcastic_messages_count_as_repeated_negative(h):
    assert not h.send("Oh great, another delay. Just wonderful.").escalated
    assert not h.send("Thanks for nothing.").escalated
    out = h.send("Wow, amazing service, only three weeks late")
    assert out.escalated and out.records[0].rule == "repeated_negative"


def test_genuine_thanks_never_escalates(h):
    for t in ("Thanks, that was great!", "Perfect, thank you so much", "Excellent service"):
        assert not h.send(t).records


# --- repeated negative, boundary and configuration ----------------------------------
def test_two_negatives_do_not_escalate_three_do(h):
    assert not h.send("This is terrible, the product is broken").escalated
    assert not h.send("Awful service, still broken").escalated
    assert h.send("Worst experience ever, useless").escalated


def test_a_neutral_message_breaks_the_streak(h):
    h.send("This is terrible, the product is broken")
    h.send("Awful service, still broken")
    h.send("What are your opening hours?")
    assert not h.send("Worst experience ever, useless").escalated


def test_repeated_negative_threshold_is_configurable(h, config):
    config.set("escalation.repeated_negative_count", 2)
    h.send("This is terrible, the product is broken")
    assert h.send("Awful service, still broken").escalated


# --- 15-minute unresolved rule -----------------------------------------------------------
def test_unresolved_negative_escalates_only_after_more_than_15_minutes(h, clock):
    h.send("This is terrible, the product is broken")
    clock.advance(minutes=15)
    assert h.engine.check_timeouts(h.sessions.active_conversations()) == []     # exactly 15: not yet
    clock.advance(seconds=1)
    recs = h.engine.check_timeouts(h.sessions.active_conversations())
    assert len(recs) == 1 and recs[0].rule == "unresolved_negative"
    assert h.engine.check_timeouts(h.sessions.active_conversations()) == []      # fires once


def test_timeout_also_fires_when_customer_writes_again(h, clock):
    h.send("This is terrible, the product is broken")
    clock.advance(minutes=16)
    out = h.send("Any update?")
    assert any(r.rule == "unresolved_negative" for r in out.records)


def test_resolution_cancels_timeout(h, clock):
    h.send("This is terrible, the product is broken")
    clock.advance(minutes=10)
    h.send("Thanks, it's fixed now")
    clock.advance(minutes=30)
    assert h.engine.check_timeouts(h.sessions.active_conversations()) == []


def test_timeout_minutes_configurable(h, clock, config):
    config.set("escalation.unresolved_negative_minutes", 5)
    h.send("This is terrible, the product is broken")
    clock.advance(minutes=6)
    assert h.engine.check_timeouts(h.sessions.active_conversations())


# --- after-hours routing & simulated time --------------------------------------------------
def test_urgent_complaint_after_hours_goes_to_on_call(h, clock):
    clock.set(ist(12, 22, 30))
    out = h.send("URGENT: my account shows a wrong charge and I need this fixed immediately, terrible")
    assert out.decision == "on_call_queue" and out.records[0].queue == "on_call"


def test_normal_complaint_after_hours_scheduled_next_working_day(h, clock):
    clock.set(ist(12, 22, 30))                          # Monday night
    out = h.send("The product I received is broken and the packaging was terrible")
    assert out.decision == "scheduled_next_working_day"
    assert out.records[0].kind == "scheduled" and out.records[0].scheduled_for == ist(13, 9)


def test_friday_evening_complaint_scheduled_for_monday(h, clock):
    clock.set(ist(16, 19, 0))
    out = h.send("The product I received is broken and the packaging was terrible")
    assert out.records[0].scheduled_for == ist(19, 9)


def test_holiday_pushes_schedule_forward(h, clock):
    clock.set(ist(19, 20, 0))                           # Monday evening, Tuesday 20 Oct is a holiday
    out = h.send("The product I received is broken and the packaging was terrible")
    assert out.records[0].scheduled_for == ist(21, 9)


def test_same_message_different_time_different_routing(config, clock):
    clock.set(ist(16, 17, 30))
    a = Harness(config, clock).send("This is terrible, broken, useless!! URGENT")
    clock.set(ist(16, 18, 30))
    b = Harness(config, clock).send("This is terrible, broken, useless!! URGENT")
    assert a.decision == "handle_now" and a.records == []          # open: handled immediately
    assert b.decision == "on_call_queue" and b.records[0].queue == "on_call"   # closed: on-call


def test_calm_high_risk_after_hours_goes_on_call(h, clock):
    clock.set(ist(17, 10, 0))                           # Saturday
    out = h.send("Hello, someone accessed my account without my permission, thanks")
    assert out.records[0].queue == "on_call"


def test_changing_business_hours_at_runtime_changes_routing(h, clock, config):
    clock.set(ist(12, 19, 0))
    config.set("business.hours.end", "20:00")
    out = h.send("The product I received is broken and terrible")
    assert out.decision == "handle_now" and out.records == []


def test_non_complaint_after_hours_is_not_scheduled(h, clock):
    clock.set(ist(12, 23, 0))
    assert h.send("What is your return policy?").records == []


# --- record contents -------------------------------------------------------------------------
def test_record_has_reason_rule_timestamp_summary_metadata(h, clock):
    h.conv.entities["order_ids"] = ["ORD-12345678"]
    out = h.send("I was charged twice. Contact me at jane@example.com or 9876543210")
    r = out.records[0]
    assert r.reason and r.rule == "high_risk:duplicate_payment" and r.timestamp == clock.now()
    assert r.conversation_id == h.conv.session_id and r.customer_id == "cust-1"
    assert "ORD-12345678" in r.summary
    assert "jane@example.com" not in r.summary and "9876543210" not in r.summary
    assert r.metadata["intents"] and r.metadata["sentiment"]
    assert h.engine.records == [r]
