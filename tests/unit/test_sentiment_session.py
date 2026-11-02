
import pytest

from a_real_time_gen_ai_customer_service_bot.sentiment.analyzer import SentimentAnalyzer
from a_real_time_gen_ai_customer_service_bot.sentiment.tone import adapt_tone
from a_real_time_gen_ai_customer_service_bot.session.manager import SessionManager
from a_real_time_gen_ai_customer_service_bot.session.models import Message


@pytest.fixture
def analyzer(config):
    return SentimentAnalyzer(config)


@pytest.mark.parametrize("text,label", [
    ("Thanks so much, that was really helpful and perfect!", "positive"),
    ("Where can I find my invoice?", "neutral"),
    ("This is terrible service and the product is broken", "negative"),
    ("This is the third time I'm asking and nobody helps. Ridiculous!!", "frustrated"),
    ("This is URGENT, I need it fixed immediately", "urgent"),
    ("Oh great, another delay. Just wonderful.", "sarcastic"),
    ("Thanks for nothing.", "sarcastic"),
    ("Wow, amazing service, I've only been waiting for three weeks", "sarcastic"),
    ("gracias por nada", "sarcastic"),
    ("merci pour rien", "sarcastic"),
    ("Esto es pésimo, estoy muy enojado", "negative"),
    ("Mera order bahut kharab hai, bekar service", "negative"),
    ("यह बहुत खराब सेवा है", "negative"),
    ("C'est nul, je suis furieux", "negative"),
    ("Merci beaucoup, c'est parfait", "positive"),
])
def test_primary_labels_across_languages(analyzer, text, label):
    assert analyzer.analyze(text).label == label


def test_sarcasm_is_not_positive_even_with_positive_words(analyzer):
    r = analyzer.analyze("Great, another broken delivery. Love that.")
    assert r.label == "sarcastic" and r.negative_like and r.scores["positive"] < 0.3


def test_genuine_thanks_is_not_sarcastic(analyzer):
    r = analyzer.analyze("Thank you so much, my refund arrived. Excellent service!")
    assert r.label == "positive" and not r.negative_like


def test_confidence_scores_present_and_bounded(analyzer):
    r = analyzer.analyze("This is terrible!!")
    assert set(r.scores) == {"positive", "neutral", "negative", "frustrated", "urgent", "sarcastic"}
    assert all(0 <= v <= 1 for v in r.scores.values()) and 0 <= r.confidence <= 1


def test_history_raises_negative_scores(analyzer):
    h1 = analyzer.analyze("This is terrible, the product is broken")
    h2 = analyzer.analyze("Still broken, awful")
    alone = analyzer.analyze("It is broken")
    with_history = analyzer.analyze("It is broken", history=[h1, h2])
    assert with_history.scores["negative"] > alone.scores["negative"]


def test_short_followup_carries_over_frustration(analyzer):
    h = analyzer.analyze("This is the third time, nobody helps, ridiculous!!")
    r = analyzer.analyze("so?", history=[h])
    assert r.scores["frustrated"] > analyzer.analyze("so?").scores["frustrated"]


def test_empty_message_is_neutral(analyzer):
    r = analyzer.analyze("")
    assert r.label == "neutral" and not r.negative_like


def test_low_confidence_flag_follows_config(analyzer, config):
    assert not analyzer.analyze("It is broken").low_confidence          # confident enough at the default threshold
    config.set("confidence.sentiment_min", 0.99)
    assert analyzer.analyze("It is broken").low_confidence


def test_tone_changes_prefix_but_never_the_body(analyzer):
    body = "Refunds are available within 30 days. [Source: RET-1]"
    outs = {t: adapt_tone(body, analyzer.analyze(t), "en") for t in
            ("thanks!", "this is terrible!!", "where is my invoice?")}
    assert all(o.endswith(body) for o in outs.values())
    assert len(set(outs.values())) == 3          # tone differs
    assert adapt_tone(body, analyzer.analyze("terrible"), "hi") != body


# --- sessions -----------------------------------------------------------------
@pytest.fixture
def sessions(config, clock):
    return SessionManager(config, clock)


def test_new_then_continued(sessions, clock):
    c1, e1 = sessions.get_or_create("cust-1")
    clock.advance(minutes=5)
    c2, e2 = sessions.get_or_create("cust-1")
    assert (e1, e2) == ("new", "continued") and c1 is c2


def test_customers_are_isolated(sessions):
    a, _ = sessions.get_or_create("a")
    b, _ = sessions.get_or_create("b")
    a.entities["order_ids"] = ["ORD-1111"]
    assert a is not b and b.entities == {} and a.session_id != b.session_id


def test_session_ends_after_timeout_and_summary_restored_within_24h(sessions, clock):
    conv, _ = sessions.get_or_create("c")
    conv.language = "hi"
    conv.messages.append(Message("user", "my order ORD-1111 is missing", clock.now(), "hi", "negative", True, ["order_status"]))
    sessions.touch(conv)
    clock.advance(minutes=31)
    new, event = sessions.get_or_create("c")
    assert event == "restored" and new is not conv and conv.end_reason == "inactivity"
    assert "order_status" in new.restored_summary and new.messages == [] and new.language == "hi"


def test_exact_boundaries(sessions, clock):
    sessions.get_or_create("c")
    clock.advance(minutes=29, seconds=59)
    assert sessions.get_or_create("c")[1] == "continued"
    clock.advance(minutes=30)
    assert sessions.get_or_create("c")[1] == "restored"


def test_new_session_after_24_hours(sessions, clock):
    sessions.get_or_create("c")
    clock.advance(hours=24, minutes=1)
    conv, event = sessions.get_or_create("c")
    assert event == "new_after_restore_window" and conv.restored_summary is None


def test_returns_exactly_at_24h_is_restored(sessions, clock):
    sessions.get_or_create("c")
    clock.advance(hours=24)
    assert sessions.get_or_create("c")[1] == "restored"


def test_session_timings_are_configurable(sessions, config, clock):
    config.set("session.timeout_minutes", 5)
    config.set("session.restore_hours", 1)
    sessions.get_or_create("c")
    clock.advance(minutes=6)
    assert sessions.get_or_create("c")[1] == "restored"
    clock.advance(hours=2)
    assert sessions.get_or_create("c")[1] == "new_after_restore_window"


def test_sweep_ends_idle_sessions_only(sessions, clock):
    sessions.get_or_create("idle")
    clock.advance(minutes=20)
    sessions.get_or_create("busy")
    clock.advance(minutes=15)
    ended = sessions.sweep()
    assert [c.customer_id for c in ended] == ["idle"]
    assert sessions.get_active("busy") is not None


def test_context_keeps_at_least_ten_messages(sessions, clock, config):
    conv, _ = sessions.get_or_create("c")
    for i in range(25):
        conv.messages.append(Message("user", f"m{i}", clock.now()))
    recent = conv.recent(config.get("session.context_messages"))
    assert len(recent) >= 10 and recent[-1].text == "m24"


def test_empty_customer_id_rejected(sessions):
    with pytest.raises(ValueError):
        sessions.get_or_create("")
