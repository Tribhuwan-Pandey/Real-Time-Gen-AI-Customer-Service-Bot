import pytest

from a_real_time_gen_ai_customer_service_bot.language.detector import LanguageDetector
from a_real_time_gen_ai_customer_service_bot.language.entities import (
    extract_entities, find_dates, has_correction, merge_entities, parse_date)
from a_real_time_gen_ai_customer_service_bot.language.intents import detect_intents, intent_names
from datetime import date


# --- entities: exact preservation -----------------------------------------
def test_entities_preserved_exactly():
    e = extract_entities("Hi, I'm reaching out about ORD-12345678 for product XR-2000 bought on 2026-10-12, "
                         "paid Rs 1,500. Error E1002. Mail: a.b@x.com")
    assert e["order_ids"] == ["ORD-12345678"] and e["product_codes"] == ["XR-2000"]
    assert e["dates"] == ["2026-10-12"] and e["amounts"] == ["Rs 1,500"]
    assert e["error_codes"] == ["E1002"] and e["emails"] == ["a.b@x.com"]


def test_order_number_in_other_languages_and_context():
    assert extract_entities("mi pedido número 556677 no llegó")["order_ids"] == ["556677"]
    assert extract_entities("ma commande numéro 889900")["order_ids"] == ["889900"]
    assert extract_entities("order #44556")["order_ids"] == ["44556"]


def test_names_in_several_languages():
    assert extract_entities("My name is Priya Sharma")["names"] == ["Priya Sharma"]
    assert extract_entities("mera naam Rahul hai")["names"] == ["Rahul"]
    assert extract_entities("me llamo Sofía")["names"] == ["Sofía"]


def test_nothing_invented_when_absent():
    e = extract_entities("hello there")
    assert all(v == [] for v in e.values())


def test_dates_parsing_multilingual():
    assert parse_date("2026-03-05") == date(2026, 3, 5)
    assert parse_date("05/03/2026") == date(2026, 3, 5)
    assert parse_date("5 March 2026") == date(2026, 3, 5)
    assert parse_date("March 5, 2026") == date(2026, 3, 5)
    assert parse_date("5 de marzo de 2026") == date(2026, 3, 5)
    assert parse_date("5 février 2026") == date(2026, 2, 5)
    assert parse_date("31/02/2026") is None
    assert find_dates("policy on 5 March 2026 please") == [date(2026, 3, 5)]


def test_correction_replaces_old_value():
    store = {}
    merge_entities(store, extract_entities("my order is ORD-1111"), False)
    corrected = merge_entities(store, extract_entities("sorry, I meant ORD-2222"), has_correction("sorry, I meant ORD-2222"))
    assert store["order_ids"] == ["ORD-2222"] and corrected == ["order_ids"]


def test_without_correction_second_value_is_added():
    store = {}
    merge_entities(store, extract_entities("ORD-1111"), False)
    merge_entities(store, extract_entities("also ORD-2222"), False)
    assert store["order_ids"] == ["ORD-1111", "ORD-2222"]


# --- language detection ------------------------------------------------------
@pytest.mark.parametrize("text,lang", [
    ("I have not received my order and I want a refund", "en"),
    ("Mera order abhi tak nahi aaya hai, kya aap madad kar sakte hain", "hi"),
    ("मेरा ऑर्डर अभी तक नहीं आया है", "hi"),
    ("Hola, no he recibido mi pedido y quiero un reembolso", "es"),
    ("Bonjour, je n'ai pas reçu ma commande et je veux un remboursement", "fr"),
])
def test_language_detected(config, text, lang):
    r = LanguageDetector(config).detect(text)
    assert r.language == lang and r.supported and r.confidence >= config.get("confidence.language_min")


def test_romanised_hindi_flagged_transliterated(config):
    r = LanguageDetector(config).detect("Mera paisa do baar kat gaya hai")
    assert r.language == "hi" and r.transliterated


def test_mixed_language_message(config):
    r = LanguageDetector(config).detect("Mera order ORD-123456 nahi aaya, I want a refund please")
    assert r.mixed and {"hi", "en"} <= set(r.languages_present)


def test_unsupported_language_recognised(config):
    r = LanguageDetector(config).detect("Ich habe meine Bestellung nicht und bitte hallo")
    assert r.language == "de" and not r.supported


def test_unknown_or_tiny_input_has_zero_confidence(config):
    for text in ("ok", "???", "ORD-123456", ""):
        r = LanguageDetector(config).detect(text)
        assert r.language is None and r.confidence == 0


def test_supported_languages_are_configurable(config):
    config.set("languages.supported", ["en", "es"])
    assert not LanguageDetector(config).detect("Bonjour, je n'ai pas reçu ma commande").supported
    config.set("languages.supported", ["en", "es", "fr"])
    assert LanguageDetector(config).detect("Bonjour, je n'ai pas reçu ma commande").supported


# --- intents ------------------------------------------------------------------
@pytest.mark.parametrize("text,intent", [
    ("I was charged twice for my order", "duplicate_payment"),
    ("Meri payment do baar kat gayi", "duplicate_payment"),
    ("Me cobraron dos veces el pedido", "duplicate_payment"),
    ("J'ai été débité deux fois", "duplicate_payment"),
    ("someone accessed my account and changed the password", "account_compromise"),
    ("mi cuenta fue hackeada", "account_compromise"),
    ("mon compte a été piraté", "account_compromise"),
    ("मेरा अकाउंट हैक हो गया", "account_compromise"),
    ("I will take this to court and call my lawyer", "legal_threat"),
    ("voy a contratar un abogado", "legal_threat"),
    ("je vais appeler mon avocat", "legal_threat"),
    ("I want a refund", "refund"),
    ("where is my order", "order_status"),
    ("the router keeps disconnecting", "technical_issue"),
    ("what is the return policy?", "policy_question"),
    ("thanks, it's fixed now", "resolved"),
])
def test_intent_detected(text, intent):
    assert intent in intent_names(detect_intents(text))


def test_spelling_errors_tolerated():
    assert "duplicate_payment" in intent_names(detect_intents("I was chargd twise for the same ordr"))
    assert "refund" in intent_names(detect_intents("i need a refnd plzzz"))
    assert "account_compromise" in intent_names(detect_intents("my acount was hakced"))


def test_multiple_requests_in_one_message():
    names = intent_names(detect_intents("I want a refund and also cancel my order, plus where is my parcel?"))
    assert {"refund", "cancel_order", "order_status"} <= set(names)


def test_calm_high_risk_still_detected():
    assert "duplicate_payment" in intent_names(detect_intents("Hello, just a note that I appear to have been charged twice. Thank you."))


def test_issue_is_not_sue():
    assert "legal_threat" not in intent_names(detect_intents("I have an issue with my order"))


def test_unknown_message_has_no_intent():
    assert detect_intents("the blue one, please") == []
