"""Sarcasm benchmark: unseen-style sentences in four languages, plus genuine praise that must NOT be flagged."""
import pytest

from a_real_time_gen_ai_customer_service_bot.sentiment.analyzer import SentimentAnalyzer

SARCASTIC = [
    # English
    "Wow, what a great way to waste my afternoon.",
    "Thanks a lot for keeping me on hold for an hour.",
    "Fantastic, my order is lost again.",
    "Oh brilliant, the app crashed once more.",
    "Great job breaking it even more.",
    "Yeah, because waiting three weeks is totally normal.",
    "I just love paying twice for the same thing.",
    "Sure, take your time, it's not like I need the product.",
    "Excellent service, only took five attempts to get a reply.",
    "Nice work, you've managed to lose my package again.",
    "Thanks for the amazing support, I'm still waiting for an answer.",
    "Can't wait to wait another week.",
    "Such a wonderful experience, my router died in two days.",
    "Congratulations on making the checkout impossible.",
    "Bravo, another delayed delivery.",
    "Good luck getting any help from this company.",
    "Oh perfect, now the screen is flickering too.",
    "Really helpful, you answered a question I never asked.",
    "So glad I chose you guys, my package arrived broken.",
    "Love how nobody ever replies here \U0001f644",
    # Spanish
    "Qué maravilla, otro retraso más.",
    "Excelente servicio, llevo dos semanas esperando.",
    "Gracias por nada, de verdad.",
    "Genial, ahora me cobraron de nuevo.",
    # French
    "Super, encore une livraison en retard.",
    "Merci pour rien.",
    "Quel service formidable, mon colis est cassé.",
    "Bravo, ils ont encore perdu ma commande.",
    # Hindi (romanised and Devanagari)
    "Waah, kya shandar service hai, phir se der ho gayi.",
    "Bahut badhiya, order phir se nahi aaya.",
    "Shukriya, bahut madad ki, abhi tak kuch nahi mila.",
    "वाह, क्या शानदार सर्विस है, फिर से देरी हो गई।",
    "बहुत बढ़िया, ऑर्डर फिर से नहीं आया।",
]

GENUINE_PRAISE = [
    "Thank you so much, the replacement arrived quickly and works perfectly.",
    "Great service, my issue was solved in five minutes.",
    "Excellent, thanks for the quick refund!",
    "Gracias, el pedido llegó perfecto.",
    "Merci beaucoup, tout est parfait.",
    "Shukriya, mera order aa gaya.",
    "Love the new design, it works great.",
    "Thanks, that fixed it.",
    "Awesome, it works now.",
    "Great, thanks for your help.",
]


# Genuine messages that contain risky words ("another", "again", "waiting", "late", "broken", "twice", "delay").
HARD_GENUINE = [
    "I'll order another one, thanks!", "Thanks, it works again!", "I love this product, will buy again.",
    "Thank you for waiting on my behalf.", "Great news, the delivery was not late at all, thanks!",
    "Amazing, my package arrived a day early.", "Thanks so much for the quick reply, very helpful.",
    "I was waiting for this, thank you!", "Perfect, that is exactly what I needed.",
    "Wonderful, the broken screen was replaced for free. Thank you!", "Super, merci de votre aide rapide.",
    "Excelente, gracias por la ayuda rápida.", "Bahut badhiya, ab sab theek ho gaya.",
    "धन्यवाद, आपकी मदद बहुत अच्छी थी।", "Nice, I will also take a second one.", "Bravo for the fast support team!",
    "Thanks, the issue was fixed and everything works now.", "Good job, the new update fixed the crash.",
    "Glad the delay is over, thanks.", "Great, I received the refund twice as fast as expected.",
]


@pytest.fixture
def analyzer(config):
    return SentimentAnalyzer(config)


@pytest.mark.parametrize("text", SARCASTIC, ids=[t[:50] for t in SARCASTIC])
def test_sarcasm_is_detected(analyzer, text):
    r = analyzer.analyze(text)
    assert r.label == "sarcastic" and r.negative_like, (r.label, r.scores)


@pytest.mark.parametrize("text", GENUINE_PRAISE, ids=[t[:50] for t in GENUINE_PRAISE])
def test_genuine_praise_is_not_flagged(analyzer, text):
    r = analyzer.analyze(text)
    assert r.label != "sarcastic" and not r.negative_like, (r.label, r.scores)


@pytest.mark.parametrize("text", HARD_GENUINE, ids=[t[:50] for t in HARD_GENUINE])
def test_genuine_messages_with_risky_words_are_not_flagged(analyzer, text):
    r = analyzer.analyze(text)
    assert r.label != "sarcastic", (r.label, r.scores, r.signals)


def test_sarcasm_in_context_of_earlier_complaints(analyzer):
    h = [analyzer.analyze("My order is late again"), analyzer.analyze("Still nothing, this is unacceptable")]
    r = analyzer.analyze("Great, thanks for the help", history=h)
    assert r.negative_like and r.scores["sarcastic"] >= 0.5        # praise after two complaints reads as sarcasm


def test_pass_rate_summary(analyzer):
    hit = sum(analyzer.analyze(t).label == "sarcastic" for t in SARCASTIC)
    clean = sum(analyzer.analyze(t).label != "sarcastic" for t in GENUINE_PRAISE)
    print(f"sarcasm recall {hit}/{len(SARCASTIC)}, genuine-praise precision {clean}/{len(GENUINE_PRAISE)}")
    assert hit / len(SARCASTIC) >= 0.9 and clean == len(GENUINE_PRAISE)
