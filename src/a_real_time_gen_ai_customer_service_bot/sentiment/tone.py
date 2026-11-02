"""Tone adaptation. Only a courteous prefix changes with sentiment; the policy-bearing body is untouched."""
from .analyzer import SentimentResult

_PREFIX = {
    "en": {"negative": "I'm sorry about the trouble you've had. ", "frustrated": "I understand how frustrating this has been, and I'm sorry. ",
           "urgent": "I understand this is urgent, so I'll prioritise it. ", "sarcastic": "I'm sorry we've let you down. ",
           "positive": "Glad to help! ", "neutral": ""},
    "hi": {"negative": "असुविधा के लिए मुझे खेद है। ", "frustrated": "मैं समझता हूँ कि यह कितना निराशाजनक रहा है, क्षमा करें। ",
           "urgent": "मैं समझता हूँ कि यह जरूरी है, इसे प्राथमिकता दूँगा। ", "sarcastic": "हमारी वजह से आपको निराशा हुई, क्षमा करें। ",
           "positive": "मदद करके खुशी हुई! ", "neutral": ""},
    "es": {"negative": "Lamento los inconvenientes. ", "frustrated": "Entiendo lo frustrante que ha sido y lo lamento. ",
           "urgent": "Entiendo que es urgente, lo priorizaré. ", "sarcastic": "Lamento haberle fallado. ",
           "positive": "¡Me alegra poder ayudar! ", "neutral": ""},
    "fr": {"negative": "Je suis désolé pour ce désagrément. ", "frustrated": "Je comprends à quel point c'est frustrant, et je suis désolé. ",
           "urgent": "Je comprends que c'est urgent, je vais le traiter en priorité. ", "sarcastic": "Je suis désolé de vous avoir déçu. ",
           "positive": "Ravi de pouvoir aider ! ", "neutral": ""},
}


def tone_prefix(sentiment: SentimentResult | None, language: str) -> str:
    if sentiment is None:
        return ""
    table = _PREFIX.get(language, _PREFIX["en"])
    return table.get(sentiment.label, "")


def adapt_tone(body: str, sentiment: SentimentResult | None, language: str) -> str:
    """Prefix the body with a tone-appropriate sentence. The body text is never altered."""
    return tone_prefix(sentiment, language) + body
