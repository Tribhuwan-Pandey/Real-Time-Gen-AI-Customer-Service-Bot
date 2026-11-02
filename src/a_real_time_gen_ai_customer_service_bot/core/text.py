"""Small text helpers shared by language, sentiment, intent and security code."""
import re
import unicodedata

_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_COMBINING_LATIN = re.compile("[\u0300-\u036f]")
_WS = re.compile(r"\s+")


def fold(text: str) -> str:
    """Lower-case, strip Latin accents (but keep Devanagari marks), drop zero-width chars."""
    text = _ZERO_WIDTH.sub("", text)
    text = unicodedata.normalize("NFKD", text)
    text = _COMBINING_LATIN.sub("", text)
    text = unicodedata.normalize("NFC", text).lower()
    return _WS.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u0900-\u097f]+", fold(text))
