"""Masking of personal / payment / credential data.

Used on every log line, summary, handoff note and debug output.
Order IDs and product codes are NOT masked (the bot must preserve them exactly).
"""
import re
from typing import Any

_SECRET_WORD = re.compile(
    r"(?i)\b(password|passwd|pwd|pin|otp|cvv|cvc|secret|token|api[_ -]?key)\b(\s*(?:is|:|=)?\s*)(\S+)")
_SK_KEY = re.compile(r"\bsk-[A-Za-z0-9_\-]{10,}")
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_AADHAAR = re.compile(r"(?<![\w-])\d{4}[ -]\d{4}[ -]\d{4}(?![\w-])")
_PHONE_INTL = re.compile(r"(?<![\w-])\+\d{1,3}[ -]?\(?\d{2,5}\)?[ -]?\d{3,5}[ -]?\d{3,5}(?![\w-])")
_PHONE_IN = re.compile(r"(?<![\w-])[6-9]\d{9}(?![\w-])")
_PHONE_US = re.compile(r"(?<![\w-])\(?\d{3}\)?[ -]\d{3}[ -]\d{4}(?![\w-])")


def _mask_card(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return f"[CARD ****{digits[-4:]}]"


def mask_text(text: str) -> str:
    """Return text with sensitive values replaced by placeholders."""
    if not text:
        return text
    text = _SECRET_WORD.sub(lambda m: f"{m.group(1)}{m.group(2)}[SECRET]", text)
    text = _SK_KEY.sub("[SECRET]", text)
    text = _EMAIL.sub("[EMAIL]", text)
    text = _CARD.sub(_mask_card, text)
    text = _AADHAAR.sub("[ID-NUMBER]", text)
    text = _PHONE_INTL.sub("[PHONE]", text)
    text = _PHONE_IN.sub("[PHONE]", text)
    text = _PHONE_US.sub("[PHONE]", text)
    return text


def mask_obj(value: Any) -> Any:
    """Recursively mask every string inside dicts / lists / tuples."""
    if isinstance(value, str):
        return mask_text(value)
    if isinstance(value, dict):
        return {k: mask_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [mask_obj(v) for v in value]
    if isinstance(value, tuple):
        return tuple(mask_obj(v) for v in value)
    return value
