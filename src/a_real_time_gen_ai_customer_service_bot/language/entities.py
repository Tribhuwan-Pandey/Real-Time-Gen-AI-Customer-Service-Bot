"""Pull exact values (order IDs, dates, product codes, ...) out of free text.

Values are returned EXACTLY as written (never normalised) so names, order IDs,
dates and product codes are preserved. Nothing is ever invented.
"""
import re
from datetime import date

from ..core.masking import _EMAIL, _PHONE_IN, _PHONE_INTL
from ..core.text import fold

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
    "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6, "juillet": 7,
    "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
}
_MONTH_ALT = "|".join(sorted(
    list(_MONTHS) + ["février", "août", "décembre"], key=len, reverse=True))

_ORD = re.compile(r"(?<![A-Za-z0-9])(?:ORD|ORDER)[-_#]?\d{3,}\b", re.I)
_ORD_CTX = re.compile(
    r"(?:order|ऑर्डर|pedido|commande)\s*(?:id|number|no\.?|num|numero|número|numéro)?\s*[#:]?\s*"
    r"([A-Z]{0,4}-?\d{4,})(?![\d-])", re.I)
_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_DMY = re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{4}\b")
_DMY_NAMED = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:de\s+)?(?:{_MONTH_ALT})\.?,?\s+(?:de\s+)?\d{{4}}\b", re.I)
_MDY_NAMED = re.compile(rf"\b(?:{_MONTH_ALT})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b", re.I)
_PRODUCT = re.compile(r"\b[A-Z]{2,4}-\d{2,6}[A-Z]?\b")
_ERR1 = re.compile(r"\b(?:ERR(?:OR)?)[-_ ]?\d{2,5}\b", re.I)
_ERR2 = re.compile(r"\bE[-_]?\d{3,5}\b")
_AMOUNT1 = re.compile(r"(?:₹|Rs\.?|INR|USD|\$|€|EUR)\s?\d[\d,]*(?:\.\d{1,2})?", re.I)
_AMOUNT2 = re.compile(r"\b\d[\d,]*(?:\.\d{1,2})?\s?(?:rupees|rs|inr|usd|dollars|euros|eur)\b", re.I)
_NAME = re.compile(
    r"(?:my name is|name\s*:|mera naam|mera nam|me llamo|mi nombre es|je m'appelle|je m’appelle|mon nom est)\s+"
    r"([^\W\d_]+(?:\s[^\W\d_]+)?)", re.I | re.U)
_NAME_STOP = {"hai", "and", "aur", "y", "et", "is", "from", "the", "ka", "ke"}
_CORRECTION = re.compile(
    r"\b(?:sorry|actually|i meant|i mean|correction|my mistake|typo|wrong (?:number|id|order)|"
    r"not that|galat|mera matlab|maaf|perdon|disculpa|quise decir|en realidad|pardon|"
    r"je voulais dire|plutot|erreur de ma part)\b", re.I)

ENTITY_KEYS = ("order_ids", "dates", "product_codes", "emails", "phones", "amounts", "error_codes", "names")


def _unique(items):
    seen, out = set(), []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def extract_entities(text: str) -> dict[str, list[str]]:
    text = text or ""
    order_ids = [m.group(0) for m in _ORD.finditer(text)]
    order_ids += [m.group(1) for m in _ORD_CTX.finditer(text)]
    order_ids = _unique(order_ids)
    order_set = {o.lower() for o in order_ids}

    dates = [m.group(0) for rx in (_ISO, _DMY, _DMY_NAMED, _MDY_NAMED) for m in rx.finditer(text)]
    products = [p for p in _PRODUCT.findall(text)
                if not p.upper().startswith(("ORD", "ERR")) and p.lower() not in order_set]
    errors = [m.group(0) for rx in (_ERR1, _ERR2) for m in rx.finditer(text)]
    amounts = [m.group(0).strip() for rx in (_AMOUNT1, _AMOUNT2) for m in rx.finditer(text)]
    names = []
    for m in _NAME.finditer(text):
        words = [w for w in m.group(1).split() if w.lower() not in _NAME_STOP]
        if words:
            names.append(" ".join(words))
    return {
        "order_ids": order_ids,
        "dates": _unique(dates),
        "product_codes": _unique(products),
        "emails": _unique(_EMAIL.findall(text)),
        "phones": _unique(_PHONE_INTL.findall(text) + _PHONE_IN.findall(text)),
        "amounts": _unique(amounts),
        "error_codes": _unique(errors),
        "names": _unique(names),
    }


def has_correction(text: str) -> bool:
    return bool(_CORRECTION.search(fold(text or "")))


def merge_entities(store: dict[str, list[str]], new: dict[str, list[str]], correction: bool) -> list[str]:
    """Merge newly extracted entities into a conversation's store.

    A correction ("sorry, I meant ORD-222") REPLACES older values of that type;
    otherwise new values are appended. Returns the list of entity types that were corrected.
    """
    corrected = []
    for key in ENTITY_KEYS:
        values = new.get(key) or []
        if not values:
            store.setdefault(key, [])
            continue
        if correction and store.get(key):
            if [v.lower() for v in store[key]] != [v.lower() for v in values]:
                corrected.append(key)
            store[key] = list(values)
        else:
            store[key] = _unique(store.get(key, []) + values)
    return corrected


def parse_date(raw: str) -> date | None:
    """Parse ISO, dd/mm/yyyy and 'D Month YYYY' / 'Month D, YYYY' (en/es/fr). None if not parseable."""
    raw = fold(raw or "")
    try:
        if m := re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", raw):
            return date(int(m[1]), int(m[2]), int(m[3]))
        if m := re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", raw):
            return date(int(m[3]), int(m[2]), int(m[1]))      # day first
        if m := re.search(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:de\s+)?({_fold_alt()})\.?,?\s+(?:de\s+)?(\d{{4}})", raw):
            return date(int(m[3]), _MONTHS[m[2]], int(m[1]))
        if m := re.search(rf"({_fold_alt()})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})", raw):
            return date(int(m[3]), _MONTHS[m[1]], int(m[2]))
    except ValueError:
        return None
    return None


def _fold_alt() -> str:
    return "|".join(sorted(_MONTHS, key=len, reverse=True))


def find_dates(text: str) -> list[date]:
    """All parseable dates mentioned in text."""
    found = extract_entities(text)["dates"]
    return [d for d in (parse_date(x) for x in found) if d]
