"""Pull exact values (order IDs, dates, product codes, ...) out of free text.

Values are returned EXACTLY as written (never normalised) so names, order IDs,
dates and product codes are preserved. Nothing is ever invented.
"""
import re
from datetime import date, timedelta

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


# ---------------------------------------------------------------- time references (explicit, relative, month)
from calendar import monthrange
from dataclasses import dataclass


@dataclass(frozen=True)
class TimeRef:
    kind: str            # explicit | relative | month
    start: date          # first day the reference can mean
    end: date            # last day it can mean (== start for explicit / relative)
    raw: str             # the exact text, so the caller can remove it from the search terms


_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "un": 1, "una": 1, "une": 1, "two": 2, "dos": 2, "deux": 2, "three": 3,
              "tres": 3, "trois": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
              "twelve": 12}
_UNIT = {"day": "d", "days": "d", "dia": "d", "dias": "d", "jour": "d", "jours": "d",
         "week": "w", "weeks": "w", "semana": "w", "semanas": "w", "semaine": "w", "semaines": "w",
         "month": "m", "months": "m", "mes": "m", "meses": "m", "mois": "m",
         "year": "y", "years": "y", "ano": "y", "anos": "y", "an": "y", "ans": "y"}
_AGO = re.compile(
    r"(?:\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s+(day|days|week|weeks|month|months|year|years)\s+ago\b"
    r"|\bhace\s+(\d+|un|una|dos|tres)\s+(dia|dias|semana|semanas|mes|meses|ano|anos)\b"
    r"|\bil y a\s+(\d+|un|une|deux|trois)\s+(jour|jours|semaine|semaines|mois|an|ans)\b)", re.I)
_LAST = re.compile(
    r"\b(last|previous) (week|month|year)\b|\b(?:la semana|el mes|el ano) pasad[oa]\b|"
    r"\b(?:la semaine|le mois|l.annee) derniere\b|\b(?:le mois|la semaine) dernier\b|\bl.an dernier\b", re.I)
_YESTERDAY = re.compile(r"\b(?:yesterday|ayer|hier)\b", re.I)
_MONTH_YEAR = re.compile(rf"\b({_MONTH_ALT})\.?,?\s+(?:de\s+)?(\d{{4}})\b", re.I)


def _minus(d: date, n: int, unit: str) -> date:
    if unit == "d":
        return d - timedelta(days=n)
    if unit == "w":
        return d - timedelta(weeks=n)
    months = n if unit == "m" else 12 * n
    y, m = divmod(d.year * 12 + d.month - 1 - months, 12)
    return date(y, m + 1, min(d.day, monthrange(y, m + 1)[1]))


def find_time_references(text: str, today: date) -> list[TimeRef]:
    """Dates the customer refers to: explicit dates first, else relative ('last year', '6 months ago'),
    else a whole month ('March 2026'). Only ONE kind is returned (the most precise one found)."""
    from datetime import timedelta  # noqa: F401  (kept local for clarity)
    text = text or ""
    explicit = []
    for raw in extract_entities(text)["dates"]:
        d = parse_date(raw)
        if d:
            explicit.append(TimeRef("explicit", d, d, raw))
    if explicit:
        return explicit
    folded = fold(text).replace("\u2019", "'")
    rel: list[TimeRef] = []
    if m := _YESTERDAY.search(folded):
        rel.append(TimeRef("relative", today - timedelta(days=1), today - timedelta(days=1), m.group(0)))
    for m in _AGO.finditer(folded):
        num, unit = [g for g in m.groups() if g][:2]
        n = int(num) if num.isdigit() else _NUM_WORDS.get(num.lower(), 1)
        d = _minus(today, n, _UNIT[unit.lower()])
        rel.append(TimeRef("relative", d, d, m.group(0)))
    for m in _LAST.finditer(folded):
        unit_word = (m.group(2) or m.group(0)).lower()
        unit = "w" if "week" in unit_word or "semana" in unit_word or "semaine" in unit_word else \
               "m" if "month" in unit_word or "mes" in unit_word or "mois" in unit_word else "y"
        d = _minus(today, 1, unit)
        rel.append(TimeRef("relative", d, d, m.group(0)))
    if rel:
        return rel
    months = []
    for m in _MONTH_YEAR.finditer(folded):
        mon, year = _MONTHS.get(fold(m.group(1))), int(m.group(2))
        if mon:
            months.append(TimeRef("month", date(year, mon, 1), date(year, mon, monthrange(year, mon)[1]), m.group(0)))
    return months
