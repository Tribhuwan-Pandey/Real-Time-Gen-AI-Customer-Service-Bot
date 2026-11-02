"""Rule-based language detection for English, Hindi (Devanagari AND romanised), Spanish, French.

German / Portuguese word lists exist only so we can RECOGNISE an unsupported
language and say so politely instead of guessing.
Supported languages and the confidence threshold come from Config.
"""
import re
from dataclasses import dataclass, field

from ..core.config import Config
from ..core.text import fold, tokens
from .entities import extract_entities

_WORDS: dict[str, str] = {
    "en": "the is are my i not was have has you your this that with for it and to of in on please can what when how "
          "been am do does did why where would could me we they will from but there want need still again help "
          "hello thanks thank received paid charged wrong working yet already its dont cant wont ive",
    "es": "el la los las es son mi mis no pero tengo tiene con por para una un que cuando como donde quiero necesito "
          "pedido reembolso cobro cobraron dos veces muy mal gracias hola ayuda se ha hemos estoy esta hace desde "
          "todavia otra vez mas cuenta pago mi tarjeta recibi recibido llego",
    "fr": "le les est sont mon ma mes pas mais ai avec pour une que quand comment ou je veux besoin commande "
          "remboursement debite fois tres merci bonjour aide vous nous il elle ne toujours encore depuis deja suis "
          "compte paiement carte recu j l d jai",
    "hi": "mera meri mere mujhe nahi nahin hai hain kya ka ki ke ko se mein main aur par paisa paise kat gaya gayi "
          "abhi jaldi kripya bhai yaar kab kaise kyun chahiye wapas dobara baar mila theek shukriya dhanyavad bahut "
          "kharab ho aaya aya pahuncha hua hui kiya kiye raha rahi tha thi lekin toh bhi",
    "de": "ich ist nicht und mein meine bitte haben kein ein eine der die das bestellung ich habe wurde zweimal danke hallo",
    "pt": "meu minha nao obrigado pedido voce tenho foi duas vezes ola ajuda cobrado",
}
_SETS = {lang: set(words.split()) for lang, words in _WORDS.items()}
_COUNT: dict[str, int] = {}
for _words in _SETS.values():
    for _w in _words:
        _COUNT[_w] = _COUNT.get(_w, 0) + 1
# a word that appears in several languages carries no information
_UNIQUE = {lang: {w for w in words if _COUNT[w] == 1} for lang, words in _SETS.items()}
_DEVANAGARI = re.compile(r"[\u0900-\u097f]")
_NAMES = {"en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese"}


@dataclass
class LanguageResult:
    language: str | None
    confidence: float
    supported: bool
    mixed: bool = False
    languages_present: list[str] = field(default_factory=list)
    transliterated: bool = False
    scores: dict[str, int] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return _NAMES.get(self.language or "", "unknown")


class LanguageDetector:
    def __init__(self, config: Config):
        self._config = config

    def detect(self, text: str) -> LanguageResult:
        stripped = text or ""
        # remove entities (order ids, codes, emails ...) so they don't count as words
        for values in extract_entities(stripped).values():
            for v in values:
                stripped = stripped.replace(v, " ")
        words = tokens(stripped)
        hits = {lang: sum(1 for w in words if w in uniq) for lang, uniq in _UNIQUE.items()}
        dev_chars = len(_DEVANAGARI.findall(stripped))
        letters = len(re.findall(r"[^\W\d_]", stripped)) or 1
        script_hindi = dev_chars / letters >= 0.3
        if script_hindi:
            hits["hi"] += max(3, len([w for w in words if _DEVANAGARI.search(w)]))
        total = sum(hits.values())
        supported_langs = self._config.get("languages.supported")
        if total == 0:
            return LanguageResult(None, 0.0, False, scores=hits)
        top = max(hits, key=lambda k: (hits[k], k == "en"))
        share = hits[top] / total
        evidence = min(1.0, total / 3)
        confidence = round(share * (0.5 + 0.5 * evidence), 3)
        present = [lang for lang, h in sorted(hits.items(), key=lambda kv: -kv[1])
                   if h >= 1 and h / total >= 0.2]
        return LanguageResult(
            language=top, confidence=confidence, supported=top in supported_langs,
            mixed=len(present) > 1, languages_present=present,
            transliterated=(top == "hi" and not script_hindi), scores=hits)
