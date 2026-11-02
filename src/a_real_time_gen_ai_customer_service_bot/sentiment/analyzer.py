"""Multilingual sentiment analysis: positive / neutral / negative / frustrated / urgent / sarcastic.

Multi-label scores (0..1) are computed for every label; the PRIMARY label is chosen by a
fixed priority. Conversation history nudges the result (repeated complaints, carried-over
frustration). Sarcasm = positive wording used in a negative context.

IMPORTANT: sentiment may change TONE only. It never changes policy decisions, and risk
(account compromise, duplicate payment, legal threat) is detected separately from intents.
"""
import re
from dataclasses import dataclass, field

from ..core.config import Config
from ..core.text import fold

LABELS = ("positive", "neutral", "negative", "frustrated", "urgent", "sarcastic")


def _rx(*parts: str) -> re.Pattern:
    return re.compile("|".join(parts), re.I)


_NEG_STRONG = _rx(
    r"\b(?:terrible|awful|horrible|worst|hate|useless|pathetic|scam|fraud|cheated|rude|angry|furious|unhappy|upset|disgusting|appalling)\b",
    r"\b(?:bekar|bura|gussa|dhokha|pareshan|takleef|bakwas)\b|खराब|बेकार|परेशान|गुस्सा|धोखा",
    r"\b(?:pesimo|horrible|enojado|molesto|inutil|estafa|fraude|decepcionado|harto)\b",
    r"\b(?:nul|furieux|enerve|inutile|arnaque|decu|mauvais service)\b")
_NEG_MILD = _rx(
    r"\b(?:bad|poor|disappointed|disappointing|broken|problem|issue|wrong|damaged|late|delay\w*|missing|failed|failure|"
    r"not working|doesn.t work|never (?:arrived|came|received)|lost|annoyed|complaint)\b",
    r"\b(?:kharab|galat|nahi mila|nahi aaya|nahi chal|der)\b",
    r"\b(?:malo|mal|roto|problema|retraso|perdido|queja)\b",
    r"\b(?:mauvais|casse|probleme|retard|perdu)\b")
_FRUSTRATED = _rx(
    r"\b(?:again|still|yet again|third time|fourth time|second time|already (?:told|said|sent|explained|gave)|nobody|no one|"
    r"waste of|fed up|ridiculous|unacceptable|sick of|tired of|how many times|why is it so hard|wtf|enough)\b",
    r"\b(?:kitni baar|phir se|abhi tak|tang aa gaya|bar bar)\b|फिर से|कितनी बार",
    r"\b(?:otra vez|de nuevo|todavia|ya les dije|harto|cansado|inaceptable)\b",
    r"\b(?:encore|toujours pas|deja dit|marre|ras le bol|inacceptable)\b")
_URGENT = _rx(
    r"\b(?:urgent\w*|immediately|asap|right now|emergency|as soon as possible|right away|critical|can.t wait|deadline|time.sensitive)\b",
    r"\b(?:jaldi|turant|foran|abhi ke abhi)\b|तुरंत|जल्दी",
    r"\b(?:urgente|inmediatamente|ahora mismo|cuanto antes|ya mismo)\b",
    r"\b(?:urgent|immediatement|tout de suite|au plus vite)\b")
_POSITIVE = _rx(
    r"\b(?:thanks|thank you|great|awesome|perfect|excellent|love|happy|wonderful|amazing|fantastic|appreciate\w*|helpful|good job|brilliant|nice)\b",
    r"\b(?:shukriya|dhanyavad|badhiya|accha|mast|bahut accha)\b|धन्यवाद|शुक्रिया|बढ़िया",
    r"\b(?:gracias|excelente|perfecto|genial|feliz|encanta)\b",
    r"\b(?:merci|parfait|genial|content|super)\b")
_SARCASM_STRONG = _rx(
    r"\bthanks for nothing\b", r"\byeah,? right\b", r"\boh,? (?:great|wonderful|brilliant|fantastic|perfect)\b",
    r"\bjust (?:great|perfect|wonderful|what i needed)\b", r"\bgreat,? (?:another|more|just)\b",
    r"\b(?:real|so|very) helpful\b[^.\n]{0,25}\b(?:not|never|nothing)\b", r"\blove (?:how|waiting|being)\b",
    r"\bwhat a (?:surprise|shock|joke)\b", r"\bnice (?:job|work|going)\b[^.\n]{0,30}\b(?:breaking|losing|ignoring)\b",
    r"\bgracias por nada\b", r"\b(?:que bien|genial|perfecto),? (?:otra vez|de nuevo)\b",
    r"\bmerci pour rien\b", r"\bsuper,? (?:encore|bravo)\b",
    r"\bwaah\b[^.\n]{0,30}\b(?:phir|nahi|again)\b", r"\bbahut badhiya\b[^.\n]{0,20}\b(?:phir|nahi)\b")
_SARCASM_POS_NEG = _rx(
    r"\b(?:great|wonderful|fantastic|amazing|perfect|brilliant|lovely|thanks a lot|thank you so much|excellent|awesome|love)\b"
    r"[^.\n]{0,50}\b(?:again|another|nothing|late|delay\w*|broken|still|waiting|weeks|days|never|wrong|lost|not working|broke|crashed|ignored|fail\w*)\b")
_NEG_EVENT = _rx(r"\b(?:again|another|nothing|late|delay\w*|broken|still waiting|weeks|never|lost|ignored)\b")


@dataclass
class SentimentResult:
    label: str
    confidence: float
    scores: dict[str, float]
    negative_like: bool
    is_urgent: bool
    low_confidence: bool
    signals: list[str] = field(default_factory=list)


def _clamp(x: float) -> float:
    return round(max(0.0, min(1.0, x)), 3)


class SentimentAnalyzer:
    def __init__(self, config: Config):
        self._config = config

    def analyze(self, text: str, history: list[SentimentResult] | None = None) -> SentimentResult:
        history = history or []
        raw = text or ""
        norm = fold(raw.replace("’", "'"))
        signals: list[str] = []

        def count(rx: re.Pattern) -> int:
            return len(rx.findall(norm)) + (len(rx.findall(raw)) if re.search(r"[\u0900-\u097f]", raw) else 0)

        neg = 0.6 * min(2, count(_NEG_STRONG)) + 0.35 * min(2, count(_NEG_MILD))
        frus = 0.4 * min(3, count(_FRUSTRATED))
        urgent = 0.65 * min(2, count(_URGENT))
        pos = 0.5 * min(2, count(_POSITIVE))

        if raw.count("!") >= 2 or raw.count("?") >= 3:
            frus += 0.15
            signals.append("emphatic_punctuation")
        letters = [c for c in raw if c.isalpha()]
        words = re.findall(r"[A-Za-z]{3,}", raw)
        if len(words) >= 3 and sum(1 for w in words if w.isupper()) / len(words) > 0.6:
            frus += 0.3
            signals.append("shouting_caps")
        if frus > 0 and neg > 0.3:
            frus += 0.2

        sarcasm = 0.0
        if _SARCASM_STRONG.search(norm):
            sarcasm = 0.8
            signals.append("sarcasm_pattern")
        elif _SARCASM_POS_NEG.search(norm) or (count(_POSITIVE) and _NEG_EVENT.search(norm)):
            sarcasm = 0.6
            signals.append("positive_wording_negative_context")
        if sarcasm and history and history[-1].negative_like:
            sarcasm += 0.1
            signals.append("prior_negative_context")
        if sarcasm:
            pos *= 0.2
            neg = max(neg, 0.3)
        elif neg >= 0.35:
            pos *= 0.3

        # history: a run of complaints raises the temperature; frustration carries over briefly
        recent_neg = sum(1 for h in history[-2:] if h.negative_like)
        if recent_neg >= 2 and (neg > 0 or frus > 0):
            neg += 0.15
            frus += 0.15
            signals.append("repeated_negative_history")
        if history and len(norm.split()) <= 4 and not pos and not sarcasm:
            prev = history[-1]
            if prev.scores.get("frustrated", 0) >= 0.5:
                frus = max(frus, 0.5 * prev.scores["frustrated"] + 0.25)
                signals.append("carried_over_frustration")

        scores = {"positive": _clamp(pos), "negative": _clamp(neg), "frustrated": _clamp(frus),
                  "urgent": _clamp(urgent), "sarcastic": _clamp(sarcasm)}
        if scores["sarcastic"] >= 0.5:
            label, conf = "sarcastic", scores["sarcastic"]
        elif scores["urgent"] >= 0.6:
            label, conf = "urgent", scores["urgent"]
        elif scores["frustrated"] >= 0.5:
            label, conf = "frustrated", scores["frustrated"]
        elif scores["negative"] >= 0.5:
            label, conf = "negative", scores["negative"]
        elif scores["positive"] >= 0.5:
            label, conf = "positive", scores["positive"]
        else:
            label, conf = "neutral", _clamp(1 - max(scores.values()))
            conf = max(conf, 0.5) if max(scores.values()) < 0.35 else conf
        scores["neutral"] = conf if label == "neutral" else _clamp(1 - max(scores.values()))
        negative_like = (max(scores["negative"], scores["frustrated"], scores["sarcastic"]) >= 0.5
                         or (scores["urgent"] >= 0.6 and scores["negative"] >= 0.35))
        return SentimentResult(
            label=label, confidence=conf, scores=scores, negative_like=negative_like,
            is_urgent=scores["urgent"] >= 0.6, signals=signals,
            low_confidence=conf < self._config.get("confidence.sentiment_min"))
