"""Rule-based multilingual intent detection with typo tolerance.

Several intents can match one message (multiple requests). Confidence rises with the
number of distinct patterns that fire. High-risk intents (account compromise, duplicate
payment, legal threat) are matched in English, Hindi (Devanagari + romanised), Spanish, French.
"""
import difflib
import re
from dataclasses import dataclass

from ..core.text import fold

_PATTERNS: dict[str, list[str]] = {
    "account_compromise": [
        r"\b(?:hacked|compromised|phished|account (?:was |got |has been )?(?:breached|stolen|taken over|hacked)|stolen account)\b",
        r"\bunauthori[sz]ed (?:login|access|transaction|purchase|activity|charge)s?\b",
        r"\bsomeone (?:else )?(?:logged|accessed|is using|has access|got into|broke into)\b",
        r"\b(?:logged in|login) from (?:an? )?(?:unknown|unfamiliar|strange)\b",
        r"\bpassword (?:was |got )?changed (?:without|by someone)\b",
        r"\bhack ho\b|\baccount hack\b|\bkisi ne (?:mera )?account\b|हैक|खाता.*हैक",
        r"\bhackearon\b|\bcuenta (?:fue |ha sido )?(?:hackeada|comprometida|robada)\b|\bacceso no autorizado\b|\balguien (?:entro|accedio|esta usando)\b",
        r"\bcompte (?:a ete |a été )?(?:pirate|hacke|compromis|vole)\b|\bacces non autorise\b|\bquelqu.un (?:s.est connecte|a acces|utilise)\b",
    ],
    "duplicate_payment": [
        r"\b(?:charged|billed|debited|deducted|paid)\b[^.\n]{0,25}\b(?:twice|two times|2 times|double|again)\b",
        r"\b(?:twice|double|duplicate)\b[^.\n]{0,20}\b(?:charge[ds]?|payment|billing|deduction|debit|transaction)\b",
        r"\bdouble[- ]?charg\w*|\bduplicate (?:charge|payment|transaction|billing)\b|\bpaid twice\b|\btwo (?:separate )?(?:charges|payments)\b",
        r"\bdo baar\b[^.\n]{0,20}\b(?:paise|paisa|payment|kat|cut)\b|\b(?:paise|paisa)\b[^.\n]{0,15}\bdo baar\b|\bdobara (?:paise|paisa)\b|दो बार",
        r"\bcobr\w+ dos veces\b|\bdoble cobro\b|\bcobro doble\b|\bpago duplicado\b|\bpague dos veces\b",
        r"\bdebit\w* deux fois\b|\bpaye deux fois\b|\bdouble debit\b|\bpaiement en double\b|\bfacture\w* deux fois\b",
    ],
    "legal_threat": [
        r"\b(?:lawyer|attorney|solicitor|legal action|legal notice|legal team|legal proceedings|lawsuit|suing)\b|\bsue\b",
        r"\btake (?:you|this|the company|it) to court\b|\bconsumer (?:court|forum|protection)\b|\bfile a (?:case|complaint) (?:against|in court)\b",
        r"\bvakeel\b|\bkanooni\b|\bcourt mein\b|कानूनी|वकील|उपभोक्ता",
        r"\babogado\b|\bdemandar\b|\bdemanda legal\b|\btribunal\b|\baccion legal\b",
        r"\bavocat\b|\baction en justice\b|\bpoursuivre en justice\b|\btribunal\b",
    ],
    "refund": [
        r"\b(?:refund|refunded|money back|reimburse\w*|return my money)\b",
        r"\bpaise wapas\b|\brefund chahiye\b|\bwapas chahiye\b|रिफंड|पैसे वापस",
        r"\breembolso\b|\bdevuelv\w+ (?:mi )?dinero\b",
        r"\bremboursement\b|\brembourser\b",
    ],
    "order_status": [
        r"\bwhere is my (?:order|package|parcel|delivery)\b|\btrack\w*\b|\border status\b|\bdelivery (?:status|date|delay\w*|late)\b",
        r"\b(?:not|hasn.t|haven.t|never)\b[^.\n]{0,15}\b(?:arrived|delivered|received|shipped)\b",
        r"\border nahi (?:aaya|aya|pahuncha|mila)\b|\bkab (?:aayega|milega|ayega)\b|नहीं (?:आया|पहुंचा|मिला)",
        r"\bdonde esta mi pedido\b|\bno (?:ha )?llegado\b|\bmi pedido no\b|\bno (?:he )?recibido\b",
        r"\bou est ma commande\b|\bpas (?:encore )?(?:recu|arrive|livre)\b|\bcommande n.est pas\b",
    ],
    "technical_issue": [
        r"\b(?:not working|doesn.t work|does not work|stopped working|broken|crash\w*|defective|faulty|damaged|no power|blue screen|won.t (?:turn on|start|connect|charge))\b",
        r"\bkeeps? (?:disconnecting|crashing|restarting|freezing)\b|\berror\b|\bglitch\w*\b",
        r"\bkaam nahi kar\w*\b|\bchal nahi\b|\bkharab\b|\bband ho gaya\b|काम नहीं|खराब",
        r"\bno funciona\b|\bdejo de funcionar\b|\broto\b|\bdanado\b",
        r"\bne marche pas\b|\bne fonctionne pas\b|\ben panne\b|\bcasse\b|\bdefectueux\b|\berreur\b",
    ],
    "policy_question": [
        r"\b(?:return|refund|warranty|exchange|cancellation|shipping|privacy|replacement) (?:policy|period|window|terms|conditions)\b",
        r"\bhow (?:long|many days)\b|\bwhat is (?:your|the) (?:policy|warranty|return)\b|\bcan i (?:return|exchange)\b|\bis (?:it|this) covered\b|\bwarranty\b",
        r"\bpolicy kya\b|\bkitne din\b|\bwapsi ki policy\b",
        r"\bpolitica de (?:devolucion|reembolso|garantia)\b|\bcuantos dias\b|\bgarantia\b",
        r"\bpolitique de (?:retour|remboursement)\b|\bgarantie\b|\bcombien de jours\b",
    ],
    "cancel_order": [
        r"\bcancel (?:my |the )?(?:order|subscription|purchase)\b|\border cancel\b|\bcancelar (?:mi )?pedido\b|\bannuler (?:ma )?commande\b",
    ],
    "human_agent": [
        r"\b(?:talk|speak|connect me) to (?:a |an |the )?(?:human|agent|person|manager|representative|supervisor)\b|\breal person\b|\binsaan se\b|\bagente humano\b|\bparler a un (?:humain|agent|conseiller)\b",
    ],
    "resolved": [
        r"\b(?:it|that|this|everything)(?:.s| is| has been| was)? (?:fixed|resolved|working now|sorted)\b|\bproblem (?:is )?(?:solved|resolved)\b|\bworks now\b|\bworking now\b|\bissue (?:is )?resolved\b|\bye(?:s)? it worked\b",
    ],
    "greeting": [r"^(?:hi|hello|hey|namaste|hola|bonjour|salut|good (?:morning|afternoon|evening))\b"],
    "thanks": [r"\bthank(?:s| you)\b|\bshukriya\b|\bdhanyavad\b|\bgracias\b|\bmerci\b|धन्यवाद|शुक्रिया"],
    "complaint": [
        r"\b(?:not happy|disappointed|unacceptable|terrible service|worst|complain\w*|ridiculous|fed up|useless|pathetic)\b",
        r"\bbahut kharab\b|\bmuy mal\b|\btres mauvais\b",
    ],
}
_COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in _PATTERNS.items()}

_VOCAB = ["refund", "charged", "twice", "payment", "duplicate", "account", "hacked", "password", "unauthorized",
          "compromised", "lawyer", "legal", "court", "attorney", "delivery", "delivered", "shipping", "received",
          "damaged", "broken", "working", "delayed", "cancel", "cancelled", "return", "warranty", "policy",
          "urgent", "immediately", "terrible", "useless", "angry", "thanks", "support", "human", "login",
          "blocked", "stolen", "fraud", "deducted", "replacement", "arrived", "double", "billed", "debited",
          "disappointed", "unacceptable", "complaint"]
_TYPOS = {"twise": "twice", "twics": "twice", "refnd": "refund", "rfund": "refund", "recieved": "received",
          "recived": "received", "acount": "account", "acconut": "account", "hakced": "hacked", "hacekd": "hacked",
          "pasword": "password", "paymnt": "payment", "paymet": "payment", "chargd": "charged", "chraged": "charged",
          "delievery": "delivery", "delivary": "delivery", "oder": "order", "ordr": "order", "lawer": "lawyer",
          "urgnt": "urgent", "imediately": "immediately"}


def normalize(text: str) -> str:
    """Fold, unify apostrophes, collapse letter floods, fix common typos (matching only)."""
    t = fold((text or "").replace("’", "'").replace("‘", "'"))
    t = re.sub(r"([a-z])\1{2,}", r"\1", t)               # heeelp -> help, refunddd -> refund
    out = []
    for word in re.split(r"(\W+)", t):
        if word in _TYPOS:
            word = _TYPOS[word]
        elif len(word) >= 5 and word.isalpha() and word not in _VOCAB:
            close = difflib.get_close_matches(word, _VOCAB, n=1, cutoff=0.82)
            if close:
                word = close[0]
        out.append(word)
    return "".join(out)


@dataclass(frozen=True)
class IntentMatch:
    name: str
    confidence: float


def detect_intents(text: str) -> list[IntentMatch]:
    norm = normalize(text)
    raw = fold(text or "")
    matches: list[IntentMatch] = []
    for name, regexes in _COMPILED.items():
        hits = sum(1 for rx in regexes if rx.search(norm) or rx.search(raw))
        if not hits:
            continue
        confidence = min(0.95, 0.7 + 0.12 * (hits - 1))
        if name in ("greeting", "thanks") and len(norm.split()) > 6:
            confidence = 0.55
        matches.append(IntentMatch(name, round(confidence, 2)))
    return sorted(matches, key=lambda m: -m.confidence)


def intent_names(matches: list[IntentMatch], min_confidence: float = 0.0) -> list[str]:
    return [m.name for m in matches if m.confidence >= min_confidence]
