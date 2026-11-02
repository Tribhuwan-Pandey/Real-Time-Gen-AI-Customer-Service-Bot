"""Query preparation: tokenising, stop-words and a small multilingual glossary.

Retrieval is lexical (no embedding model needed, fully offline and testable). The glossary maps
common Spanish / French / Hindi(romanised) support terms to the English terms used in the KB.
A multilingual embedding model would be the natural production upgrade.
"""
import re

from ..core.text import fold, tokens

STOP = set("""a an the is are was were be been of to in on for and or with my your i you it this that what how do does
can will when where which who whom why please tell me about have has had if as at by from there their they we our us
could would should may might am not no yes any some than then so such also just only into over under after before more
most other many much very get got want need know one ones same those these thing things did all time always every money card done doesn didn isn aren wasn weren haven hasn hadn won wouldn couldn shouldn like dislike fast soon quickly purchase purchased something anything everything someone thanks thank hello hi hey now english hindi spanish french ok okay kindly regards dear sorry hmm well also bought purchased ordered received took
cuantos cuanto tengo tiene para por los las del que como cual cuales hay puedo quiero necesito
combien pour les des est une que quel quelle quels avez puis veux besoin
kya hai hain ke ki ka liye kitne mera meri mujhe chahiye""".split())

_GLOSSARY = {
    # Spanish
    "devolucion": "return", "devoluciones": "return", "devolver": "return", "reembolso": "refund",
    "garantia": "warranty", "dias": "days", "dia": "days", "politica": "policy", "envio": "shipping",
    "pedido": "order", "cuenta": "account", "contrasena": "password", "cambio": "exchange",
    "cancelacion": "cancellation", "pago": "payment", "plazo": "period", "tiempo": "period",
    # French
    "retour": "return", "retours": "return", "rembourser": "refund", "remboursement": "refund",
    "politique": "policy", "jours": "days", "jour": "days", "livraison": "shipping", "commande": "order",
    "compte": "account", "echange": "exchange", "annulation": "cancellation", "paiement": "payment",
    "delai": "period", "duree": "period",
    # Hindi (romanised)
    "wapas": "return", "wapsi": "return", "paise": "refund", "kitne": "days", "din": "days", "warranty": "warranty",
}


# Phrase-level paraphrase normalisation (applied to the QUESTION only, never to documents).
_PHRASES = [
    (r"send(?:ing)? (?:it |them |this |that |my \w+ )?back|give(?:n|ing)? (?:it |them |my \w+ )?back|bring(?:ing)? (?:it )?back|"
     r"take(?:n)? (?:it )?back|sending back|returning", "return"),
    (r"money back|get my money|reimburse\w*|repay\w*|reimbursement|paid back", "refund"),
    (r"wi-?fi (?:box|router|device)|internet (?:box|modem|device)|\bmodem\b|\bwi-?fi\b", "router"),
    (r"drop(?:s|ping|ped)? (?:the |my )?(?:connection|out)|cuts? out|cutting out|not staying connected|"
     r"keeps? dropping|los(?:e|es|ing) (?:the |my )?connection|disconnect(?:s|ing|ed)?|no connection|keeps? cutting", "disconnecting"),
    (r"\bdeliver\w*|\barriv\w*|\bparcels?\b|\bpackages?\b|\bdispatch\w*|\bposted\b", "shipping"),
    (r"\bguarantee\w*", "warranty"),
    (r"\bspill\w*|\bliquid\b|water damage|\bwet\b", "liquid spills accidental damage"),
    (r"original box|\bbox\b|wrapping|wrapped", "original packaging"),
    (r"\bused\b|\bopened\b|\bworn\b", "unused"),
    (r"\bhow long\b|time limit|\bdeadline\b|\bwindow\b|\bhow soon\b|\bhow fast\b|\bhow quickly\b|\bhow much time\b|\bhow many days\b|\bperiod\b", "days"),
    (r"\b(?:credit |debit )?card\b|\bbank account\b", "payment method"),
    (r"\bfix\b|\bsolve\b|\bresolve\b|\bwhat should i do\b|\btroubleshoot\w*", "troubleshooting"),
]
_PHRASE_RX = [(re.compile(p, re.I), r) for p, r in _PHRASES]


def normalise_question(question: str) -> str:
    text = fold(question or "").replace("’", "'")
    for rx, repl in _PHRASE_RX:
        text = rx.sub(f" {repl} ", text)
    return text


def query_terms(question: str) -> list[str]:
    out: list[str] = []
    for tok in tokens(normalise_question(question)):
        tok = _GLOSSARY.get(tok, tok)
        if len(tok) > 2 and tok not in STOP and tok not in out:
            out.append(tok)
    return out


def content_tokens(text: str) -> set[str]:
    return {_GLOSSARY.get(t, t) for t in tokens(text) if len(t) > 2 and t not in STOP}


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -•*\t") for p in parts if p.strip(" -•*\t")]


_IRREGULAR = {"frozen": "freez", "froze": "freez", "freezing": "freez", "freeze": "freez", "freezes": "freez",
              "bought": "buy", "paid": "pay", "sent": "send", "gave": "give", "given": "give"}


def stem(tok: str) -> str:
    """Very small English stemmer so 'returns'/'returned'/'returning' match 'return'."""
    if tok in _IRREGULAR:
        return _IRREGULAR[tok]
    for suffix in ("ing", "ed", "es", "s"):
        if tok.endswith(suffix) and len(tok) - len(suffix) >= 3:
            return tok[: -len(suffix)]
    return tok
