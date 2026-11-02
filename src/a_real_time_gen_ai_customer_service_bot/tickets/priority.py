"""Ticket priority = severity + sentiment + customer impact + waiting time (+ SLA breach).

Thresholds come from Config. High-risk issues never fall below 'high', whatever the tone.
"""
import re

from ..core.config import Config

SEVERITY = {"account_compromise": 5, "legal_threat": 5, "duplicate_payment": 4, "technical_issue": 2,
            "refund": 2, "order_status": 1, "cancel_order": 1, "complaint": 1, "general": 1}
SENTIMENT_POINTS = {"negative": 1, "sarcastic": 1, "frustrated": 2, "urgent": 2}
_IMPACT_HIGH = re.compile(r"\b(?:all (?:customers|users)|everyone|whole team|entire (?:team|company)|production|"
                          r"business (?:is )?(?:stopped|down)|many (?:users|customers))\b", re.I)
_IMPACT_MED = re.compile(r"\b(?:my team|several|multiple|both|family|our (?:office|staff))\b", re.I)
ORDER = ["low", "medium", "high", "critical"]


def impact_level(text: str) -> int:
    if _IMPACT_HIGH.search(text or ""):
        return 3
    if _IMPACT_MED.search(text or ""):
        return 2
    return 1


def label_for(score: float, config: Config) -> str:
    t = config.get("tickets.priority_thresholds")
    if score >= t["critical"]:
        return "critical"
    if score >= t["high"]:
        return "high"
    if score >= t["medium"]:
        return "medium"
    return "low"


def raise_to(label: str, floor: str) -> str:
    return label if ORDER.index(label) >= ORDER.index(floor) else floor


def compute_priority(issue_type: str, sentiment_label: str, impact: int, waited_ratio: float,
                     breached: bool, risk: bool, config: Config) -> tuple[str, float, dict]:
    """waited_ratio = SLA time consumed / SLA time allowed for the base priority."""
    severity = SEVERITY.get(issue_type, 1)
    sentiment = SENTIMENT_POINTS.get(sentiment_label, 0)
    waiting = 0 if waited_ratio < 0.25 else 1 if waited_ratio < 0.5 else 2 if waited_ratio < 0.75 else 3
    sla_pts = 2 if breached else 0
    score = severity + sentiment + impact + waiting + sla_pts
    label = label_for(score, config)
    if risk or issue_type in ("account_compromise", "legal_threat", "duplicate_payment"):
        label = raise_to(label, "high")
    if breached:
        label = raise_to(label, "high")
    return label, float(score), {"severity": severity, "sentiment": sentiment, "impact": impact,
                                 "waiting": waiting, "sla_breach": sla_pts}
