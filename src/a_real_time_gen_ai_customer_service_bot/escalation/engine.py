"""Escalation rules (Task 1).

Rules, in plain words:
  high_risk:<intent>      account compromise / duplicate payment / legal threat -> escalate,
                          EVEN IF THE CUSTOMER SOUNDS CALM (decided from intents, not sentiment)
  repeated_negative       N consecutive negative-like customer messages (N from config)
  unresolved_negative     negative conversation unresolved for MORE than X minutes (X from config)
Routing of complaints:
  business open           -> agent queue, handle now
  closed + urgent/high-risk -> on-call queue
  closed + normal         -> scheduled for the next opening time (next working day / after weekend / holiday)
Every record stores reason, rule, timestamp, masked conversation summary and metadata.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ..core.business_calendar import BusinessCalendar
from ..core.clock import Clock
from ..core.config import Config
from ..core.logging import get_logger
from ..core.masking import mask_obj
from ..language.intents import IntentMatch
from ..sentiment.analyzer import SentimentResult
from ..session.models import Conversation
from ..session.summary import summarize

log = get_logger("escalation")


@dataclass
class EscalationRecord:
    id: str
    kind: str                       # "escalation" | "scheduled"
    customer_id: str
    conversation_id: str
    reason: str
    rule: str                       # the condition that fired
    timestamp: datetime
    summary: str                    # masked
    queue: str                      # agent_queue | on_call | scheduled
    scheduled_for: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EscalationOutcome:
    records: list[EscalationRecord] = field(default_factory=list)
    decision: str = "none"          # none | handle_now | escalated | on_call_queue | scheduled_next_working_day

    @property
    def escalated(self) -> bool:
        return any(r.kind == "escalation" for r in self.records)


class EscalationEngine:
    def __init__(self, config: Config, calendar: BusinessCalendar, clock: Clock):
        self._config = config
        self._calendar = calendar
        self._clock = clock
        self.records: list[EscalationRecord] = []
        self._counter = 0

    # -- conversation bookkeeping --------------------------------------------
    def observe(self, conv: Conversation, sentiment: SentimentResult, intents: list[IntentMatch]) -> None:
        """Update negative_since / resolved from the newest customer message."""
        now = self._clock.now()
        resolved_intent = any(i.name == "resolved" and i.confidence >= 0.5 for i in intents)
        if resolved_intent or (sentiment.label == "positive" and conv.negative_since is not None
                               and not sentiment.negative_like):
            conv.resolved = True
            conv.negative_since = None
            conv.fired_rules -= {r for r in conv.fired_rules
                                 if r in ("repeated_negative", "unresolved_negative")}
        elif sentiment.negative_like:
            conv.resolved = False
            if conv.negative_since is None:
                conv.negative_since = now

    # -- rule evaluation -------------------------------------------------------
    def _timeout_trigger(self, conv: Conversation, now: datetime):
        limit = timedelta(minutes=self._config.get("escalation.unresolved_negative_minutes"))
        if conv.negative_since and not conv.resolved and now - conv.negative_since > limit:
            minutes = (now - conv.negative_since).total_seconds() / 60
            return ("unresolved_negative",
                    f"Negative conversation unresolved for {minutes:.0f} minutes "
                    f"(limit {limit.total_seconds() / 60:.0f})",
                    {"unresolved_minutes": round(minutes, 1)})
        return None

    def evaluate(self, conv: Conversation, intents: list[IntentMatch],
                 sentiment: SentimentResult) -> EscalationOutcome:
        """Call AFTER the current customer message has been appended to `conv`."""
        self.observe(conv, sentiment, intents)
        now = self._clock.now()
        triggers: list[tuple[str, str, dict]] = []

        high_risk = set(self._config.get("escalation.high_risk_intents"))
        min_conf = self._config.get("escalation.high_risk_min_confidence")
        risk_hits = [i for i in intents if i.name in high_risk and i.confidence >= min_conf]
        for i in risk_hits:
            triggers.append((f"high_risk:{i.name}", f"High-risk issue: {i.name}",
                             {"intent": i.name, "confidence": i.confidence}))

        n = self._config.get("escalation.repeated_negative_count")
        recent = conv.user_messages()[-n:]
        if len(recent) == n and all(m.negative_like for m in recent):
            triggers.append(("repeated_negative", f"{n} consecutive negative messages", {"count": n}))

        if (t := self._timeout_trigger(conv, now)):
            triggers.append(("unresolved_negative", t[1], t[2]))

        urgent = sentiment.is_urgent or bool(risk_hits)
        complaint = (sentiment.negative_like or bool(risk_hits)
                     or any(i.name == "complaint" for i in intents))
        outcome = EscalationOutcome()
        for rule, reason, meta in triggers:
            if rule in conv.fired_rules:
                continue
            conv.fired_rules.add(rule)
            meta = {**meta, "sentiment": sentiment.label,
                    "sentiment_confidence": sentiment.confidence,
                    "intents": [i.name for i in intents], "language": conv.language}
            outcome.records.append(self._make_record(
                "escalation", conv, reason, rule, meta, urgent=urgent or rule.startswith("high_risk")))

        if not outcome.records and complaint:
            key = "after_hours_urgent_complaint" if urgent else "after_hours_normal_complaint"
            if not self._calendar.is_open(now) and key not in conv.fired_rules:
                conv.fired_rules.add(key)
                meta = {"sentiment": sentiment.label, "intents": [i.name for i in intents],
                        "language": conv.language}
                kind = "escalation" if urgent else "scheduled"
                reason = ("Urgent complaint received outside business hours" if urgent
                          else "Complaint received outside business hours; scheduled for next working day")
                outcome.records.append(self._make_record(kind, conv, reason, key, meta, urgent=urgent))

        if outcome.records:
            outcome.decision = {"on_call": "on_call_queue", "scheduled": "scheduled_next_working_day"}.get(
                outcome.records[0].queue, "escalated")
            if any(r.queue == "on_call" for r in outcome.records):
                outcome.decision = "on_call_queue"
        elif complaint:
            outcome.decision = "handle_now"
        return outcome

    def _make_record(self, kind: str, conv: Conversation, reason: str, rule: str,
                     meta: dict, urgent: bool) -> EscalationRecord:
        now = self._clock.now()
        scheduled_for = None
        if self._calendar.is_open(now):
            queue = "agent_queue"
        elif urgent:
            queue = "on_call"
        else:
            queue, scheduled_for = "scheduled", self._calendar.next_open(now)
        self._counter += 1
        rec = EscalationRecord(
            id=f"ESC-{self._counter:04d}", kind=kind, customer_id=conv.customer_id,
            conversation_id=conv.session_id, reason=reason, rule=rule, timestamp=now,
            summary=summarize(conv), queue=queue, scheduled_for=scheduled_for,
            metadata=mask_obj(meta))
        self.records.append(rec)
        log.info("escalation %s rule=%s queue=%s conv=%s", rec.id, rule, queue, conv.session_id)
        return rec

    # -- background sweep (the 15-minute rule must fire even if the customer goes silent) ----
    def check_timeouts(self, conversations: list[Conversation]) -> list[EscalationRecord]:
        now = self._clock.now()
        out = []
        for conv in conversations:
            t = self._timeout_trigger(conv, now)
            if t and "unresolved_negative" not in conv.fired_rules:
                conv.fired_rules.add("unresolved_negative")
                out.append(self._make_record(
                    "escalation", conv, t[1], "unresolved_negative",
                    {**t[2], "source": "background_sweep"}, urgent=False))
        return out

    # -- SLA breach escalation (used by Task 2) --------------------------------------------
    def escalate_sla_breach(self, ticket_id: str, customer_id: str, summary: str, meta: dict) -> EscalationRecord:
        now = self._clock.now()
        self._counter += 1
        rec = EscalationRecord(
            id=f"ESC-{self._counter:04d}", kind="escalation", customer_id=customer_id,
            conversation_id=meta.get("conversation_id", ""), reason=f"SLA breached for ticket {ticket_id}",
            rule="sla_breach", timestamp=now, summary=summary,
            queue="agent_queue" if self._calendar.is_open(now) else "on_call", metadata=mask_obj(meta))
        self.records.append(rec)
        log.info("escalation %s rule=sla_breach ticket=%s", rec.id, ticket_id)
        return rec
