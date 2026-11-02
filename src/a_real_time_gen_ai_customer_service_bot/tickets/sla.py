"""SLA engine: deadline, warning at 75% (configurable), automatic escalation after breach.

Time is counted with BusinessCalendar (weekends and holidays excluded, mode from config).
Everything is recomputed from Config on every call, so runtime SLA changes apply immediately.
"""
from dataclasses import dataclass
from datetime import datetime

from ..core.business_calendar import BusinessCalendar
from ..core.clock import Clock
from ..core.config import Config
from ..core.logging import get_logger
from ..escalation.engine import EscalationEngine, EscalationRecord
from .models import Ticket

log = get_logger("sla")


@dataclass
class SLAEvent:
    ticket_id: str
    type: str                 # warning | breach
    consumed_ratio: float
    escalation: EscalationRecord | None = None


class SLAEngine:
    def __init__(self, config: Config, calendar: BusinessCalendar, clock: Clock, escalation: EscalationEngine):
        self._config, self._calendar, self._clock, self._escalation = config, calendar, clock, escalation

    def allowed_minutes(self, priority: str) -> float:
        return self._config.get("sla.response_minutes")[priority]   # looked up on every call -> runtime changes apply

    def deadline(self, ticket: Ticket) -> datetime:
        return self._calendar.add_minutes(ticket.created_at, self.allowed_minutes(ticket.sla_priority or ticket.priority))

    def consumed_minutes(self, ticket: Ticket, now: datetime | None = None) -> float:
        return self._calendar.elapsed_minutes(ticket.created_at, now or self._clock.now())

    def ratio(self, ticket: Ticket, now: datetime | None = None) -> float:
        return self.consumed_minutes(ticket, now) / self.allowed_minutes(ticket.sla_priority or ticket.priority)

    def status(self, ticket: Ticket, now: datetime | None = None) -> str:
        r = self.ratio(ticket, now)
        if r > 1.0 + 1e-9:
            return "breached"
        if r >= self._config.get("sla.warning_ratio") - 1e-9:
            return "warning"
        return "ok"

    def check(self, tickets: list[Ticket]) -> list[SLAEvent]:
        """Emit each warning / breach once per ticket. Breach escalates automatically."""
        events: list[SLAEvent] = []
        now = self._clock.now()
        for t in tickets:
            if t.status != "open":
                continue
            st, r = self.status(t, now), self.ratio(t, now)
            if st == "warning" and not t.sla_warned:
                t.sla_warned = True
                t.history.append(f"{now.isoformat()} SLA warning at {r:.0%}")
                events.append(SLAEvent(t.ticket_id, "warning", round(r, 3)))
            if st == "breached" and not t.sla_breached:
                t.sla_breached = True
                t.sla_warned = True
                rec = self._escalation.escalate_sla_breach(
                    t.ticket_id, t.customer_id, f"Ticket {t.ticket_id} ({t.priority}) breached its SLA",
                    {"priority": t.priority, "issue_type": t.issue_type, "ratio": round(r, 3),
                     "conversation_id": t.conversation_id})
                t.history.append(f"{now.isoformat()} SLA breached; escalated as {rec.id}")
                events.append(SLAEvent(t.ticket_id, "breach", round(r, 3), rec))
        return events
