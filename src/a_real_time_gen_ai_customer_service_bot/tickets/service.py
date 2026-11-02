"""Ticket service: create tickets from unresolved conversations (Task 2)."""
from dataclasses import dataclass, field
from datetime import datetime

from ..core.business_calendar import BusinessCalendar
from ..core.clock import Clock
from ..core.config import Config
from ..core.logging import get_logger
from ..core.masking import mask_text
from ..core.text import tokens
from .models import Ticket, TicketDraft, TicketRepository
from .extractor import missing_fields
from .priority import compute_priority, impact_level, SEVERITY
from .routing import Router, RoutingDecision
from .sla import SLAEngine

log = get_logger("tickets")
_FIELD_QUESTIONS = {
    "order_id": "your order number", "issue": "a short description of the problem",
    "contact": "an email address or phone number where we can reach you",
    "product": "which product this is about", "customer_name": "your name",
    "evidence": "a screenshot, invoice or other proof",
}


@dataclass
class TicketResult:
    status: str                         # needs_info | created | duplicate
    ticket: Ticket | None = None
    missing: list[str] = field(default_factory=list)
    question: str = ""
    related: list[str] = field(default_factory=list)       # ids of grouped related tickets
    routing: RoutingDecision | None = None


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


class TicketService:
    def __init__(self, config: Config, calendar: BusinessCalendar, clock: Clock, sla: SLAEngine,
                 router: Router, repo: TicketRepository):
        self._config, self._calendar, self._clock = config, calendar, clock
        self.sla, self.router, self.repo = sla, router, repo

    # ---------------------------------------------------------------------------------------
    def request_missing(self, draft: TicketDraft) -> TicketResult | None:
        missing = missing_fields(draft, self._config)
        if not missing:
            return None
        asked = ", ".join(_FIELD_QUESTIONS.get(f, f) for f in missing)
        return TicketResult("needs_info", missing=missing, question=f"To open a ticket I still need: {asked}.")

    def create(self, draft: TicketDraft) -> TicketResult:
        need = self.request_missing(draft)
        if need:
            return need
        now = self._clock.now()

        dup, related = self._find_duplicate_and_related(draft)
        if dup:
            dup.duplicate_count += 1
            dup.history.append(f"{now.isoformat()} duplicate request merged")
            for e in draft.evidence:
                if e not in dup.evidence:
                    dup.evidence.append(e)
            return TicketResult("duplicate", dup, related=[r.ticket_id for r in related])

        risk = bool(draft.risk_intents)
        severity = SEVERITY.get(draft.issue_type, 1)
        impact = impact_level(draft.impact_text)
        ticket = Ticket(
            ticket_id=self.repo.next_ticket_id(), customer_id=draft.customer_id, conversation_id=draft.conversation_id,
            customer_name=draft.customer_name, order_id=draft.order_id, product=draft.product, issue=draft.issue or "",
            issue_type=draft.issue_type, evidence=list(draft.evidence), contact=dict(draft.contact),
            sentiment_label=draft.sentiment_label, created_at=now, impact=impact, severity=severity,
            risk_intents=list(draft.risk_intents))
        self.reprioritise(ticket)
        if related:
            gid = next((r.group_id for r in related if r.group_id), None) or self.repo.next_group_id()
            for r in related:
                r.group_id = gid
                r.history.append(f"{now.isoformat()} grouped with {ticket.ticket_id}")
            ticket.group_id = gid
        self.repo.add(ticket)
        decision = self.router.route(ticket)
        self.router.apply(ticket, decision)
        ticket.history.append(f"{now.isoformat()} created priority={ticket.priority}")
        log.info("ticket %s created priority=%s queue=%s", ticket.ticket_id, ticket.priority, ticket.queue)
        return TicketResult("created", ticket, related=[r.ticket_id for r in related], routing=decision)

    # ---------------------------------------------------------------------------------------
    def reprioritise(self, ticket: Ticket) -> None:
        """Recompute priority (waiting time and SLA state change as time passes)."""
        now = self._clock.now()
        if not ticket.sla_priority:                               # creation: fix the SLA tier
            base_label, _, _ = compute_priority(ticket.issue_type, ticket.sentiment_label, ticket.impact, 0.0,
                                                False, bool(ticket.risk_intents), self._config)
            ticket.sla_priority = base_label
        allowed = self._config.get("sla.response_minutes")[ticket.sla_priority]
        waited = self._calendar.elapsed_minutes(ticket.created_at, now) / allowed
        label, score, breakdown = compute_priority(ticket.issue_type, ticket.sentiment_label, ticket.impact, waited,
                                                   ticket.sla_breached, bool(ticket.risk_intents), self._config)
        ticket.priority, ticket.priority_score, ticket.priority_breakdown = label, score, breakdown

    def _find_duplicate_and_related(self, d: TicketDraft):
        threshold = self._config.get("tickets.duplicate_similarity")
        dup, related = None, []
        for t in self.repo.open_tickets():
            sim = _jaccard(d.issue or "", t.issue)
            same_customer = t.customer_id == d.customer_id
            same_order = bool(d.order_id) and (t.order_id or "").lower() == d.order_id.lower()
            if same_customer and same_order:
                if t.issue_type == d.issue_type and sim >= threshold:
                    dup = dup or t
                else:
                    related.append(t)                                    # same order, different problem -> group
            elif (not same_customer and d.product and (t.product or "").lower() == d.product.lower()
                  and t.issue_type == d.issue_type and sim >= threshold):
                related.append(t)                                        # same incident across customers -> group
        return dup, related

    # ---------------------------------------------------------------------------------------
    def handoff_summary(self, ticket: Ticket) -> str:
        """Masked summary for the receiving agent: no raw contact details, card numbers or credentials."""
        contact = ", ".join(f"{k}={mask_text(v)}" for k, v in ticket.contact.items()) or "none"
        deadline = self.sla.deadline(ticket).isoformat()
        lines = [
            f"Ticket {ticket.ticket_id} | priority {ticket.priority} (score {ticket.priority_score:g}) | status {ticket.status}",
            f"Customer: {mask_text(ticket.customer_id)}" + (f" ({ticket.customer_name})" if ticket.customer_name else ""),
            f"Order: {ticket.order_id or 'n/a'} | Product: {ticket.product or 'n/a'} | Type: {ticket.issue_type}",
            f"Issue: {mask_text(ticket.issue)}",
            f"Evidence: {'; '.join(mask_text(e) for e in ticket.evidence) or 'none'}",
            f"Contact (masked): {contact}",
            f"Sentiment: {ticket.sentiment_label} | SLA deadline: {deadline} | Routed: {ticket.assigned_agent or ticket.queue}",
        ]
        if ticket.group_id:
            lines.append(f"Group: {ticket.group_id}")
        if ticket.duplicate_count:
            lines.append(f"Duplicate requests merged: {ticket.duplicate_count}")
        return "\n".join(lines)

    def resolve(self, ticket_id: str) -> None:
        t = self.repo.get(ticket_id)
        if t:
            t.status = "resolved"
            t.history.append(f"{self._clock.now().isoformat()} resolved")

    def run_sla_checks(self) -> list:
        events = self.sla.check(self.repo.open_tickets())
        for t in self.repo.open_tickets():
            self.reprioritise(t)
        return events
