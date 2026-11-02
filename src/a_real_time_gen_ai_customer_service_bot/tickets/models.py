from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Agent:
    agent_id: str
    name: str
    skills: set[str]
    available: bool = True
    on_call: bool = False


@dataclass
class TicketDraft:
    customer_id: str
    conversation_id: str = ""
    customer_name: str | None = None
    order_id: str | None = None
    product: str | None = None
    issue: str | None = None
    issue_type: str = "general"
    evidence: list[str] = field(default_factory=list)
    contact: dict[str, str] = field(default_factory=dict)       # {"email": ..., "phone": ...}
    sentiment_label: str = "neutral"
    impact_text: str = ""                                        # raw text scanned for customer-impact cues
    risk_intents: list[str] = field(default_factory=list)
    language: str | None = None


@dataclass
class Ticket:
    ticket_id: str
    customer_id: str
    conversation_id: str
    customer_name: str | None
    order_id: str | None
    product: str | None
    issue: str
    issue_type: str
    evidence: list[str]
    contact: dict[str, str]
    sentiment_label: str
    created_at: datetime
    priority: str = "low"                     # dynamic: used for ordering / routing
    sla_priority: str = ""                    # SLA tier, fixed at creation (deadline never moves because of waiting)
    priority_score: float = 0.0
    priority_breakdown: dict[str, Any] = field(default_factory=dict)
    status: str = "open"                       # open | resolved
    assigned_agent: str | None = None
    queue: str | None = None
    routing_reason: str = ""
    scheduled_for: datetime | None = None
    group_id: str | None = None
    duplicate_count: int = 0
    impact: int = 1
    severity: int = 1
    sla_warned: bool = False
    sla_breached: bool = False
    risk_intents: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)


class TicketRepository:
    def __init__(self):
        self.tickets: dict[str, Ticket] = {}
        self._n = 0
        self._g = 0

    def next_ticket_id(self) -> str:
        self._n += 1
        return f"TCK-{self._n:04d}"

    def next_group_id(self) -> str:
        self._g += 1
        return f"GRP-{self._g:04d}"

    def add(self, t: Ticket) -> None:
        self.tickets[t.ticket_id] = t

    def get(self, ticket_id: str) -> Ticket | None:
        return self.tickets.get(ticket_id)

    def open_tickets(self) -> list[Ticket]:
        return [t for t in self.tickets.values() if t.status == "open"]

    def workload(self, agent_id: str) -> int:
        return sum(1 for t in self.open_tickets() if t.assigned_agent == agent_id)
