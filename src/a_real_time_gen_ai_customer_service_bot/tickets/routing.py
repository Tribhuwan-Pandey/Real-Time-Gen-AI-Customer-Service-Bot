"""Ticket routing by skill, availability, workload and business hours.

open hours      -> least-loaded available agent with the skill (ties by agent id)
team unavailable-> fall back to an available 'general' agent, else park in '<skill>_backlog'
closed + critical/high-risk -> on-call agent with the skill, else the on-call pool
closed + normal -> 'after_hours' queue, scheduled for the next opening time
"""
from dataclasses import dataclass
from datetime import datetime

from ..core.business_calendar import BusinessCalendar
from ..core.clock import Clock
from ..core.config import Config
from .models import Agent, Ticket, TicketRepository


@dataclass
class RoutingDecision:
    agent_id: str | None
    queue: str
    reason: str
    scheduled_for: datetime | None = None
    needs_supervisor: bool = False


class Router:
    def __init__(self, config: Config, calendar: BusinessCalendar, clock: Clock,
                 agents: list[Agent], repo: TicketRepository):
        self._config, self._calendar, self._clock, self.agents, self._repo = config, calendar, clock, agents, repo

    def _usable(self, skill: str, on_call_only: bool = False) -> list[Agent]:
        cap = self._config.get("routing.max_workload")
        return [a for a in self.agents if skill in a.skills and a.available
                and self._repo.workload(a.agent_id) < cap and (a.on_call or not on_call_only)]

    @staticmethod
    def _least_loaded(agents: list[Agent], repo: TicketRepository) -> Agent:
        return min(agents, key=lambda a: (repo.workload(a.agent_id), a.agent_id))

    def route(self, ticket: Ticket) -> RoutingDecision:
        skill_map = self._config.get("routing.skill_map")
        skill = skill_map.get(ticket.issue_type, skill_map["general"])
        now = self._clock.now()
        urgent = ticket.priority == "critical" or bool(ticket.risk_intents)

        if not self._calendar.is_open(now):
            if urgent:
                on_call = self._usable(skill, on_call_only=True)
                if on_call:
                    a = self._least_loaded(on_call, self._repo)
                    return RoutingDecision(a.agent_id, "on_call", f"after hours, urgent: on-call {skill} agent")
                return RoutingDecision(None, "on_call_pool", f"after hours, urgent: no on-call {skill} agent free",
                                       needs_supervisor=True)
            return RoutingDecision(None, "after_hours", "after hours: scheduled for next opening",
                                   scheduled_for=self._calendar.next_open(now))

        usable = self._usable(skill)
        if usable:
            a = self._least_loaded(usable, self._repo)
            return RoutingDecision(a.agent_id, f"{skill}_queue", f"assigned to {skill} agent with lowest workload")
        team = [a for a in self.agents if skill in a.skills]
        general = self._usable("general") if skill != "general" else []
        if general:
            a = self._least_loaded(general, self._repo)
            reason = "team unavailable" if team else "no team for skill"
            return RoutingDecision(a.agent_id, "general_queue", f"{reason}: fallback to general agent")
        reason = "team unavailable or at capacity" if team else "no agent has the required skill"
        return RoutingDecision(None, f"{skill}_backlog", reason, needs_supervisor=True)

    def apply(self, ticket: Ticket, decision: RoutingDecision) -> None:
        ticket.assigned_agent, ticket.queue = decision.agent_id, decision.queue
        ticket.routing_reason, ticket.scheduled_for = decision.reason, decision.scheduled_for
        ticket.history.append(f"{self._clock.now().isoformat()} routed to {decision.agent_id or decision.queue}: {decision.reason}")

    def reroute_waiting(self) -> list[Ticket]:
        """Re-route parked tickets (backlog / after-hours / on-call pool) when conditions change."""
        moved = []
        for t in self._repo.open_tickets():
            if t.assigned_agent is None:
                d = self.route(t)
                if d.agent_id is not None:
                    self.apply(t, d)
                    moved.append(t)
        return moved
