"""Task 2 scenarios: duplicates, unavailable teams, after-hours, runtime SLA changes."""
from datetime import date

import pytest

from a_real_time_gen_ai_customer_service_bot.escalation.engine import EscalationEngine
from a_real_time_gen_ai_customer_service_bot.tickets.models import Agent, TicketDraft, TicketRepository
from a_real_time_gen_ai_customer_service_bot.tickets.routing import Router
from a_real_time_gen_ai_customer_service_bot.tickets.service import TicketService
from a_real_time_gen_ai_customer_service_bot.tickets.sla import SLAEngine
from conftest import ist


def default_agents():
    return [Agent("bill1", "Bina", {"billing"}), Agent("bill2", "Bala", {"billing"}, on_call=True),
            Agent("tech1", "Tara", {"technical"}), Agent("sec1", "Sam", {"security"}, on_call=True),
            Agent("legal1", "Lee", {"legal"}), Agent("gen1", "Gia", {"general"})]


class Stack:
    def __init__(self, config, clock, calendar, agents=None):
        self.config, self.clock, self.calendar = config, clock, calendar
        self.repo = TicketRepository()
        self.escalation = EscalationEngine(config, calendar, clock)
        self.sla = SLAEngine(config, calendar, clock, self.escalation)
        self.agents = agents if agents is not None else default_agents()
        self.router = Router(config, calendar, clock, self.agents, self.repo)
        self.service = TicketService(config, calendar, clock, self.sla, self.router, self.repo)


@pytest.fixture
def stack(config, clock, calendar):
    return Stack(config, clock, calendar)


def draft(**kw):
    base = dict(customer_id="c1", conversation_id="S-1", order_id="ORD-1001",
                issue="My router keeps disconnecting every hour", issue_type="technical_issue",
                contact={"email": "jane@example.com"}, sentiment_label="neutral")
    base.update(kw)
    return TicketDraft(**base)


# --- extraction / missing information ----------------------------------------------------------
def test_ticket_created_with_all_fields(stack):
    r = stack.service.create(draft(customer_name="Jane", product="router", evidence=["screenshot.png: error E1002"]))
    t = r.ticket
    assert r.status == "created" and t.customer_id == "c1" and t.order_id == "ORD-1001" and t.product == "router"
    assert t.issue and t.evidence == ["screenshot.png: error E1002"] and t.contact == {"email": "jane@example.com"}
    assert t.ticket_id == "TCK-0001" and t.created_at == stack.clock.now()


@pytest.mark.parametrize("field,expected", [("order_id", ["order_id"]), ("issue", ["issue"]), ("contact", ["contact"])])
def test_missing_mandatory_field_is_requested_not_invented(stack, field, expected):
    kw = {field: ({} if field == "contact" else None)}
    r = stack.service.create(draft(**kw))
    assert r.status == "needs_info" and r.missing == expected and r.ticket is None and r.question
    assert stack.repo.tickets == {}


def test_several_missing_fields_listed_together(stack):
    r = stack.service.create(draft(order_id=None, contact={}))
    assert r.missing == ["order_id", "contact"]


def test_mandatory_fields_are_configurable(stack, config):
    assert stack.service.create(draft()).status == "created"
    config.set("tickets.mandatory_fields", ["order_id", "issue", "contact", "product"])
    r = stack.service.create(draft(order_id="ORD-2002"))
    assert r.status == "needs_info" and r.missing == ["product"]


# --- priority ---------------------------------------------------------------------------------------
def test_priority_uses_severity_sentiment_and_impact(stack):
    low = stack.service.create(draft(order_id="ORD-1", issue_type="order_status", issue="Where is my parcel please")).ticket
    angry = stack.service.create(draft(order_id="ORD-2", issue_type="order_status", issue="Parcel never came, nobody helps",
                                       sentiment_label="frustrated")).ticket
    wide = stack.service.create(draft(order_id="ORD-3", issue_type="technical_issue", issue="Production is down for all users",
                                      impact_text="all users and our production are down", sentiment_label="urgent")).ticket
    assert low.priority_breakdown["sentiment"] == 0 and angry.priority_breakdown["sentiment"] == 2
    assert angry.priority_score > low.priority_score
    assert wide.priority_breakdown["impact"] == 3 and wide.priority_score > angry.priority_score
    assert wide.priority in ("high", "critical")


@pytest.mark.parametrize("issue_type", ["account_compromise", "duplicate_payment", "legal_threat"])
def test_calm_high_risk_tickets_are_at_least_high(stack, issue_type):
    t = stack.service.create(draft(issue_type=issue_type, issue="Hello, please check this", sentiment_label="neutral",
                                   risk_intents=[issue_type])).ticket
    assert t.priority in ("high", "critical")


def test_priority_thresholds_are_configurable(stack, config):
    t1 = stack.service.create(draft(order_id="ORD-5")).ticket
    config.set("tickets.priority_thresholds", {"critical": 4, "high": 3, "medium": 2})
    t2 = stack.service.create(draft(order_id="ORD-6")).ticket
    assert t2.priority in ("high", "critical") and t1.priority in ("low", "medium")


def test_waiting_time_raises_priority_but_not_the_sla_tier(stack, clock):
    t = stack.service.create(draft(issue_type="order_status", issue="Where is my parcel", order_id="ORD-7")).ticket
    tier, p0 = t.sla_priority, t.priority_score
    clock.advance(hours=18)                                    # 75% of the 'low' SLA is more than 54h; still inside
    stack.service.run_sla_checks()
    assert t.sla_priority == tier and t.priority_score >= p0
    assert not t.sla_breached


# --- SLA: deadline, 75% warning, breach escalation, weekends/holidays, runtime changes -----------------
def make_ticket(stack, priority_minutes=None, when=None, **kw):
    if when:
        stack.clock.set(when)
    t = stack.service.create(draft(**kw)).ticket
    if priority_minutes:
        stack.config.set("sla.response_minutes", {"critical": priority_minutes, "high": priority_minutes,
                                                  "medium": priority_minutes, "low": priority_minutes})
    return t


def test_warning_at_75_percent_exactly_and_not_before(stack, clock):
    t = make_ticket(stack, 400)                                # 400 min allowed -> warning at 300
    clock.advance(minutes=299)
    assert stack.sla.check([t]) == [] and stack.sla.status(t) == "ok"
    clock.advance(minutes=1)
    ev = stack.sla.check([t])
    assert [e.type for e in ev] == ["warning"] and stack.sla.status(t) == "warning"
    assert stack.sla.check([t]) == []                          # once only


def test_breach_after_allowed_time_escalates_automatically_once(stack, clock):
    t = make_ticket(stack, 400)
    clock.advance(minutes=400)
    assert stack.sla.status(t) == "warning"                    # exactly at the deadline is not yet a breach
    clock.advance(minutes=1)
    ev = stack.sla.check([t])
    assert "breach" in [e.type for e in ev]
    rec = [e.escalation for e in ev if e.type == "breach"][0]
    assert rec.rule == "sla_breach" and t.ticket_id in rec.reason and t.sla_breached
    assert stack.sla.check([t]) == [] and len(stack.escalation.records) == 1


def test_resolved_tickets_do_not_breach(stack, clock):
    t = make_ticket(stack, 100)
    stack.service.resolve(t.ticket_id)
    clock.advance(hours=5)
    assert stack.sla.check([t]) == []


def test_weekend_time_is_excluded_from_sla(stack, clock, config):
    config.set("sla.response_minutes", {"critical": 240, "high": 480, "medium": 1440, "low": 4320})
    clock.set(ist(16, 17, 0))                                  # Friday 17:00
    t = stack.service.create(draft(issue_type="technical_issue", issue="Router dead")).ticket
    allowed = stack.sla.allowed_minutes(t.sla_priority)
    clock.set(ist(19, 0, 0))                                   # Monday 00:00: weekend skipped
    assert stack.sla.consumed_minutes(t) == pytest.approx(7 * 60)
    assert stack.sla.deadline(t) == stack.calendar.add_minutes(t.created_at, allowed)
    assert stack.sla.status(t) in ("ok", "warning")


def test_holiday_is_excluded_from_sla(stack, clock, config):
    clock.set(ist(19, 23, 0))                                  # Monday 23:00; Tuesday 20 Oct is a holiday
    t = stack.service.create(draft(order_id="ORD-9")).ticket
    clock.set(ist(21, 1, 0))
    assert stack.sla.consumed_minutes(t) == 120


def test_runtime_sla_change_alters_status_and_deadline(stack, clock, config):
    t = stack.service.create(draft()).ticket
    clock.advance(minutes=100)
    before = stack.sla.deadline(t)
    assert stack.sla.status(t) == "ok"
    config.set("sla.response_minutes", {"critical": 120, "high": 120, "medium": 120, "low": 120})
    assert stack.sla.deadline(t) < before and stack.sla.status(t) == "warning"
    config.set("sla.warning_ratio", 0.9)
    assert stack.sla.status(t) == "ok"
    config.set("sla.response_minutes", {"critical": 60, "high": 60, "medium": 60, "low": 60})
    assert stack.sla.status(t) == "breached"


def test_runtime_holiday_and_mode_change_move_deadline(stack, clock, config):
    clock.set(ist(16, 17, 0))
    t = stack.service.create(draft()).ticket
    d1 = stack.sla.deadline(t)
    config.set("holidays", [{"date": date(2026, 10, 19), "name": "Surprise holiday"}])
    assert stack.sla.deadline(t) > d1
    config.set("sla.count_mode", "business_hours")
    assert stack.sla.deadline(t) != d1


# --- routing ------------------------------------------------------------------------------------------------
def test_routes_by_skill_and_lowest_workload(stack):
    a = stack.service.create(draft(order_id="ORD-A", issue_type="refund", issue="Refund not received yet")).ticket
    b = stack.service.create(draft(order_id="ORD-B", issue_type="refund", issue="Different refund problem here")).ticket
    assert {a.assigned_agent, b.assigned_agent} == {"bill1", "bill2"}
    t = stack.service.create(draft(order_id="ORD-C")).ticket
    assert t.assigned_agent == "tech1" and t.queue == "technical_queue"


def test_unavailable_team_falls_back_to_general_agent(stack):
    stack.agents[2].available = False                          # tech1 off
    t = stack.service.create(draft()).ticket
    assert t.assigned_agent == "gen1" and "fallback" in t.routing_reason


def test_unavailable_team_and_no_fallback_parks_ticket_for_supervisor(stack):
    for a in stack.agents:
        if "technical" in a.skills or "general" in a.skills:
            a.available = False
    r = stack.service.create(draft())
    assert r.ticket.assigned_agent is None and r.ticket.queue == "technical_backlog" and r.routing.needs_supervisor
    stack.agents[2].available = True                           # team comes back
    assert [t.ticket_id for t in stack.router.reroute_waiting()] == [r.ticket.ticket_id]
    assert r.ticket.assigned_agent == "tech1"


def test_workload_cap_is_respected_and_configurable(stack, config):
    config.set("routing.max_workload", 1)
    stack.service.create(draft(order_id="ORD-1"))
    second = stack.service.create(draft(order_id="ORD-2", issue="Totally different laptop battery problem"))
    assert second.ticket.assigned_agent == "gen1" and "fallback" in second.ticket.routing_reason


def test_team_with_no_agents_goes_to_backlog(config, clock, calendar):
    s = Stack(config, clock, calendar, agents=[Agent("x", "X", {"billing"})])
    t = s.service.create(draft()).ticket
    assert t.queue == "technical_backlog" and "no agent" in t.routing_reason


def test_after_hours_normal_ticket_is_scheduled_for_next_opening(stack, clock):
    clock.set(ist(16, 19, 0))                                  # Friday evening
    t = stack.service.create(draft(order_id="ORD-AH")).ticket
    assert t.queue == "after_hours" and t.assigned_agent is None and t.scheduled_for == ist(19, 9)
    clock.set(ist(19, 9, 0))
    assert [x.ticket_id for x in stack.router.reroute_waiting()] == [t.ticket_id]
    assert t.assigned_agent == "tech1"


def test_after_hours_urgent_ticket_goes_to_on_call_agent(stack, clock):
    clock.set(ist(12, 23, 0))
    t = stack.service.create(draft(issue_type="account_compromise", issue="Someone accessed my account",
                                   risk_intents=["account_compromise"])).ticket
    assert t.assigned_agent == "sec1" and t.queue == "on_call"


def test_after_hours_urgent_without_on_call_agent_uses_pool(stack, clock):
    stack.agents[3].on_call = False
    clock.set(ist(12, 23, 0))
    r = stack.service.create(draft(issue_type="account_compromise", issue="Someone accessed my account",
                                   risk_intents=["account_compromise"]))
    assert r.ticket.queue == "on_call_pool" and r.routing.needs_supervisor


def test_business_hours_change_at_runtime_changes_routing(stack, clock, config):
    clock.set(ist(12, 19, 0))
    config.set("business.hours.end", "20:00")
    assert stack.service.create(draft()).ticket.queue == "technical_queue"


# --- duplicates, grouping, separation ---------------------------------------------------------------------------
def test_duplicate_request_is_merged_not_recreated(stack):
    first = stack.service.create(draft(evidence=["a.png"])).ticket
    r = stack.service.create(draft(issue="my router keeps disconnecting every hour!!", evidence=["b.png"]))
    assert r.status == "duplicate" and r.ticket is first and first.duplicate_count == 1
    assert first.evidence == ["a.png", "b.png"] and len(stack.repo.tickets) == 1


def test_duplicate_threshold_is_configurable(stack, config):
    stack.service.create(draft())
    similar = draft(issue="router disconnecting often")
    assert stack.service.create(similar).status == "created"        # below 0.6 similarity: a new, grouped ticket
    config.set("tickets.duplicate_similarity", 0.1)
    assert stack.service.create(draft(issue="router disconnecting often again")).status == "duplicate"


def test_same_order_different_problem_is_grouped_not_merged(stack):
    a = stack.service.create(draft(issue_type="technical_issue")).ticket
    b = stack.service.create(draft(issue_type="refund", issue="I also want a refund for this order")).ticket
    assert a.group_id and a.group_id == b.group_id and a.ticket_id != b.ticket_id


def test_unrelated_issues_are_kept_separate(stack):
    a = stack.service.create(draft(order_id="ORD-1001")).ticket
    b = stack.service.create(draft(order_id="ORD-2002", issue_type="refund", issue="Refund for my other order")).ticket
    assert a.group_id is None and b.group_id is None and len(stack.repo.tickets) == 2


def test_same_incident_from_different_customers_is_grouped(stack):
    a = stack.service.create(draft(customer_id="c1", order_id="ORD-1", product="router")).ticket
    b = stack.service.create(draft(customer_id="c2", order_id="ORD-2", product="router")).ticket
    assert a.group_id and a.group_id == b.group_id
    c = stack.service.create(draft(customer_id="c3", order_id="ORD-3", product="laptop")).ticket
    assert c.group_id is None


def test_resolved_tickets_are_not_duplicate_targets(stack):
    first = stack.service.create(draft()).ticket
    stack.service.resolve(first.ticket_id)
    assert stack.service.create(draft()).status == "created"


# --- masked handoff -----------------------------------------------------------------------------------------------------
def test_handoff_summary_masks_personal_and_payment_data(stack):
    t = stack.service.create(draft(
        issue="Charged twice on card 4111 1111 1111 1111, call me on 9876543210 or mail jane@example.com",
        contact={"email": "jane@example.com", "phone": "9876543210"}, issue_type="duplicate_payment",
        risk_intents=["duplicate_payment"], customer_name="Jane")).ticket
    text = stack.service.handoff_summary(t)
    for secret in ("4111 1111", "9876543210", "jane@example.com"):
        assert secret not in text
    assert "ORD-1001" in text and "TCK-0001" in text and "SLA deadline" in text and "[EMAIL]" in text
