"""Build a ticket draft from a conversation. Only values the customer actually provided are used; nothing is invented."""
from ..core.config import Config
from ..core.masking import mask_text
from ..core.text import tokens
from ..language.intents import IntentMatch
from ..session.models import Conversation
from .models import TicketDraft

_PRIORITY_ORDER = ["account_compromise", "legal_threat", "duplicate_payment", "technical_issue",
                   "refund", "order_status", "cancel_order", "complaint"]
_COMPLAINT_INTENTS = set(_PRIORITY_ORDER)


def pick_issue_type(intent_names: list[str]) -> str:
    for name in _PRIORITY_ORDER:
        if name in intent_names:
            return name
    return "general"


def draft_from_conversation(conv: Conversation, intents: list[IntentMatch], config: Config,
                            high_risk: list[str] | None = None) -> TicketDraft:
    users = conv.user_messages()
    seen_intents: list[str] = []
    for m in users:
        for i in m.intents:
            if i not in seen_intents:
                seen_intents.append(i)
    seen_intents += [i.name for i in intents if i.name not in seen_intents]
    issue_type = pick_issue_type(seen_intents)

    # the issue text = latest customer message that expressed a complaint intent, else the latest message
    issue_msg = next((m for m in reversed(users) if _COMPLAINT_INTENTS & set(m.intents)), users[-1] if users else None)
    issue = mask_text(issue_msg.text)[:300] if issue_msg else None

    ent = conv.entities
    product = ent.get("product_codes", [None])[-1] if ent.get("product_codes") else None
    if product is None:
        known = {p.lower() for p in config.get("tickets.known_products")}
        for m in reversed(users):
            hit = next((t for t in tokens(m.text) if t in known), None)
            if hit:
                product = hit
                break

    evidence = [mask_text(f"{a.get('filename', 'file')}: {a.get('summary', '')}".strip(": ")) for a in conv.attachments]
    evidence += [f"error code {e}" for e in ent.get("error_codes", [])]
    evidence += [f"amount {a}" for a in ent.get("amounts", [])]
    contact = {}
    if ent.get("emails"):
        contact["email"] = ent["emails"][-1]
    if ent.get("phones"):
        contact["phone"] = ent["phones"][-1]
    return TicketDraft(
        customer_id=conv.customer_id, conversation_id=conv.session_id,
        customer_name=ent.get("names", [None])[-1] if ent.get("names") else None,
        order_id=ent.get("order_ids", [None])[-1] if ent.get("order_ids") else None,
        product=product, issue=issue, issue_type=issue_type, evidence=evidence, contact=contact,
        sentiment_label=users[-1].sentiment if users and users[-1].sentiment else "neutral",
        impact_text=" ".join(m.text for m in users), risk_intents=high_risk or [],
        language=conv.language)


def missing_fields(draft: TicketDraft, config: Config) -> list[str]:
    values = {
        "customer": draft.customer_id, "customer_name": draft.customer_name, "order_id": draft.order_id,
        "product": draft.product, "issue": draft.issue, "evidence": draft.evidence or None,
        "contact": draft.contact or None,
    }
    return [f for f in config.get("tickets.mandatory_fields") if not values.get(f)]
