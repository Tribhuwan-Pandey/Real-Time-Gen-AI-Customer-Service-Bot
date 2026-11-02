from ..core.masking import mask_text
from .models import Conversation


def summarize(conv: Conversation, max_last: int = 160) -> str:
    """Deterministic, MASKED conversation summary used for session restore, escalations and handoffs."""
    users = conv.user_messages()
    intents: list[str] = []
    for m in users:
        for i in m.intents:
            if i not in intents:
                intents.append(i)
    last = mask_text(users[-1].text)[:max_last] if users else ""
    sentiments = [m.sentiment for m in users if m.sentiment]
    orders = conv.entities.get("order_ids", [])
    parts = [
        f"Customer {mask_text(conv.customer_id)}: {len(users)} customer message(s)",
        f"language={conv.language or 'unknown'}",
        f"intents={', '.join(intents) or 'none'}",
        f"orders={', '.join(orders) or 'none'}",
        f"latest sentiment={sentiments[-1] if sentiments else 'n/a'}",
        f"resolved={'yes' if conv.resolved else 'no'}",
    ]
    if conv.ticket_ids:
        parts.append(f"tickets={', '.join(conv.ticket_ids)}")
    if last:
        parts.append(f'last message: "{last}"')
    return "; ".join(parts)
