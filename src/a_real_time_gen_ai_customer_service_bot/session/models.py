from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Message:
    role: str                      # "user" | "assistant"
    text: str
    ts: datetime
    language: str | None = None
    sentiment: str | None = None
    negative_like: bool = False
    intents: list[str] = field(default_factory=list)


@dataclass
class Conversation:
    session_id: str
    customer_id: str
    started_at: datetime
    last_activity: datetime
    messages: list[Message] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    language: str | None = None
    restored_summary: str | None = None
    negative_since: datetime | None = None
    resolved: bool = False
    fired_rules: set[str] = field(default_factory=set)
    pending: dict[str, Any] | None = None          # e.g. waiting for missing ticket fields
    sentiment_history: list[Any] = field(default_factory=list)
    last_intents: list[str] = field(default_factory=list)
    attachments: list[dict[str, Any]] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    last_question: str | None = None                              # last knowledge question (for follow-ups)
    purchase_dates: list[str] = field(default_factory=list)      # dates the customer tied to an order / payment
    ticket_ids: list[str] = field(default_factory=list)
    summary: str | None = None
    ended_at: datetime | None = None
    end_reason: str | None = None

    def recent(self, n: int) -> list[Message]:
        return self.messages[-n:]

    def user_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role == "user"]
