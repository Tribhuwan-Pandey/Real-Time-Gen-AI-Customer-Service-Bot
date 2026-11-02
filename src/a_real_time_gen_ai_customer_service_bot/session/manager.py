import threading
from datetime import timedelta

from ..core.clock import Clock
from ..core.config import Config
from .models import Conversation
from .summary import summarize


class SessionManager:
    """One active conversation per customer, fully isolated from every other customer.

    Rules (all from Config, read on every call so they can change at runtime):
      * session ends after `session.timeout_minutes` of inactivity (idle >= timeout)
      * customer returns within `session.restore_hours` of last activity -> previous summary restored
      * otherwise a brand-new session starts
    """

    def __init__(self, config: Config, clock: Clock):
        self._config = config
        self._clock = clock
        self._active: dict[str, Conversation] = {}
        self._archive: dict[str, Conversation] = {}
        self._lock = threading.RLock()
        self._counter = 0

    # -- helpers -------------------------------------------------------------
    def _timeout(self) -> timedelta:
        return timedelta(minutes=self._config.get("session.timeout_minutes"))

    def _restore_window(self) -> timedelta:
        return timedelta(hours=self._config.get("session.restore_hours"))

    def _end(self, conv: Conversation, reason: str) -> None:
        conv.summary = summarize(conv)
        conv.ended_at = self._clock.now()
        conv.end_reason = reason
        self._archive[conv.customer_id] = conv
        self._active.pop(conv.customer_id, None)

    # -- public API ----------------------------------------------------------
    def get_or_create(self, customer_id: str) -> tuple[Conversation, str]:
        """Return (conversation, event) where event is one of
        continued | new | restored | new_after_restore_window."""
        if not customer_id:
            raise ValueError("customer_id is required")
        now = self._clock.now()
        with self._lock:
            conv = self._active.get(customer_id)
            if conv and now - conv.last_activity >= self._timeout():
                self._end(conv, "inactivity")
                conv = None
            if conv:
                return conv, "continued"
            prev = self._archive.get(customer_id)
            restored, event = None, "new"
            if prev is not None:
                if now - prev.last_activity <= self._restore_window():
                    restored, event = prev.summary or summarize(prev), "restored"
                else:
                    event = "new_after_restore_window"
                    self._archive.pop(customer_id, None)
            self._counter += 1
            conv = Conversation(
                session_id=f"S-{self._counter:05d}", customer_id=customer_id,
                started_at=now, last_activity=now, restored_summary=restored,
                language=prev.language if restored and prev else None)
            self._active[customer_id] = conv
            return conv, event

    def touch(self, conv: Conversation) -> None:
        conv.last_activity = self._clock.now()

    def end_session(self, customer_id: str, reason: str = "closed") -> Conversation | None:
        with self._lock:
            conv = self._active.get(customer_id)
            if conv:
                self._end(conv, reason)
            return conv

    def sweep(self) -> list[Conversation]:
        """End every session idle for >= the timeout. Safe to call from a background job."""
        now = self._clock.now()
        ended = []
        with self._lock:
            for conv in list(self._active.values()):
                if now - conv.last_activity >= self._timeout():
                    self._end(conv, "inactivity")
                    ended.append(conv)
        return ended

    def active_conversations(self) -> list[Conversation]:
        with self._lock:
            return list(self._active.values())

    def get_active(self, customer_id: str) -> Conversation | None:
        return self._active.get(customer_id)
