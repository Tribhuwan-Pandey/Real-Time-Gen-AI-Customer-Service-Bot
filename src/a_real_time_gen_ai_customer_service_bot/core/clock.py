from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone


class Clock(ABC):
    """Anything that can tell the current time."""

    @abstractmethod
    def now(self) -> datetime:
        """Return the current time as a timezone-aware datetime."""


class SystemClock(Clock):
    """Real time. Used in production."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class FakeClock(Clock):
    """Controllable time. Used in tests and simulations."""

    def __init__(self, start: datetime):
        self._now = self._require_aware(start)

    def now(self) -> datetime:
        return self._now

    def set(self, new_time: datetime) -> None:
        """Jump to any moment, forward or backward."""
        self._now = self._require_aware(new_time)

    def advance(self, **kwargs) -> None:
        """Move forward in time: advance(minutes=16), advance(days=1), etc."""
        delta = timedelta(**kwargs)
        if delta < timedelta(0):
            raise ValueError("advance() cannot go backwards; use set()")
        self._now += delta

    @staticmethod
    def _require_aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Clock requires timezone-aware datetimes")
        return value
