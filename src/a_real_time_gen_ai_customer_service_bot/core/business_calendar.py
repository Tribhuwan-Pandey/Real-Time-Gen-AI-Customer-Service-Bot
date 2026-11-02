from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterator
from zoneinfo import ZoneInfo

from .config import Config

_DAY_INDEX = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
_MAX_DAYS = 3660  # safety bound so a bad config can never loop forever


@dataclass(frozen=True)
class _Rules:
    """A snapshot of the config, so one calculation sees consistent values."""
    tz: ZoneInfo
    working_weekdays: frozenset
    holidays: frozenset
    open_time: time
    close_time: time
    mode: str


class BusinessCalendar:
    """Answers every time question that depends on hours, weekends, holidays.

    Settings are read from Config on every call, so a runtime change
    (new holiday, new hours, new SLA mode) takes effect immediately.
    All returned datetimes are timezone-aware UTC.
    """

    def __init__(self, config: Config):
        self._config = config

    # -- configuration snapshot ------------------------------------------
    def _rules(self) -> _Rules:
        c = self._config
        return _Rules(
            tz=ZoneInfo(c.get("business.timezone")),
            working_weekdays=frozenset(_DAY_INDEX[d] for d in c.get("business.working_days")),
            holidays=frozenset(h["date"] for h in c.get("holidays")),
            open_time=datetime.strptime(c.get("business.hours.start"), "%H:%M").time(),
            close_time=datetime.strptime(c.get("business.hours.end"), "%H:%M").time(),
            mode=c.get("sla.count_mode"),
        )

    @staticmethod
    def _require_aware(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            raise ValueError("BusinessCalendar requires timezone-aware datetimes")
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _is_working_date(d: date, r: _Rules) -> bool:
        return d.weekday() in r.working_weekdays and d not in r.holidays

    @staticmethod
    def _at(d: date, t: time, r: _Rules) -> datetime:
        """Local wall-clock time on date d, as UTC."""
        return datetime.combine(d, t, tzinfo=r.tz).astimezone(timezone.utc)

    # -- simple questions -------------------------------------------------
    def is_working_day(self, dt: datetime) -> bool:
        r = self._rules()
        local = self._require_aware(dt).astimezone(r.tz)
        return self._is_working_date(local.date(), r)

    def is_open(self, dt: datetime) -> bool:
        """Working day and open_time <= local time < close_time."""
        r = self._rules()
        local = self._require_aware(dt).astimezone(r.tz)
        if not self._is_working_date(local.date(), r):
            return False
        return r.open_time <= local.time().replace(tzinfo=None) < r.close_time

    def next_open(self, dt: datetime) -> datetime:
        """The earliest moment >= dt when the business is open."""
        r = self._rules()
        utc = self._require_aware(dt)
        first = utc.astimezone(r.tz).date()
        for offset in range(_MAX_DAYS):
            d = first + timedelta(days=offset)
            if not self._is_working_date(d, r):
                continue
            opens, closes = self._at(d, r.open_time, r), self._at(d, r.close_time, r)
            if utc < opens:
                return opens
            if utc < closes:
                return utc
        raise ValueError("No working day found; check working_days and holidays")

    def next_working_day_start(self, dt: datetime) -> datetime:
        """Opening time on the next working day AFTER dt's local date.
        Used for 'schedule normal after-hours complaints for the next working day'."""
        r = self._rules()
        first = self._require_aware(dt).astimezone(r.tz).date()
        for offset in range(1, _MAX_DAYS):
            d = first + timedelta(days=offset)
            if self._is_working_date(d, r):
                return self._at(d, r.open_time, r)
        raise ValueError("No working day found; check working_days and holidays")

    # -- SLA time arithmetic ---------------------------------------------
    def _windows(self, first: date, r: _Rules) -> Iterator[tuple[datetime, datetime]]:
        """Consecutive (start, end) UTC windows during which SLA time counts."""
        for offset in range(_MAX_DAYS):
            d = first + timedelta(days=offset)
            if not self._is_working_date(d, r):
                continue
            if r.mode == "calendar":
                yield self._at(d, time(0, 0), r), self._at(d + timedelta(days=1), time(0, 0), r)
            else:  # business_hours
                yield self._at(d, r.open_time, r), self._at(d, r.close_time, r)

    def elapsed_minutes(self, start: datetime, end: datetime) -> float:
        """SLA minutes consumed between start and end (weekends/holidays excluded)."""
        r = self._rules()
        start, end = self._require_aware(start), self._require_aware(end)
        if end < start:
            raise ValueError("end must not be before start")
        total = timedelta(0)
        for w_start, w_end in self._windows(start.astimezone(r.tz).date(), r):
            if w_start >= end:
                break
            lo, hi = max(w_start, start), min(w_end, end)
            if hi > lo:
                total += hi - lo
        return total.total_seconds() / 60

    def add_minutes(self, start: datetime, minutes: float) -> datetime:
        """The deadline: the moment when `minutes` of SLA time have been consumed."""
        if minutes < 0:
            raise ValueError("minutes must not be negative")
        r = self._rules()
        start = self._require_aware(start)
        if minutes == 0:
            return start
        remaining = float(minutes)
        for w_start, w_end in self._windows(start.astimezone(r.tz).date(), r):
            begin = max(w_start, start)
            if begin >= w_end:
                continue
            available = (w_end - begin).total_seconds() / 60
            if remaining <= available:
                return begin + timedelta(minutes=remaining)
            remaining -= available
        raise ValueError("No working time found; check working_days and holidays")
