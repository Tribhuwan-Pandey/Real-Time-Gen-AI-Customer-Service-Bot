"""Scheduler: runs the update at a configurable time and retries failures after 15, 30 and 60 minutes.

Retry delays are measured from the previous failure: first retry +15 min, second +30 min, third +60 min,
then the attempt is given up until the next scheduled run (an alert event is emitted).
Everything reads Config on every tick, so `update_time` and `retry_delays_minutes` can change at runtime.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from ..core.clock import Clock
from ..core.config import Config
from ..core.logging import get_logger
from .pipeline import KBPipeline

log = get_logger("kb_scheduler")


@dataclass
class SchedulerEvent:
    ts: datetime
    type: str       # update_ok | update_failed | retry_scheduled | gave_up | activated_pending | health
    detail: str = ""


class UpdateScheduler:
    def __init__(self, config: Config, clock: Clock, pipeline: KBPipeline, role: str = "kb_manager", metrics=None):
        self._config, self._clock, self._pipeline, self._role = config, clock, pipeline, role
        self._metrics = metrics
        self._anchor = clock.now()           # next run is the first update_time AFTER this moment
        self._attempt = 0
        self.retry_at: datetime | None = None
        self.events: list[SchedulerEvent] = []

    def next_run(self) -> datetime:
        tz = ZoneInfo(self._config.get("business.timezone"))
        hh, mm = map(int, self._config.get("kb_pipeline.update_time").split(":"))
        local = self._anchor.astimezone(tz)
        candidate = local.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if candidate <= local:
            candidate += timedelta(days=1)
        return candidate.astimezone(timezone.utc)

    def _log(self, type_: str, detail: str = "") -> None:
        self.events.append(SchedulerEvent(self._clock.now(), type_, detail))

    def tick(self) -> list[SchedulerEvent]:
        """Call regularly (e.g. every minute). Returns events produced by this tick."""
        start = len(self.events)
        now = self._clock.now()
        if self._pipeline.activate_pending():
            self._log("activated_pending", "maintenance window opened")
        health = self._pipeline.health_tick()
        if health.status in ("rolled_back", "confirmed", "unhealthy_no_rollback"):
            self._log("health", f"{health.status}: {health.detail}")
        due = (now >= self.retry_at) if self.retry_at else (now >= self.next_run())
        if due:
            self._run(now)
        return self.events[start:]

    def _run(self, now: datetime) -> None:
        try:
            result = self._pipeline.run_update(self._role)
        except Exception as exc:
            self._on_failure(now, f"{type(exc).__name__}: {exc}")
            return
        self._log("update_ok", result.status)
        self._finish(now)

    def _on_failure(self, now: datetime, detail: str) -> None:
        self._log("update_failed", detail)
        if self._metrics:
            self._metrics.record_failure("kb_update_failed")
        delays = self._config.get("kb_pipeline.retry_delays_minutes")
        if self._attempt < len(delays):
            self.retry_at = now + timedelta(minutes=delays[self._attempt])
            self._log("retry_scheduled", f"attempt {self._attempt + 1} in {delays[self._attempt]} minutes")
            self._attempt += 1
        else:
            self._log("gave_up", f"all {len(delays)} retries failed; waiting for next scheduled run")
            if self._metrics:
                self._metrics.record_failure("kb_update_gave_up")
            self._finish(now)

    def _finish(self, now: datetime) -> None:
        self._attempt, self.retry_at, self._anchor = 0, None, now
