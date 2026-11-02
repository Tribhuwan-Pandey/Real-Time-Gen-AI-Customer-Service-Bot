"""Monitoring: latency, failures, confidence and escalations, over a sliding window on the injected Clock."""
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..core.clock import Clock
from ..core.config import Config


@dataclass
class Alert:
    name: str
    message: str
    value: float
    threshold: float


class MetricsCollector:
    def __init__(self, config: Config, clock: Clock):
        self._config = config
        self._clock = clock
        self._events: list[tuple[datetime, str, str, float]] = []

    # -- recording
    MAX_EVENTS = 20000

    def _add(self, kind: str, key: str = "", value: float = 1.0) -> None:
        self._events.append((self._clock.now(), kind, key, float(value)))
        if len(self._events) > self.MAX_EVENTS:                 # a long-running process must not leak memory
            keep_minutes = max(self._config.get("monitoring.window_minutes") * 4, 24 * 60)
            cutoff = self._clock.now() - timedelta(minutes=keep_minutes)
            self._events = [e for e in self._events if e[0] >= cutoff][-self.MAX_EVENTS // 2:]

    def record_latency(self, name: str, ms: float) -> None:
        self._add("latency", name, ms)

    def record_failure(self, kind: str) -> None:
        self._add("failure", kind)

    def record_confidence(self, value: float) -> None:
        self._add("confidence", "", value)

    def record_escalation(self, rule: str) -> None:
        self._add("escalation", rule)

    # -- reading
    def _window(self, minutes: float | None) -> list[tuple[datetime, str, str, float]]:
        minutes = minutes if minutes is not None else self._config.get("monitoring.window_minutes")
        cutoff = self._clock.now() - timedelta(minutes=minutes)
        return [e for e in self._events if e[0] >= cutoff]

    def snapshot(self, window_minutes: float | None = None) -> dict:
        ev = self._window(window_minutes)
        lat = sorted(v for _, k, _, v in ev if k == "latency")
        conf = [v for _, k, _, v in ev if k == "confidence"]
        fails, escs = {}, {}
        for _, k, key, _ in ev:
            if k == "failure":
                fails[key] = fails.get(key, 0) + 1
            elif k == "escalation":
                escs[key] = escs.get(key, 0) + 1
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0
        return {
            "latency": {"count": len(lat), "avg_ms": round(sum(lat) / len(lat), 2) if lat else 0.0, "p95_ms": p95},
            "failures": {"count": sum(fails.values()), "by_kind": fails},
            "confidence": {"count": len(conf), "avg": round(sum(conf) / len(conf), 3) if conf else None},
            "escalations": {"count": sum(escs.values()), "by_rule": escs},
        }

    def alerts(self, window_minutes: float | None = None) -> list[Alert]:
        s, c, out = self.snapshot(window_minutes), self._config, []
        if s["latency"]["count"] and s["latency"]["avg_ms"] > c.get("monitoring.max_avg_latency_ms"):
            out.append(Alert("high_latency", "Average latency above limit", s["latency"]["avg_ms"],
                             c.get("monitoring.max_avg_latency_ms")))
        if s["confidence"]["count"] and s["confidence"]["avg"] < c.get("monitoring.min_avg_confidence"):
            out.append(Alert("low_confidence", "Average confidence below limit", s["confidence"]["avg"],
                             c.get("monitoring.min_avg_confidence")))
        if s["failures"]["count"] > c.get("monitoring.max_failures"):
            out.append(Alert("many_failures", "Failure count above limit", s["failures"]["count"],
                             c.get("monitoring.max_failures")))
        if s["escalations"]["count"] > c.get("monitoring.max_escalations"):
            out.append(Alert("many_escalations", "Escalation count above limit", s["escalations"]["count"],
                             c.get("monitoring.max_escalations")))
        return out
