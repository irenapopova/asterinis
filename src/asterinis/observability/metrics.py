from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass(slots=True)
class MetricSnapshot:
    counters: dict[str, int]
    timings: dict[str, list[float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "counters": dict(self.counters),
            "timings": {
                name: list(values)
                for name, values in self.timings.items()
            },
        }


class MetricsCollector:
    """
    Lightweight in-process metrics collector.

    This collector does not depend on Prometheus or another monitoring
    backend. Exporters can read the collected metrics separately.
    """

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._timings: dict[str, list[float]] = {}
        self._lock = Lock()

    def increment(
        self,
        name: str,
        amount: int = 1,
    ) -> None:
        name = self._validate_name(name)

        if amount < 0:
            raise ValueError("Counter increment cannot be negative.")

        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def observe(
        self,
        name: str,
        value: float,
    ) -> None:
        name = self._validate_name(name)

        if value < 0:
            raise ValueError("Observed value cannot be negative.")

        with self._lock:
            self._timings.setdefault(name, []).append(float(value))

    def counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    def observations(self, name: str) -> tuple[float, ...]:
        with self._lock:
            return tuple(self._timings.get(name, ()))

    def snapshot(self) -> MetricSnapshot:
        with self._lock:
            return MetricSnapshot(
                counters=dict(self._counters),
                timings={
                    name: list(values)
                    for name, values in self._timings.items()
                },
            )

    def clear(self) -> None:
        with self._lock:
            self._counters.clear()
            self._timings.clear()

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str):
            raise TypeError("Metric name must be a string.")

        name = name.strip()

        if not name:
            raise ValueError("Metric name cannot be empty.")

        return name