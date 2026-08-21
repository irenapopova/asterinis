from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import uuid4

from .events import ObservabilityEvent


@dataclass(slots=True)
class TraceSummary:
    trace_id: str
    event_count: int
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "event_count": self.event_count,
            "duration_seconds": self.duration_seconds,
        }


class Trace:
    """
    Collects observability events for a single Asterinis operation.
    """

    def __init__(
        self,
        trace_id: str | None = None,
    ) -> None:
        self.trace_id = trace_id or uuid4().hex
        self._events: list[ObservabilityEvent] = []
        self._started_at = perf_counter()
        self._finished_at: float | None = None

    def record(
        self,
        name: str,
        **metadata: Any,
    ) -> ObservabilityEvent:
        event = ObservabilityEvent(
            name=name,
            metadata=metadata,
        )

        self._events.append(event)

        return event

    def finish(self) -> TraceSummary:
        if self._finished_at is None:
            self._finished_at = perf_counter()

        return self.summary()

    def summary(self) -> TraceSummary:
        end = self._finished_at or perf_counter()

        return TraceSummary(
            trace_id=self.trace_id,
            event_count=len(self._events),
            duration_seconds=end - self._started_at,
        )

    @property
    def events(self) -> tuple[ObservabilityEvent, ...]:
        return tuple(self._events)

    @property
    def finished(self) -> bool:
        return self._finished_at is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "events": [
                event.to_dict()
                for event in self._events
            ],
            "summary": self.summary().to_dict(),
        }