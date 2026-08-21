from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class TraceEvent:
    """
    A single event recorded during an Asterinis workflow.
    """

    name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class Trace:
    """
    Lightweight trace collector for routing, retrieval,
    agent and provider decisions.
    """

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def add(
        self,
        name: str,
        **metadata: Any,
    ) -> TraceEvent:
        if not isinstance(name, str):
            raise TypeError("Trace event name must be a string.")

        name = name.strip()

        if not name:
            raise ValueError("Trace event name cannot be empty.")

        event = TraceEvent(
            name=name,
            metadata=metadata,
        )

        self._events.append(event)

        return event

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def clear(self) -> None:
        self._events.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [
                event.to_dict()
                for event in self._events
            ]
        }

    def __len__(self) -> int:
        return len(self._events)