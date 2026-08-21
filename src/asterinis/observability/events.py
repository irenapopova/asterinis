from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class ObservabilityEvent:
    """
    Represents a single event emitted by Asterinis.

    Events are intentionally generic so they can be used for routing,
    retrieval, agents, providers, evaluation, and other framework activity.
    """

    name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        self.name = self.name.strip()

        if not self.name:
            raise ValueError("Event name cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
        }