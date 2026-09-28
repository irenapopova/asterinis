from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class StrategyRecord:
    """
    Stores the outcome of one Asterinis strategy execution.

    A strategy can represent a routing choice, retrieval approach,
    provider combination, agent workflow, or other orchestration path.
    """

    strategy: str
    query_type: str
    success: bool
    quality_score: float | None = None
    latency_seconds: float | None = None
    cost: float | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        self.strategy = self.strategy.strip()
        self.query_type = self.query_type.strip()

        if not self.strategy:
            raise ValueError("strategy cannot be empty.")

        if not self.query_type:
            raise ValueError("query_type cannot be empty.")

        if (
            self.quality_score is not None
            and not 0.0 <= self.quality_score <= 1.0
        ):
            raise ValueError(
                "quality_score must be between 0 and 1."
            )

        if (
            self.confidence is not None
            and not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        if (
            self.latency_seconds is not None
            and self.latency_seconds < 0
        ):
            raise ValueError(
                "latency_seconds cannot be negative."
            )

        if self.cost is not None and self.cost < 0:
            raise ValueError(
                "cost cannot be negative."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "query_type": self.query_type,
            "success": self.success,
            "quality_score": self.quality_score,
            "latency_seconds": self.latency_seconds,
            "cost": self.cost,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }