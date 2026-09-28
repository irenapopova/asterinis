from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DecisionPolicy:
    """Hard constraints and preferences for provider selection."""

    mode: str = "balanced"
    max_cost: float | None = None
    max_latency_ms: float | None = None
    allow_external: bool = True
    sensitive: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"local", "balanced", "best_quality"}:
            raise ValueError(
                "mode must be local, balanced, or best_quality."
            )
        if self.max_cost is not None and self.max_cost < 0:
            raise ValueError("max_cost cannot be negative.")
        if self.max_latency_ms is not None and self.max_latency_ms < 0:
            raise ValueError("max_latency_ms cannot be negative.")

