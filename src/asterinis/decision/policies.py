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
    retry_attempts: int = 1
    retry_delay_seconds: float = 0.0
    circuit_failure_threshold: int = 3
    circuit_recovery_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.mode not in {"local", "balanced", "best_quality"}:
            raise ValueError(
                "mode must be local, balanced, or best_quality."
            )
        if self.max_cost is not None and self.max_cost < 0:
            raise ValueError("max_cost cannot be negative.")
        if self.max_latency_ms is not None and self.max_latency_ms < 0:
            raise ValueError("max_latency_ms cannot be negative.")
        if self.retry_attempts < 1:
            raise ValueError("retry_attempts must be greater than zero.")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative.")
        if self.circuit_failure_threshold < 1:
            raise ValueError("circuit_failure_threshold must be positive.")
        if self.circuit_recovery_seconds <= 0:
            raise ValueError("circuit_recovery_seconds must be positive.")
