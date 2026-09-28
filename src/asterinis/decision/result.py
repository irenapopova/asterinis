from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Decision:
    provider: str
    capability: str
    score: float
    reason: str
    estimated_cost: float
    estimated_latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "capability": self.capability,
            "score": self.score,
            "reason": self.reason,
            "estimated_cost": self.estimated_cost,
            "estimated_latency_ms": self.estimated_latency_ms,
            "metadata": dict(self.metadata),
        }
