from dataclasses import dataclass, field
from typing import Any


@dataclass
class NexusResult:
    route: str
    provider: str | None
    output: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    @property
    def payload(self) -> Any:
        """Backward-compatible alias for the provider output."""
        return self.output
