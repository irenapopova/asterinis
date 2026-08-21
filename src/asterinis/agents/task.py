from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentTask:
    instruction: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.instruction = self.instruction.strip()

        if not self.instruction:
            raise ValueError(
                "Agent task instruction cannot be empty."
            )