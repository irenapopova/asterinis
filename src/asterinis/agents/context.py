from dataclasses import dataclass, field
from typing import Any

from .memory import AgentMemory


@dataclass(slots=True)
class AgentContext:
    memory: AgentMemory = field(
        default_factory=AgentMemory
    )
    metadata: dict[str, Any] = field(
        default_factory=dict
    )