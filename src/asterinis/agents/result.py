from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentResult:
    agent: str
    output: Any
    succeeded: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
