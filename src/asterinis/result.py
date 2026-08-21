from dataclasses import dataclass
from typing import Any


@dataclass
class NexusResult:
    route: str
    payload: Any
    confidence: float = 1.0