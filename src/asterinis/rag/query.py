from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalQuery:
    text: str
    limit: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = self.text.strip()

        if not self.text:
            raise ValueError("Query text cannot be empty.")

        if self.limit < 1:
            raise ValueError("Query limit must be at least 1.")