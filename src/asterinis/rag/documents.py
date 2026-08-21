from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = self.id.strip()
        self.text = self.text.strip()

        if not self.id:
            raise ValueError("Document id cannot be empty.")

        if not self.text:
            raise ValueError("Document text cannot be empty.")


@dataclass(slots=True)
class RetrievalResult:
    document: Document
    score: float
    retriever: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": {
                "id": self.document.id,
                "text": self.document.text,
                "metadata": self.document.metadata,
            },
            "score": self.score,
            "retriever": self.retriever,
            "metadata": self.metadata,
        }