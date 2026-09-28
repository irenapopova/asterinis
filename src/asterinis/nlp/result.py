from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Entity:
    text: str
    label: str
    confidence: float | None = None
    start: int | None = None
    end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = self.text.strip()
        self.label = self.label.strip()

        if not self.text:
            raise ValueError("Entity text cannot be empty.")

        if not self.label:
            raise ValueError("Entity label cannot be empty.")

        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Entity confidence must be between 0 and 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "confidence": self.confidence,
            "start": self.start,
            "end": self.end,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class Classification:
    label: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label = self.label.strip()

        if not self.label:
            raise ValueError("Classification label cannot be empty.")

        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "Classification confidence must be between 0 and 1."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class NLPResult:
    text: str
    entities: list[Entity] = field(default_factory=list)
    classifications: list[Classification] = field(default_factory=list)
    intent: str | None = None
    language: str | None = None
    complexity: float | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = self.text.strip()

        if not self.text:
            raise ValueError("NLP result text cannot be empty.")

        if self.complexity is not None and not 0.0 <= self.complexity <= 1.0:
            raise ValueError("complexity must be between 0 and 1.")

        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "entities": [
                entity.to_dict()
                for entity in self.entities
            ],
            "classifications": [
                item.to_dict()
                for item in self.classifications
            ],
            "intent": self.intent,
            "language": self.language,
            "complexity": self.complexity,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }