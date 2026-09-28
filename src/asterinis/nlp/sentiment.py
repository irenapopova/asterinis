from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


SentimentStrategy = Callable[[str], tuple[str, float | None]]


@dataclass(slots=True)
class SentimentResult:
    label: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label = self.label.strip().lower()
        if not self.label:
            raise ValueError("sentiment label cannot be empty.")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")


class SentimentAnalyzer:
    """Provider-neutral sentiment analyzer with an injectable strategy."""

    def __init__(self, strategy: SentimentStrategy | None = None) -> None:
        self.strategy = strategy or self._lexicon_strategy
        if not callable(self.strategy):
            raise TypeError("strategy must be callable.")

    def analyze(self, text: str) -> SentimentResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty.")
        label, confidence = self.strategy(text)
        return SentimentResult(label, confidence)

    @staticmethod
    def _lexicon_strategy(text: str) -> tuple[str, float]:
        words = {
            word.strip(".,!?;:()[]{}\"").lower()
            for word in text.split()
        }
        positive = {"good", "great", "excellent", " helpful", "love", "best"}
        negative = {"bad", "poor", "terrible", "awful", "hate", "worst"}
        positive_hits = len(words & {item.strip() for item in positive})
        negative_hits = len(words & negative)
        if positive_hits == negative_hits:
            return "neutral", 0.5
        if positive_hits > negative_hits:
            return "positive", min(1.0, 0.6 + 0.1 * positive_hits)
        return "negative", min(1.0, 0.6 + 0.1 * negative_hits)
