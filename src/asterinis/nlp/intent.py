from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


IntentStrategy = Callable[[str], tuple[str, float | None]]


@dataclass(slots=True)
class IntentResult:
    intent: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        self.intent = self.intent.strip()

        if not self.intent:
            raise ValueError("Intent cannot be empty.")

        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Intent confidence must be between 0 and 1.")


class IntentDetector:
    """
    Provider-neutral intent detector.

    The detection strategy can be rule-based, model-based, or supplied
    by another NLP provider.
    """

    def __init__(
        self,
        strategy: IntentStrategy,
    ) -> None:
        if not callable(strategy):
            raise TypeError("strategy must be callable.")

        self.strategy = strategy

    def detect(
        self,
        text: str,
    ) -> IntentResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = text.strip()

        if not text:
            raise ValueError("text cannot be empty.")

        intent, confidence = self.strategy(text)

        return IntentResult(
            intent=intent,
            confidence=confidence,
        )