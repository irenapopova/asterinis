from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


LanguageStrategy = Callable[
    [str],
    tuple[str, float | None],
]


@dataclass(slots=True)
class LanguageResult:
    language: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.language, str):
            raise TypeError("language must be a string.")

        self.language = self.language.strip().lower()

        if not self.language:
            raise ValueError("language cannot be empty.")

        if (
            self.confidence is not None
            and not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError(
                "confidence must be between 0 and 1."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


class LanguageDetector:
    """
    Provider-neutral language detector.

    The returned language identifier is intentionally not restricted to a
    specific standard. Applications may use ISO codes such as 'en', 'de',
    or 'bg', or another consistent naming scheme.
    """

    def __init__(
        self,
        strategy: LanguageStrategy,
    ) -> None:
        if not callable(strategy):
            raise TypeError("strategy must be callable.")

        self.strategy = strategy

    def detect(
        self,
        text: str,
    ) -> LanguageResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = text.strip()

        if not text:
            raise ValueError("text cannot be empty.")

        language, confidence = self.strategy(text)

        return LanguageResult(
            language=language,
            confidence=confidence,
        )