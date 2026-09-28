from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


ClassificationStrategy = Callable[
    [str],
    list[tuple[str, float | None]],
]


@dataclass(slots=True)
class ClassificationResult:
    """
    A single classification prediction.
    """

    label: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.label, str):
            raise TypeError("label must be a string.")

        self.label = self.label.strip()

        if not self.label:
            raise ValueError("label cannot be empty.")

        if (
            self.confidence is not None
            and not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError(
                "confidence must be between 0 and 1."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


class TextClassifier:
    """
    Provider-neutral text classifier.

    The classification strategy is supplied by the application and can be
    backed by Flair, a local model, deterministic rules, or another service.
    """

    def __init__(
        self,
        strategy: ClassificationStrategy,
        *,
        minimum_confidence: float = 0.0,
        max_results: int | None = None,
    ) -> None:
        if not callable(strategy):
            raise TypeError("strategy must be callable.")

        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence must be between 0 and 1."
            )

        if max_results is not None and max_results < 1:
            raise ValueError(
                "max_results must be greater than zero."
            )

        self.strategy = strategy
        self.minimum_confidence = minimum_confidence
        self.max_results = max_results

    def classify(
        self,
        text: str,
    ) -> list[ClassificationResult]:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = text.strip()

        if not text:
            raise ValueError("text cannot be empty.")

        raw_results = self.strategy(text)

        if not isinstance(raw_results, list):
            raise TypeError(
                "classification strategy must return a list."
            )

        results: list[ClassificationResult] = []

        for item in raw_results:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
            ):
                raise TypeError(
                    "Each classification result must be "
                    "a (label, confidence) tuple."
                )

            label, confidence = item

            result = ClassificationResult(
                label=label,
                confidence=confidence,
            )

            if (
                result.confidence is not None
                and result.confidence
                < self.minimum_confidence
            ):
                continue

            results.append(result)

        results.sort(
            key=lambda result: (
                result.confidence
                if result.confidence is not None
                else -1.0
            ),
            reverse=True,
        )

        if self.max_results is not None:
            results = results[: self.max_results]

        return results