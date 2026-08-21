from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


ContradictionComparator = Callable[[str, str], float]


@dataclass(slots=True)
class ContradictionPair:
    first_index: int
    second_index: int
    first_text: str
    second_text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_index": self.first_index,
            "second_index": self.second_index,
            "first_text": self.first_text,
            "second_text": self.second_text,
            "score": self.score,
        }


@dataclass(slots=True)
class ContradictionResult:
    detected: bool
    severity: float
    pairs: list[ContradictionPair] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "severity": self.severity,
            "pairs": [
                pair.to_dict()
                for pair in self.pairs
            ],
            "metadata": dict(self.metadata),
        }


class ContradictionAgent:
    """
    Evaluates retrieved evidence for conflicting statements.

    The comparison strategy is injected so callers can use a rule-based
    comparator, an NLI model, a local classifier, or an LLM-backed verifier.
    """

    name = "contradiction"

    def __init__(
        self,
        comparator: ContradictionComparator,
        *,
        threshold: float = 0.75,
        max_pairs: int = 100,
    ) -> None:
        if not callable(comparator):
            raise TypeError("comparator must be callable.")

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        if max_pairs < 1:
            raise ValueError(
                "max_pairs must be greater than zero."
            )

        self.comparator = comparator
        self.threshold = threshold
        self.max_pairs = max_pairs

    def evaluate(
        self,
        evidence: Iterable[str],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ContradictionResult:
        items = self._normalize_evidence(evidence)

        if len(items) < 2:
            return ContradictionResult(
                detected=False,
                severity=0.0,
                metadata={
                    **(metadata or {}),
                    "evidence_count": len(items),
                    "pairs_checked": 0,
                },
            )

        contradictions: list[ContradictionPair] = []
        pairs_checked = 0

        for first_index in range(len(items)):
            for second_index in range(
                first_index + 1,
                len(items),
            ):
                if pairs_checked >= self.max_pairs:
                    break

                score = float(
                    self.comparator(
                        items[first_index],
                        items[second_index],
                    )
                )

                if not 0.0 <= score <= 1.0:
                    raise ValueError(
                        "Comparator must return a score between 0 and 1."
                    )

                pairs_checked += 1

                if score >= self.threshold:
                    contradictions.append(
                        ContradictionPair(
                            first_index=first_index,
                            second_index=second_index,
                            first_text=items[first_index],
                            second_text=items[second_index],
                            score=score,
                        )
                    )

            if pairs_checked >= self.max_pairs:
                break

        severity = max(
            (pair.score for pair in contradictions),
            default=0.0,
        )

        return ContradictionResult(
            detected=bool(contradictions),
            severity=severity,
            pairs=contradictions,
            metadata={
                **(metadata or {}),
                "evidence_count": len(items),
                "pairs_checked": pairs_checked,
                "threshold": self.threshold,
            },
        )

    @staticmethod
    def _normalize_evidence(
        evidence: Iterable[str],
    ) -> list[str]:
        normalized: list[str] = []

        for item in evidence:
            if not isinstance(item, str):
                raise TypeError(
                    "Each evidence item must be a string."
                )

            item = item.strip()

            if not item:
                continue

            if item not in normalized:
                normalized.append(item)

        return normalized