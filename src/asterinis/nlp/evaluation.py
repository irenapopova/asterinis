from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class NLPEvaluation:
    total: int
    correct: int
    accuracy: float


def evaluate_labels(expected: Iterable[str], predicted: Iterable[str]) -> NLPEvaluation:
    expected_items = list(expected)
    predicted_items = list(predicted)
    if len(expected_items) != len(predicted_items):
        raise ValueError("expected and predicted must have equal lengths.")
    correct = sum(
        left.strip().lower() == right.strip().lower()
        for left, right in zip(expected_items, predicted_items)
    )
    total = len(expected_items)
    return NLPEvaluation(total, correct, correct / total if total else 0.0)
