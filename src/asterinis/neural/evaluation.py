"""Entity-level evaluation for native sequence tagging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..nlp.native_sequence import NativeSequenceTagger, TokenPrediction
from ..nlp.training import TrainingSample


@dataclass(frozen=True, slots=True)
class EntityMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def _spans_from_predictions(predictions: Iterable[TokenPrediction]) -> set[tuple[str, int, int]]:
    spans: set[tuple[str, int, int]] = set()
    current: list[TokenPrediction] = []

    def flush() -> None:
        if current:
            spans.add((current[0].label[2:], current[0].start, current[-1].end))
            current.clear()

    for prediction in predictions:
        if prediction.label == "O":
            flush()
        elif prediction.label.startswith("B-"):
            flush()
            current.append(prediction)
        elif prediction.label.startswith("I-"):
            if not current or current[0].label[2:] != prediction.label[2:]:
                flush()
            current.append(prediction)
        else:
            flush()
    flush()
    return spans


def evaluate_ner(
    model: NativeSequenceTagger,
    samples: Iterable[TrainingSample],
) -> EntityMetrics:
    """Evaluate exact entity span matches, not only token accuracy."""
    expected: set[tuple[str, int, int]] = set()
    predicted: set[tuple[str, int, int]] = set()
    for sample in samples:
        expected.update(
            (annotation.label, annotation.start, annotation.end)
            for annotation in sample.annotations
            if annotation.start is not None and annotation.end is not None
        )
        predicted.update(_spans_from_predictions(model.predict(sample.text)))

    true_positives = len(expected & predicted)
    false_positives = len(predicted - expected)
    false_negatives = len(expected - predicted)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EntityMetrics(true_positives, false_positives, false_negatives, precision, recall, f1)


__all__ = ["EntityMetrics", "evaluate_ner"]
