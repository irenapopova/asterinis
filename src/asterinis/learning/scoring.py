from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .records import StrategyRecord


@dataclass(slots=True)
class StrategyScore:
    strategy: str
    score: float
    sample_count: int
    success_rate: float
    average_quality: float | None
    average_latency: float | None
    average_cost: float | None
    average_confidence: float | None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "strategy": self.strategy,
            "score": self.score,
            "sample_count": self.sample_count,
            "success_rate": self.success_rate,
            "average_quality": self.average_quality,
            "average_latency": self.average_latency,
            "average_cost": self.average_cost,
            "average_confidence": self.average_confidence,
        }


@dataclass(slots=True)
class ScoringWeights:
    success: float = 0.35
    quality: float = 0.35
    latency: float = 0.15
    cost: float = 0.10
    confidence: float = 0.05

    def __post_init__(self) -> None:
        weights = [
            self.success,
            self.quality,
            self.latency,
            self.cost,
            self.confidence,
        ]

        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Scoring weights cannot be negative."
            )

        if sum(weights) <= 0:
            raise ValueError(
                "At least one scoring weight must be positive."
            )


class StrategyScorer:
    """
    Scores strategies from historical outcomes.

    Higher success, quality, and confidence improve the score.
    Higher latency and cost reduce it.
    """

    def __init__(
        self,
        *,
        weights: ScoringWeights | None = None,
        latency_reference: float = 1.0,
        cost_reference: float = 1.0,
    ) -> None:
        if latency_reference <= 0:
            raise ValueError(
                "latency_reference must be greater than zero."
            )

        if cost_reference <= 0:
            raise ValueError(
                "cost_reference must be greater than zero."
            )

        self.weights = weights or ScoringWeights()
        self.latency_reference = latency_reference
        self.cost_reference = cost_reference

    def score(
        self,
        strategy: str,
        records: list[StrategyRecord] | tuple[StrategyRecord, ...],
    ) -> StrategyScore:
        if not records:
            raise ValueError(
                "At least one strategy record is required."
            )

        success_rate = (
            sum(record.success for record in records)
            / len(records)
        )

        qualities = [
            record.quality_score
            for record in records
            if record.quality_score is not None
        ]

        latencies = [
            record.latency_seconds
            for record in records
            if record.latency_seconds is not None
        ]

        costs = [
            record.cost
            for record in records
            if record.cost is not None
        ]

        confidences = [
            record.confidence
            for record in records
            if record.confidence is not None
        ]

        average_quality = (
            mean(qualities)
            if qualities
            else None
        )

        average_latency = (
            mean(latencies)
            if latencies
            else None
        )

        average_cost = (
            mean(costs)
            if costs
            else None
        )

        average_confidence = (
            mean(confidences)
            if confidences
            else None
        )

        quality_component = (
            average_quality
            if average_quality is not None
            else 0.0
        )

        confidence_component = (
            average_confidence
            if average_confidence is not None
            else 0.0
        )

        latency_component = self._inverse_penalty(
            average_latency,
            self.latency_reference,
        )

        cost_component = self._inverse_penalty(
            average_cost,
            self.cost_reference,
        )

        raw_score = (
            self.weights.success * success_rate
            + self.weights.quality * quality_component
            + self.weights.latency * latency_component
            + self.weights.cost * cost_component
            + self.weights.confidence * confidence_component
        )

        normalized = raw_score / (
            self.weights.success
            + self.weights.quality
            + self.weights.latency
            + self.weights.cost
            + self.weights.confidence
        )

        return StrategyScore(
            strategy=strategy,
            score=max(0.0, min(1.0, normalized)),
            sample_count=len(records),
            success_rate=success_rate,
            average_quality=average_quality,
            average_latency=average_latency,
            average_cost=average_cost,
            average_confidence=average_confidence,
        )

    @staticmethod
    def _inverse_penalty(
        value: float | None,
        reference: float,
    ) -> float:
        if value is None:
            return 0.0

        return 1.0 / (
            1.0 + value / reference
        )