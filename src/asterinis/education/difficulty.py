"""Exercise difficulty estimation from learner outcomes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DifficultyEstimate:
    item_id: str
    difficulty: float
    samples: int
    recommendation: str


class DifficultyEstimator:
    def estimate(
        self,
        item_id: str,
        *,
        scores: tuple[float, ...],
        average_attempts: float = 1.0,
        average_hints: float = 0.0,
    ) -> DifficultyEstimate:
        if not item_id.strip() or not scores:
            raise ValueError("item_id and scores are required.")
        if any(not 0.0 <= score <= 1.0 for score in scores):
            raise ValueError("scores must be between 0 and 1.")
        if average_attempts < 1 or average_hints < 0:
            raise ValueError("attempts must be at least 1 and hints cannot be negative.")
        success_difficulty = 1.0 - sum(scores) / len(scores)
        attempt_signal = min(1.0, (average_attempts - 1.0) / 4.0)
        hint_signal = min(1.0, average_hints / 5.0)
        difficulty = round(min(1.0, 0.7 * success_difficulty + 0.2 * attempt_signal + 0.1 * hint_signal), 4)
        recommendation = "increase difficulty" if difficulty < 0.3 else "keep difficulty" if difficulty < 0.7 else "simplify or add scaffolding"
        return DifficultyEstimate(item_id, difficulty, len(scores), recommendation)
