from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .scoring import StrategyScore, StrategyScorer
from .store import StrategyStore


@dataclass(slots=True)
class StrategySelection:
    strategy: str
    score: float
    candidates: list[StrategyScore]
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "score": self.score,
            "reason": self.reason,
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
            "metadata": dict(self.metadata),
        }


class AdaptiveStrategySelector:
    """
    Selects the best-performing strategy for a query type using recorded
    historical outcomes.

    The selector is deterministic and explainable: every selection can be
    traced back to the scores of the available strategies.
    """

    def __init__(
        self,
        store: StrategyStore,
        *,
        scorer: StrategyScorer | None = None,
        minimum_samples: int = 1,
    ) -> None:
        if not isinstance(store, StrategyStore):
            raise TypeError(
                "store must be a StrategyStore."
            )

        if minimum_samples < 1:
            raise ValueError(
                "minimum_samples must be greater than zero."
            )

        self.store = store
        self.scorer = scorer or StrategyScorer()
        self.minimum_samples = minimum_samples

    def select(
        self,
        query_type: str,
    ) -> StrategySelection:
        query_type = query_type.strip()

        if not query_type:
            raise ValueError(
                "query_type cannot be empty."
            )

        grouped = self.store.grouped_by_strategy(
            query_type=query_type
        )

        candidates: list[StrategyScore] = []

        for strategy, records in grouped.items():
            if len(records) < self.minimum_samples:
                continue

            candidates.append(
                self.scorer.score(
                    strategy,
                    records,
                )
            )

        if not candidates:
            raise LookupError(
                f"No strategy has enough history for query type "
                f"'{query_type}'."
            )

        candidates.sort(
            key=lambda candidate: (
                candidate.score,
                candidate.sample_count,
            ),
            reverse=True,
        )

        selected = candidates[0]

        return StrategySelection(
            strategy=selected.strategy,
            score=selected.score,
            candidates=candidates,
            reason=(
                "Selected the strategy with the strongest "
                "historical performance for this query type."
            ),
            metadata={
                "query_type": query_type,
                "minimum_samples": self.minimum_samples,
            },
        )