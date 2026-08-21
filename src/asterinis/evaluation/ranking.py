from __future__ import annotations

import math
from collections.abc import Iterable


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be greater than zero.")


def precision_at_k(
    relevance: Iterable[bool],
    *,
    k: int,
) -> float:
    _validate_k(k)

    values = list(relevance)[:k]

    if not values:
        return 0.0

    return sum(values) / len(values)


def recall_at_k(
    relevance: Iterable[bool],
    *,
    total_relevant: int,
    k: int,
) -> float:
    _validate_k(k)

    if total_relevant < 0:
        raise ValueError(
            "total_relevant cannot be negative."
        )

    if total_relevant == 0:
        return 0.0

    values = list(relevance)[:k]

    return sum(values) / total_relevant


def hit_rate_at_k(
    relevance: Iterable[bool],
    *,
    k: int,
) -> float:
    _validate_k(k)

    values = list(relevance)[:k]

    return 1.0 if any(values) else 0.0


def reciprocal_rank(
    relevance: Iterable[bool],
) -> float:
    for rank, relevant in enumerate(
        relevance,
        start=1,
    ):
        if relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    rankings: Iterable[Iterable[bool]],
) -> float:
    scores = [
        reciprocal_rank(ranking)
        for ranking in rankings
    ]

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def dcg_at_k(
    relevance_scores: Iterable[float],
    *,
    k: int,
) -> float:
    _validate_k(k)

    scores = list(relevance_scores)[:k]

    total = 0.0

    for index, relevance in enumerate(
        scores,
        start=1,
    ):
        relevance = float(relevance)

        if relevance < 0:
            raise ValueError(
                "Relevance scores cannot be negative."
            )

        denominator = math.log2(index + 1)

        total += (
            (2**relevance - 1)
            / denominator
        )

    return total


def ndcg_at_k(
    relevance_scores: Iterable[float],
    *,
    k: int,
) -> float:
    _validate_k(k)

    scores = list(relevance_scores)

    if not scores:
        return 0.0

    actual = dcg_at_k(
        scores,
        k=k,
    )

    ideal = dcg_at_k(
        sorted(
            scores,
            reverse=True,
        ),
        k=k,
    )

    if ideal == 0.0:
        return 0.0

    return actual / ideal