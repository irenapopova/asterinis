import pytest

from asterinis.evaluation.ranking import (
    dcg_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k() -> None:
    score = precision_at_k(
        [True, False, True],
        k=3,
    )

    assert score == pytest.approx(
        2 / 3
    )


def test_precision_at_k_uses_only_top_k() -> None:
    score = precision_at_k(
        [True, False, True],
        k=2,
    )

    assert score == pytest.approx(0.5)


def test_precision_at_k_empty_input() -> None:
    assert precision_at_k(
        [],
        k=5,
    ) == 0.0


def test_recall_at_k() -> None:
    score = recall_at_k(
        [True, False, True],
        total_relevant=4,
        k=3,
    )

    assert score == pytest.approx(0.5)


def test_recall_at_k_zero_relevant_documents() -> None:
    score = recall_at_k(
        [False, False],
        total_relevant=0,
        k=2,
    )

    assert score == 0.0


def test_recall_at_k_rejects_negative_total() -> None:
    with pytest.raises(ValueError):
        recall_at_k(
            [True],
            total_relevant=-1,
            k=1,
        )


def test_hit_rate_at_k_returns_one_when_hit_exists() -> None:
    score = hit_rate_at_k(
        [False, True, False],
        k=2,
    )

    assert score == 1.0


def test_hit_rate_at_k_returns_zero_without_hit() -> None:
    score = hit_rate_at_k(
        [False, False, True],
        k=2,
    )

    assert score == 0.0


def test_reciprocal_rank() -> None:
    score = reciprocal_rank(
        [False, True, False]
    )

    assert score == pytest.approx(0.5)


def test_reciprocal_rank_first_result() -> None:
    assert reciprocal_rank(
        [True, False]
    ) == 1.0


def test_reciprocal_rank_without_relevant_result() -> None:
    assert reciprocal_rank(
        [False, False]
    ) == 0.0


def test_mean_reciprocal_rank() -> None:
    score = mean_reciprocal_rank(
        [
            [True, False],
            [False, True],
            [False, False],
        ]
    )

    expected = (
        1.0
        + 0.5
        + 0.0
    ) / 3

    assert score == pytest.approx(
        expected
    )


def test_mean_reciprocal_rank_empty_input() -> None:
    assert mean_reciprocal_rank([]) == 0.0


def test_dcg_at_k_is_positive() -> None:
    score = dcg_at_k(
        [3, 2, 1],
        k=3,
    )

    assert score > 0.0


def test_dcg_at_k_rewards_high_relevance_early() -> None:
    good = dcg_at_k(
        [3, 2, 1],
        k=3,
    )

    bad = dcg_at_k(
        [1, 2, 3],
        k=3,
    )

    assert good > bad


def test_dcg_at_k_rejects_negative_relevance() -> None:
    with pytest.raises(ValueError):
        dcg_at_k(
            [2, -1, 1],
            k=3,
        )


def test_ndcg_at_k_is_one_for_ideal_order() -> None:
    score = ndcg_at_k(
        [3, 2, 1],
        k=3,
    )

    assert score == pytest.approx(1.0)


def test_ndcg_at_k_penalizes_non_ideal_order() -> None:
    score = ndcg_at_k(
        [1, 2, 3],
        k=3,
    )

    assert 0.0 < score < 1.0


def test_ndcg_at_k_empty_input() -> None:
    assert ndcg_at_k(
        [],
        k=3,
    ) == 0.0


def test_ranking_metrics_reject_invalid_k() -> None:
    with pytest.raises(ValueError):
        precision_at_k(
            [True],
            k=0,
        )

    with pytest.raises(ValueError):
        hit_rate_at_k(
            [True],
            k=0,
        )

    with pytest.raises(ValueError):
        dcg_at_k(
            [1],
            k=0,
        )