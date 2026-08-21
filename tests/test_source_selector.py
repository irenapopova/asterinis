import pytest

from asterinis.rag.documents import (
    Document,
    RetrievalResult,
)
from asterinis.rag.source_selector import (
    SourceSelectionResult,
    SourceSelector,
)


def make_result(
    document_id: str,
    score: float,
    *,
    metadata: dict | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        document=Document(
            id=document_id,
            text=f"Document {document_id}",
        ),
        score=score,
        retriever="test",
        metadata=metadata or {},
    )


def test_source_selector_returns_highest_scoring_results() -> None:
    selector = SourceSelector()

    results = [
        make_result("a", 0.40),
        make_result("b", 0.95),
        make_result("c", 0.70),
    ]

    selection = selector.select(
        results,
        limit=2,
    )

    assert isinstance(
        selection,
        SourceSelectionResult,
    )

    assert [
        result.document.id
        for result in selection.selected
    ] == ["b", "c"]

    assert [
        result.document.id
        for result in selection.rejected
    ] == ["a"]


def test_source_selector_respects_minimum_score() -> None:
    selector = SourceSelector(
        minimum_score=0.75,
    )

    results = [
        make_result("a", 0.50),
        make_result("b", 0.90),
        make_result("c", 0.74),
    ]

    selection = selector.select(results)

    assert selection.selected_count == 1
    assert selection.selected[0].document.id == "b"

    assert {
        result.document.id
        for result in selection.rejected
    } == {"a", "c"}


def test_source_selector_uses_custom_scoring_function() -> None:
    selector = SourceSelector(
        score_function=lambda result: float(
            result.metadata.get(
                "quality",
                0.0,
            )
        )
    )

    first = make_result(
        "a",
        0.95,
        metadata={
            "quality": 0.20,
        },
    )

    second = make_result(
        "b",
        0.60,
        metadata={
            "quality": 0.90,
        },
    )

    selection = selector.select(
        [first, second],
        limit=1,
    )

    assert selection.selected_count == 1
    assert selection.selected[0].document.id == "b"


def test_source_selector_records_selection_scores() -> None:
    selector = SourceSelector()

    selection = selector.select(
        [
            make_result("a", 0.80),
            make_result("b", 0.60),
        ]
    )

    scores = {
        item.document_id: item.selection_score
        for item in selection.scores
    }

    assert scores["a"] == pytest.approx(0.80)
    assert scores["b"] == pytest.approx(0.60)


def test_source_selector_clamps_default_retrieval_score() -> None:
    selector = SourceSelector()

    result = make_result(
        "a",
        1.25,
    )

    selection = selector.select([result])

    assert (
        selection.scores[0].selection_score
        == 1.0
    )


def test_source_selector_rejects_invalid_custom_score() -> None:
    selector = SourceSelector(
        score_function=lambda result: 1.5
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        selector.select(
            [
                make_result(
                    "a",
                    0.50,
                )
            ]
        )


def test_source_selector_rejects_zero_limit() -> None:
    selector = SourceSelector()

    with pytest.raises(
        ValueError,
        match="limit",
    ):
        selector.select(
            [make_result("a", 0.80)],
            limit=0,
        )


def test_source_selector_rejects_invalid_result_type() -> None:
    selector = SourceSelector()

    with pytest.raises(TypeError):
        selector.select(
            ["not-a-retrieval-result"]  # type: ignore[list-item]
        )


def test_source_selector_keeps_metadata() -> None:
    selector = SourceSelector()

    selection = selector.select(
        [
            make_result(
                "a",
                0.80,
            )
        ],
        metadata={
            "request_id": "request-1",
        },
    )

    assert (
        selection.metadata["request_id"]
        == "request-1"
    )

    assert selection.metadata["limit"] == 5