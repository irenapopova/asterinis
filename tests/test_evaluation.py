from asterinis.evaluation import (
    Trace,
    entity_overlap,
    precision_at_k,
    reciprocal_rank,
)


def test_precision_at_k() -> None:
    score = precision_at_k(
        [True, False, True],
        k=3,
    )

    assert score == 2 / 3


def test_precision_at_k_with_smaller_k() -> None:
    score = precision_at_k(
        [True, False, True],
        k=2,
    )

    assert score == 0.5


def test_precision_at_k_empty_input() -> None:
    score = precision_at_k(
        [],
        k=3,
    )

    assert score == 0.0


def test_reciprocal_rank() -> None:
    score = reciprocal_rank(
        [False, True, False]
    )

    assert score == 0.5


def test_reciprocal_rank_first_result() -> None:
    score = reciprocal_rank(
        [True, False, False]
    )

    assert score == 1.0


def test_reciprocal_rank_no_relevant_results() -> None:
    score = reciprocal_rank(
        [False, False, False]
    )

    assert score == 0.0


def test_entity_overlap() -> None:
    score = entity_overlap(
        ["Berlin", "TU Berlin"],
        ["TU Berlin", "Germany"],
    )

    assert score == 0.5


def test_entity_overlap_full_match() -> None:
    score = entity_overlap(
        ["Deutsche Bank", "Frankfurt"],
        ["Frankfurt", "Deutsche Bank", "Germany"],
    )

    assert score == 1.0


def test_entity_overlap_is_case_insensitive() -> None:
    score = entity_overlap(
        ["Berlin"],
        ["berlin"],
    )

    assert score == 1.0


def test_entity_overlap_empty_query_entities() -> None:
    score = entity_overlap(
        [],
        ["Berlin"],
    )

    assert score == 0.0


def test_trace_adds_event() -> None:
    trace = Trace()

    event = trace.add(
        "route_selected",
        route="rag",
    )

    assert len(trace) == 1
    assert event.name == "route_selected"
    assert event.metadata["route"] == "rag"


def test_trace_events_are_readable() -> None:
    trace = Trace()

    trace.add(
        "retrieval_completed",
        results=5,
        top_score=0.87,
    )

    event = trace.events[0]

    assert event.name == "retrieval_completed"
    assert event.metadata["results"] == 5
    assert event.metadata["top_score"] == 0.87


def test_trace_to_dict() -> None:
    trace = Trace()

    trace.add(
        "agent_selected",
        agent="retrieval-quality",
    )

    data = trace.to_dict()

    assert "events" in data
    assert len(data["events"]) == 1
    assert data["events"][0]["name"] == "agent_selected"
    assert data["events"][0]["metadata"]["agent"] == "retrieval-quality"


def test_trace_clear() -> None:
    trace = Trace()

    trace.add("first_event")
    trace.add("second_event")

    assert len(trace) == 2

    trace.clear()

    assert len(trace) == 0