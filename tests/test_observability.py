import time

import pytest

from asterinis.observability import (
    MetricsCollector,
    ObservabilityEvent,
    Trace,
)


def test_event_keeps_name_and_metadata() -> None:
    event = ObservabilityEvent(
        name="route_selected",
        metadata={
            "route": "rag",
            "provider": "flair",
        },
    )

    assert event.name == "route_selected"
    assert event.metadata["route"] == "rag"
    assert event.metadata["provider"] == "flair"


def test_event_strips_whitespace_from_name() -> None:
    event = ObservabilityEvent(
        name="  retrieval_completed  "
    )

    assert event.name == "retrieval_completed"


def test_event_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ObservabilityEvent(name="   ")


def test_event_serialization() -> None:
    event = ObservabilityEvent(
        name="agent_selected",
        metadata={
            "agent": "verification",
        },
    )

    data = event.to_dict()

    assert data["name"] == "agent_selected"
    assert data["metadata"] == {
        "agent": "verification"
    }
    assert isinstance(data["timestamp"], str)


def test_metrics_increment_counter() -> None:
    metrics = MetricsCollector()

    metrics.increment("requests")
    metrics.increment("requests", 2)

    assert metrics.counter("requests") == 3


def test_unknown_counter_returns_zero() -> None:
    metrics = MetricsCollector()

    assert metrics.counter("unknown") == 0


def test_metrics_reject_negative_increment() -> None:
    metrics = MetricsCollector()

    with pytest.raises(ValueError):
        metrics.increment(
            "requests",
            -1,
        )


def test_metrics_record_observations() -> None:
    metrics = MetricsCollector()

    metrics.observe(
        "request_duration",
        0.15,
    )
    metrics.observe(
        "request_duration",
        0.25,
    )

    assert metrics.observations(
        "request_duration"
    ) == (0.15, 0.25)


def test_metrics_reject_negative_observation() -> None:
    metrics = MetricsCollector()

    with pytest.raises(ValueError):
        metrics.observe(
            "request_duration",
            -0.1,
        )


def test_metrics_reject_empty_name() -> None:
    metrics = MetricsCollector()

    with pytest.raises(ValueError):
        metrics.increment("   ")


def test_metrics_snapshot_is_independent() -> None:
    metrics = MetricsCollector()

    metrics.increment("requests")
    metrics.observe(
        "request_duration",
        0.10,
    )

    snapshot = metrics.snapshot()

    metrics.increment("requests")
    metrics.observe(
        "request_duration",
        0.20,
    )

    assert snapshot.counters["requests"] == 1
    assert snapshot.timings["request_duration"] == [0.10]

    assert metrics.counter("requests") == 2
    assert metrics.observations(
        "request_duration"
    ) == (0.10, 0.20)


def test_metrics_clear() -> None:
    metrics = MetricsCollector()

    metrics.increment("requests")
    metrics.observe(
        "request_duration",
        0.10,
    )

    metrics.clear()

    assert metrics.counter("requests") == 0
    assert metrics.observations(
        "request_duration"
    ) == ()


def test_trace_has_unique_identifier() -> None:
    first = Trace()
    second = Trace()

    assert first.trace_id
    assert second.trace_id
    assert first.trace_id != second.trace_id


def test_trace_accepts_custom_identifier() -> None:
    trace = Trace(
        trace_id="request-123"
    )

    assert trace.trace_id == "request-123"


def test_trace_records_events_in_order() -> None:
    trace = Trace()

    trace.record(
        "route_selected",
        route="rag",
    )
    trace.record(
        "retrieval_completed",
        results=5,
    )
    trace.record(
        "agent_selected",
        agent="retrieval-quality",
    )

    assert len(trace.events) == 3

    assert trace.events[0].name == "route_selected"
    assert trace.events[1].name == "retrieval_completed"
    assert trace.events[2].name == "agent_selected"


def test_trace_preserves_event_metadata() -> None:
    trace = Trace()

    event = trace.record(
        "provider_called",
        provider="flair",
        route="nlp",
    )

    assert event.metadata["provider"] == "flair"
    assert event.metadata["route"] == "nlp"


def test_trace_summary_reports_event_count() -> None:
    trace = Trace(
        trace_id="trace-test"
    )

    trace.record("first")
    trace.record("second")

    summary = trace.summary()

    assert summary.trace_id == "trace-test"
    assert summary.event_count == 2
    assert summary.duration_seconds >= 0.0


def test_trace_finish_marks_trace_as_finished() -> None:
    trace = Trace()

    assert not trace.finished

    summary = trace.finish()

    assert trace.finished
    assert summary.duration_seconds >= 0.0


def test_trace_finish_is_stable() -> None:
    trace = Trace()

    trace.record("route_selected")

    first = trace.finish()

    time.sleep(0.001)

    second = trace.finish()

    assert first.duration_seconds == second.duration_seconds


def test_trace_serialization() -> None:
    trace = Trace(
        trace_id="request-456"
    )

    trace.record(
        "route_selected",
        route="rag",
    )

    data = trace.to_dict()

    assert data["trace_id"] == "request-456"
    assert len(data["events"]) == 1

    assert (
        data["events"][0]["name"]
        == "route_selected"
    )

    assert (
        data["events"][0]["metadata"]["route"]
        == "rag"
    )

    assert data["summary"]["event_count"] == 1