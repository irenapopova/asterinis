from datetime import datetime, timezone

from asterinis.education import (
    BayesianKnowledgeTracer,
    ConceptGraph,
    DifficultyEstimator,
    LearnerProfile,
    SpacedRepetitionScheduler,
)


def test_spaced_repetition_increases_interval_after_success() -> None:
    scheduler = SpacedRepetitionScheduler()
    first = scheduler.review("loops", quality=5, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    second = scheduler.review("loops", quality=5, schedule=first, now=first.next_review)

    assert first.interval_days == 1
    assert second.interval_days == 6


def test_bayesian_knowledge_tracing_updates_concept_state() -> None:
    tracer = BayesianKnowledgeTracer()
    before = tracer.state("loops")
    after = tracer.observe("loops", correct=True)

    assert before.probability_known < after.probability_known
    assert after.observations == 1


def test_concept_graph_returns_ready_concepts_and_rejects_cycles() -> None:
    graph = ConceptGraph()
    graph.add_prerequisite("functions", "loops")
    profile = LearnerProfile("student", "python", {"loops": 0.8})

    assert graph.ready(profile) == ("functions",)
    try:
        graph.add_prerequisite("loops", "functions")
    except ValueError as error:
        assert "cycle" in str(error)
    else:
        raise AssertionError("concept cycles should be rejected")


def test_difficulty_estimator_uses_outcomes_and_hints() -> None:
    estimate = DifficultyEstimator().estimate(
        "exercise-1",
        scores=(0.2, 0.4),
        average_attempts=3,
        average_hints=2,
    )

    assert estimate.samples == 2
    assert estimate.difficulty > 0.5
