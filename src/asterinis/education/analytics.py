"""Learning analytics and instructor dashboard summaries."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ProgressEvent


@dataclass(frozen=True, slots=True)
class LearnerAnalytics:
    learner_id: str
    events: int
    assessments: int
    average_score: float
    completed_lessons: int


class LearningAnalytics:
    def summarize(self, learner_id: str, events: tuple[ProgressEvent, ...]) -> LearnerAnalytics:
        learner_events = tuple(event for event in events if event.learner_id == learner_id)
        scores = [event.score for event in learner_events if event.score is not None]
        return LearnerAnalytics(
            learner_id,
            len(learner_events),
            sum(event.event_type == "assessment" for event in learner_events),
            sum(scores) / len(scores) if scores else 0.0,
            sum(event.event_type == "lesson_completed" for event in learner_events),
        )


@dataclass(frozen=True, slots=True)
class InstructorDashboard:
    learner_count: int
    average_score: float
    completion_count: int
    struggling_learners: tuple[str, ...]


def build_dashboard(summaries: tuple[LearnerAnalytics, ...], *, struggling_below: float = 0.5) -> InstructorDashboard:
    scores = [summary.average_score for summary in summaries]
    return InstructorDashboard(
        learner_count=len(summaries),
        average_score=sum(scores) / len(scores) if scores else 0.0,
        completion_count=sum(summary.completed_lessons for summary in summaries),
        struggling_learners=tuple(summary.learner_id for summary in summaries if summary.average_score < struggling_below),
    )
