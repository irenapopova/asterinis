"""Progress event recording and feedback integration."""

from __future__ import annotations

from .mastery import MasteryTracker
from .models import AssessmentResult, LearnerProfile, ProgressEvent
from .storage import EducationStore


class ProgressTracker:
    """Record learner activity and update persistent mastery state."""

    def __init__(self, store: EducationStore, *, mastery: MasteryTracker | None = None) -> None:
        self.store = store
        self.mastery = mastery or MasteryTracker()

    def record_assessment(self, profile: LearnerProfile, result: AssessmentResult) -> float:
        score = self.mastery.record(profile, result)
        self.store.save_profile(profile)
        self.store.add_event(
            ProgressEvent(
                learner_id=profile.learner_id,
                course_id=profile.course_id,
                event_type="assessment",
                item_id=result.lesson_id or result.concept,
                score=result.score,
                metadata={"concept": result.concept, "passed": result.passed},
            )
        )
        return score

    def record_lesson_completion(self, profile: LearnerProfile, lesson_id: str) -> None:
        profile.completed_lessons.add(lesson_id)
        self.store.save_profile(profile)
        self.store.add_event(
            ProgressEvent(
                learner_id=profile.learner_id,
                course_id=profile.course_id,
                event_type="lesson_completed",
                item_id=lesson_id,
            )
        )
