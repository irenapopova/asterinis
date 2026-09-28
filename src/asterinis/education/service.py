"""High-level education workflow orchestration."""

from __future__ import annotations

from typing import Mapping

from .grading import AssessmentEngine, AssessmentGrade
from .models import AssessmentResult, LearnerProfile
from .progress import ProgressTracker
from .questions import Assessment
from .storage import EducationStore


class EducationService:
    """Connect assessments, mastery updates, and persistent learner progress."""

    def __init__(
        self,
        store: EducationStore,
        *,
        assessment_engine: AssessmentEngine | None = None,
        progress: ProgressTracker | None = None,
    ) -> None:
        self.store = store
        self.assessments = assessment_engine or AssessmentEngine()
        self.progress = progress or ProgressTracker(store)

    def submit_assessment(
        self,
        profile: LearnerProfile,
        assessment: Assessment,
        answers: Mapping[str, object],
        *,
        lesson_id: str | None = None,
    ) -> AssessmentGrade:
        """Grade an assessment and update mastery for every tested concept."""
        result = self.assessments.grade(assessment, answers)
        for question, grade in zip(assessment.questions, result.questions):
            self.progress.record_assessment(
                profile,
                AssessmentResult(
                    learner_id=profile.learner_id,
                    concept=question.concept,
                    score=grade.score,
                    passed=grade.correct,
                    metadata={"assessment_id": assessment.assessment_id},
                ),
            )
        if result.passed and lesson_id:
            self.progress.record_lesson_completion(profile, lesson_id)
        return result

    def close(self) -> None:
        self.store.close()
