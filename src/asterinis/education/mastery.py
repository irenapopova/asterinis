"""Mastery updates and adaptive next-lesson recommendations."""

from __future__ import annotations

from .models import AssessmentResult, Course, LearnerProfile


class MasteryTracker:
    """Update concept mastery using an exponential moving average."""

    def __init__(self, *, learning_rate: float = 0.3, pass_threshold: float = 0.7) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be between 0 and 1.")
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be between 0 and 1.")
        self.learning_rate = learning_rate
        self.pass_threshold = pass_threshold

    def record(self, profile: LearnerProfile, result: AssessmentResult) -> float:
        if profile.learner_id != result.learner_id:
            raise ValueError("assessment learner does not match profile.")
        previous = profile.mastered_concepts.get(result.concept, 0.0)
        updated = previous + self.learning_rate * (result.score - previous)
        profile.mastered_concepts[result.concept] = round(updated, 6)
        if result.lesson_id and result.passed:
            profile.completed_lessons.add(result.lesson_id)
        return profile.mastered_concepts[result.concept]

    def next_lesson(self, profile: LearnerProfile, course: Course):
        """Return the first lesson with unmet prerequisites and weak concepts."""
        for lesson in course.lessons:
            if lesson.lesson_id in profile.completed_lessons:
                continue
            if any(profile.mastered_concepts.get(item, 0.0) < self.pass_threshold for item in lesson.prerequisites):
                continue
            if not lesson.concepts or any(profile.mastered_concepts.get(item, 0.0) < self.pass_threshold for item in lesson.concepts):
                return lesson
        return None
