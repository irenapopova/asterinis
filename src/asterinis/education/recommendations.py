"""Recommendations based on course prerequisites and learner mastery."""

from .mastery import MasteryTracker
from .models import Course, LearnerProfile, Lesson


class RecommendationEngine:
    def __init__(self, *, mastery: MasteryTracker | None = None) -> None:
        self.mastery = mastery or MasteryTracker()

    def next_lesson(self, profile: LearnerProfile, course: Course) -> Lesson | None:
        if profile.course_id != course.course_id:
            raise ValueError("profile course does not match course.")
        return self.mastery.next_lesson(profile, course)
