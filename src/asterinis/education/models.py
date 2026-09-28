"""Framework-neutral education domain objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _required(value: str, name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{name} cannot be empty.")
    return value


@dataclass(frozen=True, slots=True)
class Lesson:
    lesson_id: str
    title: str
    content: str
    concepts: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    difficulty: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "lesson_id", _required(self.lesson_id, "lesson_id"))
        object.__setattr__(self, "title", _required(self.title, "title"))
        object.__setattr__(self, "content", _required(self.content, "content"))
        if not 0.0 <= self.difficulty <= 1.0:
            raise ValueError("difficulty must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class Course:
    course_id: str
    title: str
    lessons: tuple[Lesson, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "course_id", _required(self.course_id, "course_id"))
        object.__setattr__(self, "title", _required(self.title, "title"))
        lesson_ids = [lesson.lesson_id for lesson in self.lessons]
        if len(lesson_ids) != len(set(lesson_ids)):
            raise ValueError("course lesson IDs must be unique.")


@dataclass(slots=True)
class LearnerProfile:
    learner_id: str
    course_id: str
    mastered_concepts: dict[str, float] = field(default_factory=dict)
    completed_lessons: set[str] = field(default_factory=set)
    preferences: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.learner_id = _required(self.learner_id, "learner_id")
        self.course_id = _required(self.course_id, "course_id")
        for concept, score in self.mastered_concepts.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"mastery for {concept!r} must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class AssessmentResult:
    learner_id: str
    concept: str
    score: float
    passed: bool
    lesson_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required(self.learner_id, "learner_id")
        _required(self.concept, "concept")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    learner_id: str
    course_id: str
    event_type: str
    item_id: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _required(self.learner_id, "learner_id")
        _required(self.course_id, "course_id")
        _required(self.event_type, "event_type")
        _required(self.item_id, "item_id")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1.")
