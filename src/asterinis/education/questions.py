"""Question and assessment definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"


@dataclass(frozen=True, slots=True)
class Question:
    question_id: str
    prompt: str
    concept: str
    question_type: QuestionType
    answer: str | tuple[str, ...]
    options: tuple[str, ...] = ()
    points: float = 1.0
    explanation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in (
            (self.question_id, "question_id"),
            (self.prompt, "prompt"),
            (self.concept, "concept"),
        ):
            if not value.strip():
                raise ValueError(f"{name} cannot be empty.")
        if self.points <= 0:
            raise ValueError("points must be positive.")
        if self.question_type == QuestionType.MULTIPLE_CHOICE and len(self.options) < 2:
            raise ValueError("multiple-choice questions require at least two options.")
        if self.question_type == QuestionType.TRUE_FALSE and len(self.options) not in {0, 2}:
            raise ValueError("true/false questions may have zero or two options.")


@dataclass(frozen=True, slots=True)
class Assessment:
    assessment_id: str
    title: str
    questions: tuple[Question, ...]
    pass_threshold: float = 0.7

    def __post_init__(self) -> None:
        if not self.assessment_id.strip() or not self.title.strip():
            raise ValueError("assessment_id and title cannot be empty.")
        if not self.questions:
            raise ValueError("assessment must contain at least one question.")
        if len({question.question_id for question in self.questions}) != len(self.questions):
            raise ValueError("question IDs must be unique.")
        if not 0.0 <= self.pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be between 0 and 1.")
