"""Deterministic assessment grading with learner-friendly feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .questions import Assessment, Question, QuestionType


@dataclass(frozen=True, slots=True)
class QuestionGrade:
    question_id: str
    correct: bool
    score: float
    points_earned: float
    points_possible: float
    feedback: str


@dataclass(frozen=True, slots=True)
class AssessmentGrade:
    assessment_id: str
    score: float
    points_earned: float
    points_possible: float
    passed: bool
    questions: tuple[QuestionGrade, ...]


def _normalize(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def _accepted_answers(question: Question) -> tuple[str, ...]:
    if isinstance(question.answer, tuple):
        return tuple(_normalize(answer) for answer in question.answer)
    return (_normalize(question.answer),)


def grade_question(question: Question, answer: object) -> QuestionGrade:
    normalized = _normalize(answer)
    correct = normalized in _accepted_answers(question)
    feedback = "Correct." if correct else (question.explanation or "Review this concept and try again.")
    return QuestionGrade(
        question_id=question.question_id,
        correct=correct,
        score=1.0 if correct else 0.0,
        points_earned=question.points if correct else 0.0,
        points_possible=question.points,
        feedback=feedback,
    )


class AssessmentEngine:
    def grade(self, assessment: Assessment, answers: Mapping[str, object]) -> AssessmentGrade:
        grades = tuple(
            grade_question(question, answers.get(question.question_id, ""))
            for question in assessment.questions
        )
        points_possible = sum(item.points_possible for item in grades)
        points_earned = sum(item.points_earned for item in grades)
        score = points_earned / points_possible if points_possible else 0.0
        return AssessmentGrade(
            assessment_id=assessment.assessment_id,
            score=score,
            points_earned=points_earned,
            points_possible=points_possible,
            passed=score >= assessment.pass_threshold,
            questions=grades,
        )
