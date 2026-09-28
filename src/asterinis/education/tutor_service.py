"""Course-grounded tutoring orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from asterinis.rag.citations import CitationBuilder
from asterinis.rag.context_builder import ContextBuilder

from .course_context import CourseContext
from .models import LearnerProfile
from .recommendations import RecommendationEngine
from .tutoring import Tutor, TutorResponse


AnswerProvider = Callable[[str, str], str]


@dataclass(frozen=True, slots=True)
class TutorAnswer:
    answer: str
    teaching: TutorResponse
    citations: tuple[dict[str, object], ...]
    next_lesson_id: str | None = None


class TutorService:
    """Combine pedagogy, course retrieval, citations, and answer generation."""

    def __init__(
        self,
        course_context: CourseContext,
        answer_provider: AnswerProvider,
        *,
        tutor: Tutor | None = None,
        recommendations: RecommendationEngine | None = None,
    ) -> None:
        if not callable(answer_provider):
            raise TypeError("answer_provider must be callable.")
        self.course_context = course_context
        self.answer_provider = answer_provider
        self.tutor = tutor or Tutor()
        self.recommendations = recommendations or RecommendationEngine()
        self.context_builder = ContextBuilder()
        self.citation_builder = CitationBuilder()

    def answer(
        self,
        profile: LearnerProfile,
        *,
        concept: str,
        question: str,
        attempts: int = 0,
        student_attempt: str | None = None,
    ) -> TutorAnswer:
        teaching = self.tutor.respond(
            profile,
            concept=concept,
            question=question,
            attempts=attempts,
            student_attempt=student_attempt,
        )
        results = self.course_context.retrieve(question)
        context = self.context_builder.build(results)
        prompt = teaching.content_prompt
        if context:
            prompt += f"\n\nUse only this course material when relevant:\n{context}"
        answer = self.answer_provider(prompt, context)
        citations = self.citation_builder.build(results).citations
        next_lesson = self.recommendations.next_lesson(profile, self.course_context.course)
        return TutorAnswer(
            answer=answer,
            teaching=teaching,
            citations=tuple(citation.to_dict() for citation in citations),
            next_lesson_id=next_lesson.lesson_id if next_lesson else None,
        )
