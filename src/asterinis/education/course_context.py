"""Connect course lessons to Asterinis retrieval and citations."""

from __future__ import annotations

from asterinis.rag import BM25Retriever, Document, Retriever
from asterinis.rag.documents import RetrievalResult

from .models import Course


class CourseContext:
    def __init__(self, course: Course, retriever: Retriever) -> None:
        self.course = course
        self.retriever = retriever

    @classmethod
    def from_course(cls, course: Course) -> "CourseContext":
        documents = [
            Document(
                id=lesson.lesson_id,
                text=lesson.content,
                metadata={
                    "course_id": course.course_id,
                    "lesson_id": lesson.lesson_id,
                    "title": lesson.title,
                    "concepts": lesson.concepts,
                },
            )
            for lesson in course.lessons
        ]
        return cls(course, BM25Retriever(documents))

    def retrieve(self, question: str, *, limit: int = 4) -> list[RetrievalResult]:
        return self.retriever.retrieve(question, limit=limit)
