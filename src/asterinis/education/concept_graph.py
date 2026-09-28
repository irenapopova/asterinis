"""Prerequisite graph for adaptive lesson recommendations."""

from __future__ import annotations

from collections import defaultdict, deque

from .models import LearnerProfile


class ConceptGraph:
    def __init__(self) -> None:
        self._prerequisites: dict[str, set[str]] = defaultdict(set)

    def add_concept(self, concept: str) -> None:
        concept = concept.strip()
        if not concept:
            raise ValueError("concept cannot be empty.")
        self._prerequisites.setdefault(concept, set())

    def add_prerequisite(self, concept: str, prerequisite: str) -> None:
        self.add_concept(concept)
        self.add_concept(prerequisite)
        if concept == prerequisite or self._depends_on(prerequisite, concept):
            raise ValueError("prerequisite would create a cycle.")
        self._prerequisites[concept].add(prerequisite)

    def ready(self, profile: LearnerProfile, *, threshold: float = 0.7) -> tuple[str, ...]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        return tuple(
            concept
            for concept, prerequisites in self._prerequisites.items()
            if profile.mastered_concepts.get(concept, 0.0) < threshold
            and all(profile.mastered_concepts.get(item, 0.0) >= threshold for item in prerequisites)
        )

    def _depends_on(self, concept: str, target: str) -> bool:
        pending = deque([concept])
        visited: set[str] = set()
        while pending:
            current = pending.popleft()
            if current in visited:
                continue
            visited.add(current)
            if current == target:
                return True
            pending.extend(self._prerequisites.get(current, ()))
        return False
