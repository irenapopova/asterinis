"""Pair-programming session primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class CollaborationMessage:
    learner_id: str
    text: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class PairProgrammingSession:
    session_id: str
    exercise_id: str
    driver_id: str
    navigator_id: str
    messages: list[CollaborationMessage] = field(default_factory=list)

    def add_message(self, learner_id: str, text: str) -> None:
        if learner_id not in {self.driver_id, self.navigator_id}:
            raise ValueError("only session participants may send messages.")
        if not text.strip():
            raise ValueError("message cannot be empty.")
        self.messages.append(CollaborationMessage(learner_id, text.strip()))

    def switch_roles(self) -> None:
        self.driver_id, self.navigator_id = self.navigator_id, self.driver_id
