"""Small application service for integrating Asterinis with a learning platform."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from .decision import Decision, DecisionEngine, DecisionPolicy, ProviderOption
from .learning import SQLiteStrategyStore, StrategyRecord


@dataclass(frozen=True, slots=True)
class LearningResponse:
    """Answer returned to a learner, including routing information for UI use."""

    response_id: str
    answer: object
    decision: Decision


class LearningPlatform:
    """Application-facing service for adaptive educational responses.

    A provider is any callable that accepts a learner's question and returns an
    answer. The service persists execution history and learner feedback in
    SQLite, allowing routing decisions to improve across application restarts.
    """

    def __init__(self, database: str = "asterinis-learning.db") -> None:
        self.store = SQLiteStrategyStore(database)
        self.engine = DecisionEngine(learning_store=self.store)
        self._responses: dict[str, tuple[str, str, str]] = {}

    def register_provider(self, option: ProviderOption, *, replace: bool = False) -> None:
        self.engine.register(option, replace=replace)

    def ask(
        self,
        question: str,
        capability: str = "generation",
        *,
        policy: DecisionPolicy | None = None,
    ) -> LearningResponse:
        if not question.strip():
            raise ValueError("question cannot be empty.")

        decision, answer = self.engine.execute(
            question,
            capability,
            policy=policy,
        )
        response_id = uuid4().hex
        self._responses[response_id] = (
            decision.provider,
            capability.strip().lower(),
            question,
        )
        return LearningResponse(response_id, answer, decision)

    def record_feedback(
        self,
        response_id: str,
        *,
        helpful: bool,
        quality_score: float | None = None,
    ) -> None:
        """Store learner feedback used by future provider selection."""

        try:
            provider, capability, question = self._responses[response_id]
        except KeyError as exc:
            raise ValueError("unknown response_id; feedback must follow ask().") from exc

        self.store.add(
            StrategyRecord(
                strategy=provider,
                query_type=capability,
                success=helpful,
                quality_score=(
                    quality_score
                    if quality_score is not None
                    else (1.0 if helpful else 0.0)
                ),
                metadata={"response_id": response_id, "question": question},
            )
        )

    def close(self) -> None:
        self.store.close()


__all__ = ["LearningPlatform", "LearningResponse"]
