from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


PlannerStrategy = Callable[
    [str, dict[str, Any]],
    list["QueryPlanStep"],
]


@dataclass(slots=True)
class QueryPlanStep:
    action: str
    input: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action = self.action.strip()
        self.input = self.input.strip()
        self.reason = self.reason.strip()

        if not self.action:
            raise ValueError("Plan action cannot be empty.")

        if not self.input:
            raise ValueError("Plan input cannot be empty.")

        if not self.reason:
            raise ValueError("Plan reason cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "input": self.input,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class QueryPlan:
    query: str
    steps: list[QueryPlanStep]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.steps

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "step_count": self.step_count,
            "steps": [
                step.to_dict()
                for step in self.steps
            ],
            "metadata": dict(self.metadata),
        }


class QueryPlannerAgent:
    """
    Creates an explicit processing plan for a query.

    Planning logic is injected so the agent can use deterministic rules,
    NLP signals, a local model, or an LLM without coupling Asterinis to a
    single planning strategy.
    """

    name = "query-planner"

    def __init__(
        self,
        planner: PlannerStrategy,
        *,
        max_steps: int = 10,
        allowed_actions: set[str] | None = None,
    ) -> None:
        if not callable(planner):
            raise TypeError("planner must be callable.")

        if max_steps < 1:
            raise ValueError(
                "max_steps must be greater than zero."
            )

        self.planner = planner
        self.max_steps = max_steps
        self.allowed_actions = (
            set(allowed_actions)
            if allowed_actions is not None
            else {
                "nlp",
                "retrieve",
                "rewrite",
                "decompose",
                "verify",
                "rerank",
                "generate",
                "clarify",
            }
        )

    def plan(
        self,
        query: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> QueryPlan:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty.")

        planner_context = context or {}

        raw_steps = self.planner(
            query,
            planner_context,
        )

        if not isinstance(raw_steps, list):
            raise TypeError(
                "planner must return a list of QueryPlanStep objects."
            )

        steps = self._validate_steps(raw_steps)

        return QueryPlan(
            query=query,
            steps=steps,
            metadata={
                "max_steps": self.max_steps,
                "allowed_actions": sorted(
                    self.allowed_actions
                ),
            },
        )

    def _validate_steps(
        self,
        steps: list[QueryPlanStep],
    ) -> list[QueryPlanStep]:
        if len(steps) > self.max_steps:
            raise ValueError(
                f"Query plan exceeds the maximum of "
                f"{self.max_steps} steps."
            )

        validated: list[QueryPlanStep] = []

        for step in steps:
            if not isinstance(step, QueryPlanStep):
                raise TypeError(
                    "Every plan item must be a QueryPlanStep."
                )

            if step.action not in self.allowed_actions:
                raise ValueError(
                    f"Action '{step.action}' is not allowed."
                )

            validated.append(step)

        return validated