from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


RoutingRule = Callable[[str, dict[str, Any]], bool]


@dataclass(slots=True)
class NLPRoutingDecision:
    route: str
    reason: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "reason": self.reason,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class NLPRoutingRule:
    route: str
    predicate: RoutingRule
    reason: str
    priority: int = 0


class NLPRouterAgent:
    """
    Selects a processing route using explicit NLP-aware rules.

    Rules can inspect the original text together with metadata produced by an
    NLP provider, such as entities, labels, or confidence values.
    """

    name = "nlp-router"

    def __init__(
        self,
        *,
        default_route: str = "llm",
    ) -> None:
        default_route = default_route.strip()

        if not default_route:
            raise ValueError("default_route cannot be empty.")

        self.default_route = default_route
        self._rules: list[NLPRoutingRule] = []

    def register(
        self,
        route: str,
        predicate: RoutingRule,
        *,
        reason: str,
        priority: int = 0,
    ) -> None:
        route = route.strip()
        reason = reason.strip()

        if not route:
            raise ValueError("route cannot be empty.")

        if not reason:
            raise ValueError("reason cannot be empty.")

        if not callable(predicate):
            raise TypeError("predicate must be callable.")

        self._rules.append(
            NLPRoutingRule(
                route=route,
                predicate=predicate,
                reason=reason,
                priority=priority,
            )
        )

        self._rules.sort(
            key=lambda rule: rule.priority,
            reverse=True,
        )

    def decide(
        self,
        text: str,
        *,
        nlp_metadata: dict[str, Any] | None = None,
    ) -> NLPRoutingDecision:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = text.strip()

        if not text:
            raise ValueError("text cannot be empty.")

        metadata = nlp_metadata or {}

        for rule in self._rules:
            if rule.predicate(text, metadata):
                return NLPRoutingDecision(
                    route=rule.route,
                    reason=rule.reason,
                    confidence=self._extract_confidence(metadata),
                    metadata=metadata,
                )

        return NLPRoutingDecision(
            route=self.default_route,
            reason="No NLP routing rule matched.",
            confidence=self._extract_confidence(metadata),
            metadata=metadata,
        )

    @staticmethod
    def _extract_confidence(
        metadata: dict[str, Any],
    ) -> float | None:
        value = metadata.get("confidence")

        if value is None:
            return None

        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None

        if not 0.0 <= confidence <= 1.0:
            return None

        return confidence