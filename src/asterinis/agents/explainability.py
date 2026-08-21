from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Explanation:
    summary: str
    route: str | None = None
    provider: str | None = None
    agent: str | None = None
    confidence: float | None = None
    evidence_count: int | None = None
    decision: str | None = None
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "route": self.route,
            "provider": self.provider,
            "agent": self.agent,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "decision": self.decision,
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }


class ExplainabilityAgent:
    """
    Builds a structured explanation from explicit Asterinis decisions.

    It does not attempt to reconstruct hidden model reasoning. Instead, it
    summarizes observable information such as routes, providers, confidence,
    retrieval evidence, and agent decisions.
    """

    name = "explainability"

    def explain(
        self,
        *,
        route: str | None = None,
        provider: str | None = None,
        agent: str | None = None,
        confidence: float | None = None,
        evidence_count: int | None = None,
        decision: str | None = None,
        reasons: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Explanation:
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        if evidence_count is not None and evidence_count < 0:
            raise ValueError("evidence_count cannot be negative.")

        reason_list = [
            reason.strip()
            for reason in (reasons or [])
            if isinstance(reason, str) and reason.strip()
        ]

        summary = self._build_summary(
            route=route,
            provider=provider,
            agent=agent,
            confidence=confidence,
            evidence_count=evidence_count,
            decision=decision,
        )

        return Explanation(
            summary=summary,
            route=route,
            provider=provider,
            agent=agent,
            confidence=confidence,
            evidence_count=evidence_count,
            decision=decision,
            reasons=reason_list,
            metadata=metadata or {},
        )

    @staticmethod
    def _build_summary(
        *,
        route: str | None,
        provider: str | None,
        agent: str | None,
        confidence: float | None,
        evidence_count: int | None,
        decision: str | None,
    ) -> str:
        parts: list[str] = []

        if route:
            parts.append(f"route={route}")

        if provider:
            parts.append(f"provider={provider}")

        if agent:
            parts.append(f"agent={agent}")

        if confidence is not None:
            parts.append(f"confidence={confidence:.3f}")

        if evidence_count is not None:
            parts.append(f"evidence={evidence_count}")

        if decision:
            parts.append(f"decision={decision}")

        if not parts:
            return "No explanation data was provided."

        return "Asterinis decision: " + ", ".join(parts) + "."