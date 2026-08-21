from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


VerificationHandler = Callable[[Any, dict[str, Any]], bool]


@dataclass(slots=True)
class VerificationResult:
    verified: bool
    verifier: str
    reason: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "verifier": self.verifier,
            "reason": self.reason,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


class VerificationAgent:
    """
    Verifies uncertain results using an application-supplied strategy.

    The verification strategy is injected so the agent can work with NLP
    predictions, retrieval evidence, model outputs, or custom domain checks.
    """

    name = "verification"

    def __init__(
        self,
        verifier: VerificationHandler,
        *,
        verifier_name: str = "verifier",
    ) -> None:
        if not callable(verifier):
            raise TypeError("verifier must be callable.")

        verifier_name = verifier_name.strip()

        if not verifier_name:
            raise ValueError("verifier_name cannot be empty.")

        self.verifier = verifier
        self.verifier_name = verifier_name

    def verify(
        self,
        value: Any,
        *,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VerificationResult:
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        context = metadata or {}

        try:
            verified = bool(
                self.verifier(
                    value,
                    context,
                )
            )

        except Exception as exc:
            return VerificationResult(
                verified=False,
                verifier=self.verifier_name,
                reason="Verification failed with an exception.",
                confidence=confidence,
                metadata={
                    **context,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

        reason = (
            "Verification succeeded."
            if verified
            else "Verification did not confirm the result."
        )

        return VerificationResult(
            verified=verified,
            verifier=self.verifier_name,
            reason=reason,
            confidence=confidence,
            metadata=context,
        )