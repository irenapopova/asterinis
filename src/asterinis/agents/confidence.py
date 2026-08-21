from dataclasses import dataclass


@dataclass(slots=True)
class ConfidenceDecision:
    action: str
    confidence: float
    reason: str


class ConfidenceAgent:
    """
    Converts model confidence scores into explicit actions.
    """

    name = "confidence"

    def __init__(
        self,
        *,
        accept_threshold: float = 0.90,
        verify_threshold: float = 0.65,
    ) -> None:
        if not 0 <= verify_threshold <= accept_threshold <= 1:
            raise ValueError(
                "Thresholds must satisfy "
                "0 <= verify_threshold <= accept_threshold <= 1."
            )

        self.accept_threshold = accept_threshold
        self.verify_threshold = verify_threshold

    def decide(
        self,
        confidence: float,
    ) -> ConfidenceDecision:
        if not 0 <= confidence <= 1:
            raise ValueError(
                "Confidence must be between 0 and 1."
            )

        if confidence >= self.accept_threshold:
            return ConfidenceDecision(
                action="accept",
                confidence=confidence,
                reason="Prediction confidence is high.",
            )

        if confidence >= self.verify_threshold:
            return ConfidenceDecision(
                action="verify",
                confidence=confidence,
                reason=(
                    "Prediction confidence is moderate "
                    "and should be verified."
                ),
            )

        return ConfidenceDecision(
            action="reject",
            confidence=confidence,
            reason=(
                "Prediction confidence is below "
                "the configured verification threshold."
            ),
        )