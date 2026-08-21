from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


FallbackHandler = Callable[[str, dict[str, Any]], Any]


@dataclass(slots=True)
class FallbackResult:
    reason: str
    handler: str
    output: Any
    succeeded: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "handler": self.handler,
            "output": self.output,
            "succeeded": self.succeeded,
            "metadata": dict(self.metadata),
        }


class FallbackAgent:
    """
    Provides a controlled fallback when the preferred processing path fails.

    A fallback can be used for provider failures, weak retrieval results,
    unsupported routes, or other recoverable conditions.
    """

    name = "fallback"

    def __init__(
        self,
        handler: FallbackHandler,
        *,
        handler_name: str = "fallback-handler",
    ) -> None:
        if not callable(handler):
            raise TypeError("handler must be callable.")

        handler_name = handler_name.strip()

        if not handler_name:
            raise ValueError("handler_name cannot be empty.")

        self.handler = handler
        self.handler_name = handler_name

    def run(
        self,
        reason: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> FallbackResult:
        if not isinstance(reason, str):
            raise TypeError("reason must be a string.")

        reason = reason.strip()

        if not reason:
            raise ValueError("reason cannot be empty.")

        context = metadata or {}

        try:
            output = self.handler(reason, context)

            return FallbackResult(
                reason=reason,
                handler=self.handler_name,
                output=output,
                succeeded=True,
                metadata=context,
            )

        except Exception as exc:
            return FallbackResult(
                reason=reason,
                handler=self.handler_name,
                output=None,
                succeeded=False,
                metadata={
                    **context,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )