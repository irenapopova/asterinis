from __future__ import annotations

from typing import Any

from asterinis.exceptions import RoutingError

from .router import NLPTaskRouter


class NLPFallbackRouter:
    """Retries an NLP request with the next eligible provider after failure."""

    def __init__(
        self,
        router: NLPTaskRouter,
        *,
        max_attempts: int | None = None,
    ) -> None:
        if not isinstance(router, NLPTaskRouter):
            raise TypeError("router must be an NLPTaskRouter.")
        if max_attempts is not None and max_attempts < 1:
            raise ValueError("max_attempts must be greater than zero.")

        self.router = router
        self.max_attempts = max_attempts

    def analyze(
        self,
        text: str,
        *,
        task: str = "classification",
        language: str | None = None,
        max_cost: float | None = None,
        max_latency_ms: float | None = None,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if not text.strip():
            raise ValueError("text cannot be empty.")

        excluded: set[str] = set()
        failures: list[str] = []
        attempts = 0

        while (
            self.max_attempts is None
            or attempts < self.max_attempts
        ):
            try:
                selection = self.router.select(
                    task,
                    language=language,
                    max_cost=max_cost,
                    max_latency_ms=max_latency_ms,
                    exclude=excluded,
                )
            except RoutingError:
                break

            attempts += 1
            excluded.add(selection.provider)

            try:
                result = self.router.invoke_selection(
                    selection,
                    text,
                    **kwargs,
                )
            except Exception as exc:
                failures.append(
                    f"{selection.provider}: {type(exc).__name__}"
                )
                continue

            if hasattr(result, "metadata"):
                result.metadata["fallback_attempts"] = attempts
                result.metadata["fallback_failures"] = list(failures)
            return result

        detail = ", ".join(failures) or "no eligible providers"
        raise RoutingError(
            f"All NLP providers failed for '{task}': {detail}"
        )
