from __future__ import annotations

from enum import Enum
from time import monotonic
from typing import Callable, TypeVar


T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a provider circuit is temporarily open."""


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        recovery_seconds: float = 30.0,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be greater than zero.")
        if recovery_seconds <= 0:
            raise ValueError("recovery_seconds must be greater than zero.")
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.failures = 0
        self.state = CircuitState.CLOSED
        self._opened_at: float | None = None

    def call(self, operation: Callable[[], T]) -> T:
        if self.state is CircuitState.OPEN:
            if self._opened_at is None or monotonic() - self._opened_at < self.recovery_seconds:
                raise CircuitOpenError("provider circuit is open.")
            self.state = CircuitState.HALF_OPEN

        try:
            result = operation()
        except Exception:
            self._record_failure()
            raise

        self.failures = 0
        self.state = CircuitState.CLOSED
        self._opened_at = None
        return result

    def _record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self._opened_at = monotonic()
