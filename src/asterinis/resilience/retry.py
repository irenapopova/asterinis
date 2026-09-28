from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar


T = TypeVar("T")


def retry_call(
    operation: Callable[[], T],
    *,
    attempts: int = 1,
    delay_seconds: float = 0.0,
    backoff: float = 2.0,
) -> T:
    if attempts < 1:
        raise ValueError("attempts must be greater than zero.")
    if delay_seconds < 0 or backoff < 1:
        raise ValueError("delay_seconds must be non-negative and backoff >= 1.")

    for attempt in range(attempts):
        try:
            return operation()
        except Exception:
            if attempt == attempts - 1:
                raise
            delay = delay_seconds * (backoff ** attempt)
            if delay:
                time.sleep(delay)

    raise RuntimeError("retry operation did not complete.")


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int = 1,
    delay_seconds: float = 0.0,
    backoff: float = 2.0,
) -> T:
    if attempts < 1:
        raise ValueError("attempts must be greater than zero.")
    if delay_seconds < 0 or backoff < 1:
        raise ValueError("delay_seconds must be non-negative and backoff >= 1.")

    for attempt in range(attempts):
        try:
            return await operation()
        except Exception:
            if attempt == attempts - 1:
                raise
            delay = delay_seconds * (backoff ** attempt)
            if delay:
                await asyncio.sleep(delay)

    raise RuntimeError("retry operation did not complete.")
