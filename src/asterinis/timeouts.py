from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from .exceptions import ProviderTimeoutError


T = TypeVar("T")


# Backward-compatible name. ProviderTimeoutError is the canonical exception.
AsterinisTimeoutError = ProviderTimeoutError


async def with_timeout(
    operation: Awaitable[T],
    *,
    timeout_seconds: float,
    operation_name: str = "operation",
) -> T:
    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds must be greater than zero."
        )

    try:
        return await asyncio.wait_for(
            operation,
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise ProviderTimeoutError(
            f"{operation_name} exceeded the "
            f"{timeout_seconds:g} second timeout."
        ) from exc
