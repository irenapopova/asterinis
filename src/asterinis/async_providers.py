from __future__ import annotations

import asyncio
from typing import Any

from .providers import Provider
from .timeouts import with_timeout


async def invoke_provider(
    provider: Provider,
    text: str,
    *,
    timeout_seconds: float | None = None,
    **kwargs: Any,
) -> Any:
    """
    Invoke a synchronous Asterinis provider without blocking the event loop.

    Provider execution is moved to a worker thread. An optional timeout can
    be applied by the caller.
    """

    if not isinstance(provider, Provider):
        raise TypeError(
            "provider must implement the Asterinis Provider interface."
        )

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("text cannot be empty.")

    operation = asyncio.to_thread(
        provider.invoke,
        text,
        **kwargs,
    )

    if timeout_seconds is None:
        return await operation

    return await with_timeout(
        operation,
        timeout_seconds=timeout_seconds,
        operation_name=f"provider '{provider.name}'",
    )