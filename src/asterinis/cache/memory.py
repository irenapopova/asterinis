from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Any


@dataclass(slots=True)
class CacheEntry:
    value: Any
    expires_at: float | None = None

    def expired(self) -> bool:
        return (
            self.expires_at is not None
            and monotonic() >= self.expires_at
        )


class MemoryCache:
    """
    Small thread-safe in-memory cache for local applications and tests.

    It is intentionally not a replacement for Redis or another distributed
    cache. Applications can provide a different cache backend later.
    """

    def __init__(self) -> None:
        self._entries: dict[
            str,
            CacheEntry,
        ] = {}
        self._lock = Lock()

    def set(
        self,
        key: str,
        value: Any,
        *,
        ttl_seconds: float | None = None,
    ) -> None:
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError(
                "ttl_seconds must be greater than zero."
            )

        expires_at = (
            monotonic() + ttl_seconds
            if ttl_seconds is not None
            else None
        )

        with self._lock:
            self._entries[key] = CacheEntry(
                value=value,
                expires_at=expires_at,
            )

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                return default

            if entry.expired():
                self._entries.pop(
                    key,
                    None,
                )
                return default

            return entry.value

    def delete(
        self,
        key: str,
    ) -> None:
        with self._lock:
            self._entries.pop(
                key,
                None,
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __contains__(
        self,
        key: str,
    ) -> bool:
        sentinel = object()
        return self.get(
            key,
            sentinel,
        ) is not sentinel