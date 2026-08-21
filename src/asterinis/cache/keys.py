from __future__ import annotations

import hashlib
import json
from typing import Any


def cache_key(
    namespace: str,
    value: Any,
) -> str:
    namespace = namespace.strip()

    if not namespace:
        raise ValueError(
            "namespace cannot be empty."
        )

    serialized = json.dumps(
        value,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

    return f"{namespace}:{digest}"