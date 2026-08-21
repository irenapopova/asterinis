import re
from typing import Any


_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
}


def redact_value(value: str) -> str:
    if not value:
        return value

    if len(value) <= 8:
        return "***"

    return f"{value[:3]}***{value[-3:]}"


def redact_mapping(
    data: dict[str, Any],
) -> dict[str, Any]:
    redacted: dict[str, Any] = {}

    for key, value in data.items():
        normalized = key.lower().strip()

        if normalized in _SECRET_KEYS:
            redacted[key] = "***"
            continue

        if isinstance(value, dict):
            redacted[key] = redact_mapping(value)
        else:
            redacted[key] = value

    return redacted


def redact_text(text: str) -> str:
    patterns = [
        r"(?i)(api[_-]?key\s*[:=]\s*)(\S+)",
        r"(?i)(token\s*[:=]\s*)(\S+)",
        r"(?i)(password\s*[:=]\s*)(\S+)",
        r"(?i)(authorization\s*[:=]\s*)(\S+)",
    ]

    result = text

    for pattern in patterns:
        result = re.sub(
            pattern,
            r"\1***",
            result,
        )

    return result