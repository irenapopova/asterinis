from typing import Any


def validate_input(
    text: str,
    *,
    max_length: int = 20_000,
) -> str:
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Input cannot be empty.")

    if len(text) > max_length:
        raise ValueError(
            f"Input exceeds maximum length of {max_length} characters."
        )

    return text


def validate_metadata(
    metadata: dict[str, Any] | None,
    *,
    max_items: int = 100,
) -> dict[str, Any]:
    if metadata is None:
        return {}

    if not isinstance(metadata, dict):
        raise TypeError("Metadata must be a dictionary.")

    if len(metadata) > max_items:
        raise ValueError(
            f"Metadata cannot contain more than {max_items} items."
        )

    return metadata