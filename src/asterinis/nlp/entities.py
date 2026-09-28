from __future__ import annotations

from collections.abc import Iterable

from .result import Entity


def normalize_entities(
    entities: Iterable[Entity],
    *,
    case_sensitive: bool = False,
) -> list[Entity]:
    """
    Remove duplicate entities while preserving their original order.
    """

    normalized: list[Entity] = []
    seen: set[tuple[str, str]] = set()

    for entity in entities:
        if not isinstance(entity, Entity):
            raise TypeError("Every item must be an Entity.")

        text_key = (
            entity.text
            if case_sensitive
            else entity.text.lower()
        )

        key = (
            text_key,
            entity.label.lower(),
        )

        if key in seen:
            continue

        seen.add(key)
        normalized.append(entity)

    return normalized


def entities_by_label(
    entities: Iterable[Entity],
    label: str,
) -> list[Entity]:
    label = label.strip()

    if not label:
        raise ValueError("label cannot be empty.")

    return [
        entity
        for entity in entities
        if entity.label.lower() == label.lower()
    ]