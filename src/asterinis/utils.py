from typing import Any


def compact_dict(data: dict[str, Any]) -> dict[str, Any]:
    """
    Remove keys whose values are None.
    """
    return {
        key: value
        for key, value in data.items()
        if value is not None
    }