from typing import Any


class AgentMemory:
    def __init__(self):
        self._state: dict[str, Any] = {}

    def set(
        self,
        key: str,
        value: Any,
    ) -> None:
        self._state[key] = value

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        return self._state.get(key, default)

    def delete(
        self,
        key: str,
    ) -> None:
        self._state.pop(key, None)

    def clear(self) -> None:
        self._state.clear()

    def snapshot(self) -> dict[str, Any]:
        return dict(self._state)