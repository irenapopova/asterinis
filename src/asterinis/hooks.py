from collections import defaultdict
from collections.abc import Callable
from typing import Any


Hook = Callable[..., Any]


class HookManager:
    def __init__(self):
        self._hooks: dict[str, list[Hook]] = defaultdict(list)

    def register(self, event: str, callback: Hook) -> None:
        self._hooks[event].append(callback)

    def emit(self, event: str, **kwargs: Any) -> None:
        for callback in self._hooks.get(event, []):
            callback(**kwargs)