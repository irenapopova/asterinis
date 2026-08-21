from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    name: str = "tool"

    @abstractmethod
    def run(
        self,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError