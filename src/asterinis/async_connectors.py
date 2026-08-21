from abc import ABC, abstractmethod
from typing import Any


class AsyncConnector(ABC):
    name: str = "async_connector"

    @abstractmethod
    async def execute(self, payload: Any) -> Any:
        raise NotImplementedError