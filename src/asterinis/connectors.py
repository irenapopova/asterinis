from abc import ABC, abstractmethod
from typing import Any


class Connector(ABC):
    """
    Base connector interface.

    Connectors allow Asterinis to communicate with
    external NLP, RAG, agent, or LLM systems.
    """

    name: str = "connector"

    @abstractmethod
    def execute(self, payload: Any) -> Any:
        raise NotImplementedError


class EchoConnector(Connector):
    name = "echo"

    def execute(self, payload: Any) -> Any:
        return {
            "connector": self.name,
            "payload": payload,
        }