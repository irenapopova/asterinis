from typing import Dict

from .connectors import Connector


class ConnectorRegistry:
    def __init__(self):
        self._connectors: Dict[str, Connector] = {}

    def register(self, name: str, connector: Connector) -> None:
        if not name:
            raise ValueError("Connector name cannot be empty.")

        self._connectors[name] = connector

    def get(self, name: str) -> Connector:
        if name not in self._connectors:
            raise KeyError(
                f"Connector '{name}' is not registered."
            )

        return self._connectors[name]

    def remove(self, name: str) -> None:
        self._connectors.pop(name, None)

    def exists(self, name: str) -> bool:
        return name in self._connectors

    def list(self) -> list[str]:
        return list(self._connectors.keys())