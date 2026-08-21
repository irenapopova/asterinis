from typing import Any

from asterinis.connectors import Connector


class OpenAIConnector(Connector):
    name = "openai"

    def __init__(self, client: Any = None):
        self.client = client

    def execute(self, payload: Any) -> Any:
        text = payload["text"]

        return {
            "provider": "openai",
            "status": "connector-ready",
            "text": text,
            "message": (
                "OpenAI connector initialized. "
                "Provider execution can be configured by the application."
            ),
        }