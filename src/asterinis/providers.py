from abc import ABC, abstractmethod
from typing import Any


class Provider(ABC):
    """
    Base interface for Asterinis providers.

    A provider represents an external model, library,
    service, or processing backend.
    """

    name: str = "provider"

    @abstractmethod
    def invoke(self, text: str, **kwargs: Any) -> Any:
        """
        Process input text and return a provider-specific result.
        """
        raise NotImplementedError