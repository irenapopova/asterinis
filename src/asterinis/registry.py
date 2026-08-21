from .providers import Provider


class ProviderRegistry:
    """
    Registry for providers used by Asterinis.

    Providers can represent NLP libraries, LLM services,
    retrieval systems, agents, or other processing backends.
    """

    def __init__(self):
        self._providers: dict[str, Provider] = {}

    def register(
        self,
        name: str,
        provider: Provider,
        *,
        replace: bool = False,
    ) -> None:
        """
        Register a provider under a unique name.
        """

        if not isinstance(name, str):
            raise TypeError("Provider name must be a string.")

        name = name.strip()

        if not name:
            raise ValueError("Provider name cannot be empty.")

        if name in self._providers and not replace:
            raise ValueError(
                f"Provider '{name}' is already registered."
            )

        self._providers[name] = provider

    def get(self, name: str) -> Provider:
        """
        Return a registered provider.
        """

        try:
            return self._providers[name]
        except KeyError as exc:
            raise KeyError(
                f"Provider '{name}' is not registered."
            ) from exc

    def remove(self, name: str) -> None:
        """
        Remove a provider if it exists.
        """
        self._providers.pop(name, None)

    def contains(self, name: str) -> bool:
        """
        Check whether a provider is registered.
        """
        return name in self._providers

    def names(self) -> tuple[str, ...]:
        """
        Return all registered provider names.
        """
        return tuple(self._providers.keys())

    def clear(self) -> None:
        """
        Remove all registered providers.
        """
        self._providers.clear()

    def __len__(self) -> int:
        return len(self._providers)

    def __contains__(self, name: str) -> bool:
        return name in self._providers