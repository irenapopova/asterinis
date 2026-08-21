from collections.abc import Callable
from dataclasses import dataclass


RoutePredicate = Callable[[str], bool]


@dataclass(slots=True)
class Route:
    """
    Represents a routing rule inside Asterinis.
    """

    name: str
    predicate: RoutePredicate
    priority: int = 0


class Router:
    """
    Lightweight and extensible request router.

    Routes are evaluated by priority, with higher-priority
    routes evaluated first.
    """

    def __init__(self, default_route: str = "llm"):
        self.default_route = default_route
        self._routes: list[Route] = []

    def register(
        self,
        name: str,
        predicate: RoutePredicate,
        *,
        priority: int = 0,
    ) -> None:
        """
        Register a new route.
        """

        if not isinstance(name, str):
            raise TypeError("Route name must be a string.")

        name = name.strip()

        if not name:
            raise ValueError("Route name cannot be empty.")

        if not callable(predicate):
            raise TypeError("Route predicate must be callable.")

        self._routes.append(
            Route(
                name=name,
                predicate=predicate,
                priority=priority,
            )
        )

        self._routes.sort(
            key=lambda route: route.priority,
            reverse=True,
        )

    def resolve(self, text: str) -> str:
        """
        Determine which route should handle the given text.
        """

        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        for route in self._routes:
            if route.predicate(text):
                return route.name

        return self.default_route

    @property
    def routes(self) -> tuple[str, ...]:
        """
        Return all registered route names.
        """
        return tuple(route.name for route in self._routes)