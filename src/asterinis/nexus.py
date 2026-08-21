from collections.abc import Callable


RouteRule = Callable[[str], bool]


class Router:
    def __init__(self, default_route: str = "llm"):
        self.default_route = default_route
        self._rules: list[tuple[str, RouteRule]] = []

        self._register_default_rules()

    def _register_default_rules(self) -> None:
        self.add_route(
            "rag",
            lambda text: any(
                word in text.lower()
                for word in (
                    "document",
                    "source",
                    "retrieve",
                    "search",
                )
            ),
        )

        self.add_route(
            "nlp",
            lambda text: any(
                word in text.lower()
                for word in (
                    "entity",
                    "ner",
                    "language",
                    "nlp",
                )
            ),
        )

        self.add_route(
            "agent",
            lambda text: any(
                word in text.lower()
                for word in (
                    "agent",
                    "tool",
                    "workflow",
                )
            ),
        )

    def add_route(
        self,
        name: str,
        rule: RouteRule,
        *,
        first: bool = False,
    ) -> None:
        route = (name, rule)

        if first:
            self._rules.insert(0, route)
        else:
            self._rules.append(route)

    def route(self, text: str) -> str:
        for name, rule in self._rules:
            if rule(text):
                return name

        return self.default_route

    @property
    def routes(self) -> list[str]:
        return [name for name, _ in self._rules]