from collections.abc import Callable

from .base import Agent


AgentPredicate = Callable[[str], bool]


class AgentRouter:
    def __init__(self):
        self._routes: list[
            tuple[int, AgentPredicate, Agent]
        ] = []

    def register(
        self,
        agent: Agent,
        predicate: AgentPredicate,
        *,
        priority: int = 0,
    ) -> None:
        self._routes.append(
            (
                priority,
                predicate,
                agent,
            )
        )

        self._routes.sort(
            key=lambda item: item[0],
            reverse=True,
        )

    def resolve(
        self,
        text: str,
    ) -> Agent | None:

        for _, predicate, agent in self._routes:
            if predicate(text):
                return agent

        return None