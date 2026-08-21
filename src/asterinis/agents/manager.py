from .base import Agent


class AgentManager:
    """
    Registry and lifecycle manager for Asterinis agents.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(
        self,
        agent: Agent,
        *,
        replace: bool = False,
    ) -> None:
        name = agent.name.strip()

        if not name:
            raise ValueError("Agent name cannot be empty.")

        if name in self._agents and not replace:
            raise ValueError(
                f"Agent '{name}' is already registered."
            )

        self._agents[name] = agent

    def get(self, name: str) -> Agent:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise KeyError(
                f"Agent '{name}' is not registered."
            ) from exc

    def remove(self, name: str) -> None:
        self._agents.pop(name, None)

    def contains(self, name: str) -> bool:
        return name in self._agents

    def names(self) -> tuple[str, ...]:
        return tuple(self._agents.keys())

    def clear(self) -> None:
        self._agents.clear()

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents