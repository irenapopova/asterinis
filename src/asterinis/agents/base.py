from abc import ABC, abstractmethod

from .context import AgentContext
from .result import AgentResult
from .task import AgentTask


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def run(
        self,
        task: AgentTask,
        *,
        context: AgentContext | None = None,
    ) -> AgentResult:
        raise NotImplementedError