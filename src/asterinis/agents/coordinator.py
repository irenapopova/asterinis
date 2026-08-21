from .context import AgentContext
from .result import AgentResult
from .router import AgentRouter
from .task import AgentTask


class AgentCoordinator:
    """
    Routes tasks to registered agents.
    """

    def __init__(
        self,
        router: AgentRouter | None = None,
    ):
        self.router = router or AgentRouter()

    def execute(
        self,
        task: AgentTask,
        *,
        context: AgentContext | None = None,
    ) -> AgentResult:

        agent = self.router.resolve(
            task.instruction
        )

        if agent is None:
            return AgentResult(
                agent="none",
                output=None,
                succeeded=False,
                metadata={
                    "reason": "No matching agent found."
                },
            )

        return agent.run(
            task,
            context=context,
        )