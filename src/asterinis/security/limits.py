from dataclasses import dataclass


@dataclass(slots=True)
class ExecutionLimits:
    max_agent_steps: int = 10
    max_tool_calls: int = 20
    max_retries: int = 3

    def validate(self) -> None:
        if self.max_agent_steps < 1:
            raise ValueError(
                "max_agent_steps must be greater than zero."
            )

        if self.max_tool_calls < 0:
            raise ValueError(
                "max_tool_calls cannot be negative."
            )

        if self.max_retries < 0:
            raise ValueError(
                "max_retries cannot be negative."
            )


class ExecutionCounter:
    def __init__(
        self,
        limits: ExecutionLimits,
    ) -> None:
        limits.validate()

        self.limits = limits
        self.agent_steps = 0
        self.tool_calls = 0
        self.retries = 0

    def record_agent_step(self) -> None:
        self.agent_steps += 1

        if self.agent_steps > self.limits.max_agent_steps:
            raise RuntimeError(
                "Maximum agent step limit exceeded."
            )

    def record_tool_call(self) -> None:
        self.tool_calls += 1

        if self.tool_calls > self.limits.max_tool_calls:
            raise RuntimeError(
                "Maximum tool call limit exceeded."
            )

    def record_retry(self) -> None:
        self.retries += 1

        if self.retries > self.limits.max_retries:
            raise RuntimeError(
                "Maximum retry limit exceeded."
            )