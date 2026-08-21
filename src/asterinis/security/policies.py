from dataclasses import dataclass


@dataclass(slots=True)
class SecurityPolicy:
    max_agent_steps: int = 10
    max_retrieval_results: int = 20
    provider_timeout_seconds: float = 30.0
    max_input_length: int = 20_000
    allow_external_tools: bool = False

    def __post_init__(self) -> None:
        if self.max_agent_steps < 1:
            raise ValueError("max_agent_steps must be greater than zero.")

        if self.max_retrieval_results < 1:
            raise ValueError("max_retrieval_results must be greater than zero.")

        if self.provider_timeout_seconds <= 0:
            raise ValueError(
                "provider_timeout_seconds must be greater than zero."
            )

        if self.max_input_length < 1:
            raise ValueError("max_input_length must be greater than zero.")