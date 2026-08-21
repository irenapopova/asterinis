from dataclasses import dataclass


@dataclass
class AsterinisConfig:
    default_route: str = "llm"
    enable_rag: bool = True
    enable_nlp: bool = True
    enable_agents: bool = True
    debug: bool = False