from collections.abc import Callable
from typing import Any

from .context import NexusContext
from .exceptions import PipelineError


PipelineStep = Callable[[NexusContext], NexusContext]


class Pipeline:
    def __init__(self, name: str):
        self.name = name
        self._steps: list[PipelineStep] = []

    def add(self, step: PipelineStep) -> "Pipeline":
        self._steps.append(step)
        return self

    def run(self, context: NexusContext) -> NexusContext:
        try:
            for step in self._steps:
                context = step(context)

            return context

        except Exception as exc:
            raise PipelineError(
                f"Pipeline '{self.name}' failed."
            ) from exc

    def __len__(self) -> int:
        return len(self._steps)