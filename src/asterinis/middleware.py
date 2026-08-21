from abc import ABC, abstractmethod
from typing import Any


class Middleware(ABC):

    def before(self, context: Any) -> Any:
        return context

    def after(self, result: Any) -> Any:
        return result


class TimingMiddleware(Middleware):

    def before(self, context):
        import time

        context.set("_start_time", time.perf_counter())
        return context

    def after(self, result):
        return result