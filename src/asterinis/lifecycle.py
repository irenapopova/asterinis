from enum import Enum


class LifecycleState(str, Enum):
    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"