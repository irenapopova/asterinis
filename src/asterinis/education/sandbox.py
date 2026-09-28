"""Safe execution boundary for coding exercises."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import subprocess
import time


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    timeout_seconds: float = 3.0
    memory: str = "128m"
    cpus: float = 0.5
    pids: int = 64
    output_bytes: int = 16_384

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.cpus <= 0 or self.pids < 1 or self.output_bytes < 1:
            raise ValueError("sandbox limits must be positive.")
        if not self.memory.strip():
            raise ValueError("memory limit cannot be empty.")


@dataclass(frozen=True, slots=True)
class SandboxResult:
    passed: bool
    stdout: str
    stderr: str
    exit_code: int | None
    execution_time_ms: float
    timed_out: bool = False


class CodeSandbox(ABC):
    @abstractmethod
    def run(self, source: str, *, stdin: str = "") -> SandboxResult:
        raise NotImplementedError


class DockerPythonSandbox(CodeSandbox):
    """Run Python in a container with no network and restricted resources."""

    def __init__(self, *, image: str = "python:3.13-slim", limits: SandboxLimits | None = None) -> None:
        if not image.strip() or any(character in image for character in " \t\n"):
            raise ValueError("image must be a non-empty Docker image name.")
        self.image = image
        self.limits = limits or SandboxLimits()

    def run(self, source: str, *, stdin: str = "") -> SandboxResult:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string.")
        command = [
            "docker", "run", "--rm", "-i", "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=32m", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", str(self.limits.pids),
            "--memory", self.limits.memory, "--cpus", str(self.limits.cpus),
            self.image, "python", "-I", "-",
        ]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                input=source.encode(),
                capture_output=True,
                timeout=self.limits.timeout_seconds,
                check=False,
            )
            return SandboxResult(
                passed=completed.returncode == 0,
                stdout=completed.stdout[: self.limits.output_bytes].decode(errors="replace"),
                stderr=completed.stderr[: self.limits.output_bytes].decode(errors="replace"),
                exit_code=completed.returncode,
                execution_time_ms=(time.perf_counter() - started) * 1000,
            )
        except subprocess.TimeoutExpired as error:
            output = error.stdout or b""
            return SandboxResult(
                passed=False,
                stdout=output[: self.limits.output_bytes].decode(errors="replace") if isinstance(output, bytes) else str(output),
                stderr="Execution timed out.",
                exit_code=None,
                execution_time_ms=(time.perf_counter() - started) * 1000,
                timed_out=True,
            )
        except FileNotFoundError as error:
            raise RuntimeError("Docker is required for DockerPythonSandbox.") from error


__all__ = ["CodeSandbox", "DockerPythonSandbox", "SandboxLimits", "SandboxResult"]
