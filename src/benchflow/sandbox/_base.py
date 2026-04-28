"""BaseSandbox ABC + lifecycle/result types.

CONTRACT SURFACE — semver-stable. This is the plug axis for sandbox
adapters (docker.py, daytona.py, and any future backend). Changes here
cascade to every adapter and to every caller that accepts a sandbox.
Kept in ``sandbox/`` rather than ``contracts/`` by the hybrid-layout rule
(interface lives next to its implementations when there is a plug axis).
Prefer extending adapter implementations unless the interface itself
must change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal


class SandboxType(StrEnum):
    DOCKER = "docker"
    DAYTONA = "daytona"


class SandboxState(StrEnum):
    UNCREATED = "uncreated"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"


class SandboxClosed(RuntimeError):
    """Raised when an operation is attempted on a closed sandbox."""


ExecOutcome = Literal["success", "error", "timeout"]


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False

    @property
    def outcome(self) -> ExecOutcome:
        if self.timed_out:
            return "timeout"
        return "success" if self.return_code == 0 else "error"


class BaseSandbox(ABC):
    """Minimal lifecycle + I/O contract for sandboxes."""

    sandbox_type: SandboxType

    def __init__(self) -> None:
        self._state: SandboxState = SandboxState.UNCREATED

    @property
    def state(self) -> SandboxState:
        return self._state

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult: ...

    @abstractmethod
    async def upload_file(self, source_path: Path | str, target_path: str) -> None: ...

    @abstractmethod
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None: ...

    @abstractmethod
    async def download_file(
        self, source_path: str, target_path: Path | str
    ) -> None: ...

    @abstractmethod
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None: ...

    async def is_dir(self, path: str, *, user: str | int | None = None) -> bool:
        import shlex

        result = await self.exec(
            f"test -d {shlex.quote(path)}", timeout_sec=10, user=user
        )
        return result.return_code == 0

    async def is_file(self, path: str, *, user: str | int | None = None) -> bool:
        import shlex

        result = await self.exec(
            f"test -f {shlex.quote(path)}", timeout_sec=10, user=user
        )
        return result.return_code == 0
