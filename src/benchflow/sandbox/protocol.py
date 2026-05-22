from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ExecResult:
    return_code: int
    stdout: str
    stderr: str


@runtime_checkable
class Sandbox(Protocol):
    """Run-only: isolated execution environment.

    All roles in a scene share one Sandbox instance, so inter-agent
    communication over localhost is available by default.  If an agent
    needs to expose additional ports (e.g. for an agent-as-tool HTTP
    endpoint), configure ``expose_ports`` on the sandbox before
    ``start()``.

    BenchFlow provides the sandbox infrastructure; it does **not**
    orchestrate agent-internal loops or tool protocols (ENG-50).
    """

    async def exec(
        self, cmd: str, *, user: str = "root", timeout_sec: int = 30
    ) -> ExecResult: ...

    async def read_file(self, path: str) -> bytes: ...
    async def write_file(self, path: str, content: bytes) -> None: ...

    async def upload_file(self, src: Path, dst: str) -> None: ...
    async def upload_dir(
        self, src: Path, dst: str, service: str = "main"
    ) -> None: ...
    async def download_file(self, src: str, dst: Path) -> None: ...
    async def download_dir(
        self, src: str, dst: Path, service: str = "main"
    ) -> None: ...

    async def start(self) -> None: ...
    async def stop(self, *, delete: bool = True) -> None: ...

    @property
    def host(self) -> str: ...

    @property
    def expose_ports(self) -> list[int]:
        """Ports the sandbox exposes for inter-agent communication.

        Defaults to an empty list.  Sandbox implementations that support
        port mapping should honour this list at ``start()`` time.
        """
        ...


@dataclass(frozen=True)
class ImageRef:
    tag: str
    digest: str | None = None


@dataclass
class ImageConfig:
    dockerfile: Path
    context_dir: Path
    build_args: dict[str, str] | None = None
    cache_key: str | None = None


class ImageBuilder(Protocol):
    """Build-only: produces image refs from Dockerfiles/configs."""

    async def build(self, config: ImageConfig) -> ImageRef: ...
    async def cached(self, config: ImageConfig) -> ImageRef | None: ...
