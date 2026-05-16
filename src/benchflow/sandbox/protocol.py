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
    """Run-only: isolated execution environment."""

    async def exec(
        self, cmd: str, *, user: str = "root", timeout_sec: int = 30
    ) -> ExecResult: ...

    async def read_file(self, path: str) -> bytes: ...
    async def write_file(self, path: str, content: bytes) -> None: ...

    async def upload_file(self, src: Path, dst: str) -> None: ...
    async def upload_dir(self, src: Path, dst: str) -> None: ...
    async def download_file(self, src: str, dst: Path) -> None: ...

    async def start(self) -> None: ...
    async def stop(self, *, delete: bool = True) -> None: ...

    @property
    def host(self) -> str: ...


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
