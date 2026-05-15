"""Protocols for sandbox runtimes and image builders."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol, Sequence

from benchflow.sandboxes.specs import ExecResult, ImageConfig, ImageRef, SandboxSpec


class Sandbox(Protocol):
    """Run-only isolated execution environment."""

    async def start(self) -> None: ...

    async def stop(self, *, delete: bool = True) -> None: ...

    async def exec(
        self,
        cmd: str | Sequence[str],
        *,
        user: str = "root",
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_sec: int = 30,
    ) -> ExecResult: ...

    async def read_file(self, path: str) -> bytes: ...

    async def write_file(self, path: str, content: bytes) -> None: ...

    async def upload_dir(self, src: Path, dst: str) -> None: ...

    async def download_dir(self, src: str, dst: Path) -> None: ...


class SandboxProvider(Protocol):
    """Factory for sandbox sessions."""

    name: str

    async def create(self, spec: SandboxSpec) -> Sandbox: ...


class ImageBuilder(Protocol):
    """Build-only interface that produces image references."""

    async def cached(self, config: ImageConfig) -> ImageRef | None: ...

    async def build(self, context_dir: Path, config: ImageConfig) -> ImageRef: ...
