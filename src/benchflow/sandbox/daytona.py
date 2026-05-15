"""DaytonaSandbox — adapts Harbor's DaytonaEnvironment to the Sandbox protocol."""

from __future__ import annotations

import os
import shlex
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from benchflow.sandbox.protocol import ExecResult

if TYPE_CHECKING:
    from harbor.environments.daytona import DaytonaEnvironment


class DaytonaSandbox:
    """Adapts Harbor's DaytonaEnvironment to the Sandbox protocol."""

    def __init__(self, inner: DaytonaEnvironment, *, expose_ports: list[int] | None = None) -> None:
        self._inner = inner
        self._expose_ports = expose_ports or []

    async def exec(
        self, cmd: str, *, user: str = "root", timeout_sec: int = 30
    ) -> ExecResult:
        result = await self._inner.exec(cmd, user=user, timeout_sec=timeout_sec)
        return ExecResult(
            return_code=result.return_code,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    async def read_file(self, path: str) -> bytes:
        result = await self._inner.exec(f"cat {shlex.quote(path)}", timeout_sec=30)
        if result.return_code != 0:
            raise FileNotFoundError(
                f"read_file failed (rc={result.return_code}): {result.stderr or ''}"
            )
        return (result.stdout or "").encode()

    async def write_file(self, path: str, content: bytes) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            try:
                await self._inner.upload_file(tmp.name, path)
            finally:
                os.unlink(tmp.name)

    async def upload_file(self, src: Path, dst: str) -> None:
        await self._inner.upload_file(src, dst)

    async def upload_dir(self, src: Path, dst: str) -> None:
        await self._inner.upload_dir(src, dst)

    async def download_file(self, src: str, dst: Path) -> None:
        await self._inner.download_file(src, dst)

    async def start(self) -> None:
        await self._inner.start(force_build=False)

    async def stop(self, *, delete: bool = True) -> None:
        await self._inner.stop(delete=delete)

    @property
    def host(self) -> str:
        return "localhost"

    @property
    def expose_ports(self) -> list[int]:
        return list(self._expose_ports)
