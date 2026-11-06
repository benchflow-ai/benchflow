"""A stand-in for ``docker cp CONTAINER:SRC -``, the Docker download transport.

Docker downloads stream the copied entry as a tar on stdout and publish it
through held descriptors; tests that fake the Docker CLI answer
that one call here.
"""

from __future__ import annotations

import asyncio
import io
import os
import tarfile
from pathlib import PurePosixPath
from types import SimpleNamespace
from unittest.mock import AsyncMock


def fake_docker_cp(
    monkeypatch, files: dict[str, bytes], *, directory: bool = True
) -> list[tuple[str, ...]]:
    """Answer ``docker cp X:SRC -`` with ``files`` below SRC's basename.

    Returns the list the argv of every call is appended to.
    """
    calls: list[tuple[str, ...]] = []

    async def docker_cp(*argv, stdout=None, **kwargs):
        calls.append(tuple(argv))
        assert argv[-1] == "-" and isinstance(stdout, int)
        base = PurePosixPath(argv[-2].split(":", 1)[1]).name
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            if directory:
                folder = tarfile.TarInfo(base)
                folder.type = tarfile.DIRTYPE
                folder.mode = 0o755
                tar.addfile(folder)
            for name, payload in files.items():
                member = tarfile.TarInfo(f"{base}/{name}" if directory else base)
                member.size = len(payload)
                member.mode = 0o644
                tar.addfile(member, io.BytesIO(payload))
        os.write(stdout, buf.getvalue())
        stderr = asyncio.StreamReader()
        stderr.feed_eof()
        return SimpleNamespace(
            stderr=stderr, wait=AsyncMock(return_value=0), kill=lambda: None
        )

    monkeypatch.setattr("asyncio.create_subprocess_exec", docker_cp)
    return calls
