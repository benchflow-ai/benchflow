"""The DinD download check keeps its daemon connection open until it is
answered (PR #146).

Guards against the Docker Engine 29.9 failure the release gate of PR #146 found:
``_download_path_stat`` fed the archive HEAD request to ``docker system
dial-stdio`` from a pipe, so dial-stdio half-closed the connection as soon as
the request was written. Engine 29.9 treats that as the client going away and
answers a missing path with 499 ("request cancelled by client") instead of
404, so every absence check failed with "DinD archive metadata failed (HTTP
499)"; Engine 29.1 answered 404. The fake ``docker`` below answers as 29.9
does: 499 when its stdin is already at end of file after the request, 404
otherwise.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.daytona_dind import DIAL_STDIO_HELD, _DaytonaDinD

FAKE_DOCKER = """#!{python}
import os, select, sys

if sys.argv[1:] != ["system", "dial-stdio"]:
    sys.exit(2)
data = b""
while b"\\r\\n\\r\\n" not in data:
    chunk = os.read(0, 4096)
    if not chunk:
        break
    data += chunk
ready, _, _ = select.select([0], [], [], 0.5)
closed = bool(ready) and os.read(0, 1) == b""
status = "499 status code 499" if closed else "404 Not Found"
sys.stdout.write("HTTP/1.1 " + status + "\\r\\nConnection: close\\r\\n\\r\\n")
"""

REQUEST = b"HEAD /containers/c/archive?path=%2Fno-such HTTP/1.1\r\nHost: docker\r\nConnection: close\r\n\r\n"


@pytest.fixture
def fake_docker(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER.format(python=sys.executable))
    docker.chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}


def _run(command: str, env: dict[str, str]) -> str:
    # As the VM runs it: /bin/sh -c, the payload on stdin, then end of file.
    return subprocess.run(
        ["/bin/sh", "-c", command],
        input=REQUEST,
        env=env,
        capture_output=True,
        timeout=30,
        check=True,
    ).stdout.decode()


def test_the_fake_answers_a_half_closed_request_as_engine_29_9(fake_docker):
    assert _run("docker system dial-stdio", fake_docker).startswith("HTTP/1.1 499")


def test_the_held_connection_gets_the_daemons_404(fake_docker):
    assert _run(DIAL_STDIO_HELD, fake_docker).startswith("HTTP/1.1 404")


async def test_a_missing_path_reads_as_absent_not_as_a_failure(fake_docker):
    strategy = object.__new__(_DaytonaDinD)

    async def vm(command, **kwargs):
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=fake_docker,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(kwargs["stdin_payload"]), timeout=20
        )
        return ExecResult(
            stdout=stdout.decode(),
            stderr=stderr.decode(),
            return_code=process.returncode,
        )

    strategy._vm_exec = vm
    assert await strategy._download_path_stat("c", "/no-such") is None
