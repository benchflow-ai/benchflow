"""DockerSandbox's download check keeps its daemon connection open until it is
answered (EN-718; the DinD twin is EN-482, tests/test_dind_dial_stdio_held.py).

``DockerSandbox._download_path_stat`` sends an archive HEAD through ``docker
system dial-stdio`` with the request as the process's whole standard input.
When it ends, dial-stdio half-closes the connection, and Docker Engine 29.9
answers a missing path with 499 ("request cancelled by client") instead of
404, so ``_check_download_file`` raised "Docker archive metadata failed (HTTP
499)" instead of RemoteFileAbsentError. The fake ``docker`` below answers as
29.9 does: a missing path is 499 when stdin is already at end of file after
the request and 404 otherwise; an existing path is 200 either way.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

import pytest

from benchflow.sandbox import _download, daytona_dind
from benchflow.sandbox._download import RemoteFileAbsentError
from benchflow.sandbox.docker import DockerSandbox

FAKE_DOCKER = """#!{python}
import base64, json, os, select, sys
from urllib.parse import parse_qs, urlsplit

argv = sys.argv[1:]
while argv and argv[0] != "system":
    argv = argv[1:]
if argv != ["system", "dial-stdio"]:
    sys.exit(2)
data = b""
while b"\\r\\n\\r\\n" not in data:
    chunk = os.read(0, 4096)
    if not chunk:
        break
    data += chunk
endpoint = data.decode().split("\\r\\n")[0].split()[1]
path = parse_qs(urlsplit(endpoint).query)["path"][0]
ready, _, _ = select.select([0], [], [], 0.5)
closed = bool(ready) and os.read(0, 1) == b""
stat = {{"/": 0x800001ED, "/logs": 0x800001ED, "/logs/present.txt": 0o644}}
if path in stat:
    meta = base64.b64encode(json.dumps({{"mode": stat[path], "linkTarget": ""}}).encode()).decode()
    head = "HTTP/1.1 200 OK\\r\\nX-Docker-Container-Path-Stat: " + meta
else:
    head = "HTTP/1.1 499 status code 499" if closed else "HTTP/1.1 404 Not Found"
sys.stdout.write(head + "\\r\\nConnection: close\\r\\n\\r\\n")
"""


@pytest.fixture
def fake_docker(tmp_path: Path, monkeypatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER.format(python=sys.executable))
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return docker


def _sandbox(flags: list[str]) -> DockerSandbox:
    sandbox = object.__new__(DockerSandbox)
    sandbox._docker_compose_global_args = lambda: flags
    sandbox._docker_process_env = lambda _environment=None: dict(os.environ)
    return sandbox


def test_the_dind_command_is_unchanged():
    """EN-482's DinD command, byte for byte, from the shared snippet."""
    assert (
        "sh -c "
        + shlex.quote(
            'd=$(mktemp -d) && mkfifo "$d/in" || exit 1; '
            'docker system dial-stdio < "$d/in" & pid=$!; '
            'exec 3>"$d/in"; cat >&3; '
            'wait "$pid"; rc=$?; exec 3>&-; rm -rf "$d"; exit "$rc"'
        )
        == daytona_dind.DIAL_STDIO_HELD
    )


def test_the_held_argv_keeps_the_daemon_flags():
    argv = _download.dial_stdio_held(
        ["docker", "-H", "ssh://h", "system", "dial-stdio"]
    )
    assert argv[:2] == ["sh", "-c"] and argv[3:] == [
        "sh",
        "docker",
        "-H",
        "ssh://h",
        "system",
        "dial-stdio",
    ]


@pytest.mark.parametrize("flags", [[], ["--config", "/nonexistent-config"]])
async def test_a_missing_path_reads_as_absent_not_as_a_failure(fake_docker, flags):
    sandbox = _sandbox(flags)
    assert await sandbox._download_path_stat("c", "/logs/missing.txt") is None
    with pytest.raises(RemoteFileAbsentError):
        await sandbox._check_download_file("c", "/logs/missing.txt")


async def test_an_existing_path_is_unchanged(fake_docker):
    sandbox = _sandbox([])
    metadata = await sandbox._download_path_stat("c", "/logs/present.txt")
    assert metadata == {"mode": 0o644, "linkTarget": ""}
    await sandbox._check_download_file("c", "/logs/present.txt")


async def test_the_fake_answers_a_half_closed_request_as_engine_29_9(fake_docker):
    """The control: the same request on a plain (half-closing) dial-stdio."""
    import subprocess

    request = (
        b"HEAD /containers/c/archive?path=%2Flogs%2Fmissing.txt HTTP/1.1\r\n"
        b"Host: docker\r\nConnection: close\r\n\r\n"
    )
    out = subprocess.run(
        [str(fake_docker), "system", "dial-stdio"],
        input=request,
        capture_output=True,
        timeout=30,
    ).stdout
    assert out.startswith(b"HTTP/1.1 499")
