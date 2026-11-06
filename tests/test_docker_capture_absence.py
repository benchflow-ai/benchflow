"""Guards FIX-49 r6's Docker recovery regression at 0294404ae.

Both transports run their real download/metadata handlers. Only subprocess
creation is replaced, supplying daemon HTTP replies and actual tar streams.
"""

import asyncio
import base64
import io
import json
import os
import tarfile
from pathlib import PurePosixPath
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import pytest

from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.remote_docker import RemoteDockerHost, RemoteDockerSandbox
from tests.test_verifier_own_failures import _assess, _task


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("present", ["reward.json", "reward.txt"])
@pytest.mark.parametrize(
    "companion",
    [
        "absent",
        "directory",
        "link",
        "transport",
        "lost-container",
        "lost-file",
        "unsafe-parent",
        "missing-cli",
    ],
)
async def test_recovery_uses_verified_remote_absence(
    tmp_path, monkeypatch, remote, present, companion
):
    sandbox, verifier = _task(tmp_path)
    sandbox.is_mounted = False
    backend = object.__new__(RemoteDockerSandbox if remote else DockerSandbox)
    if remote:
        backend.docker_host = RemoteDockerHost("ssh://fixture.invalid")
    backend._main_container = "a" * 64
    backend.environment_name = "fixture"
    backend._docker_compose_global_args = lambda: (
        ["-H", "ssh://fixture.invalid"] if remote else []
    )
    backend._docker_compose_env = lambda: {}
    backend._compose_transport_env = lambda: {}
    backend._docker_process_env = lambda _environment=None: {}
    metadata_calls, copy_calls = [], []
    missing = "reward.txt" if present.endswith("json") else "reward.json"

    async def spawn(*argv, stdout=None, **kwargs):
        assert argv[0] == "docker"
        if companion == "missing-cli":
            raise FileNotFoundError("fixture docker CLI unavailable")
        if remote:
            assert argv[1:3] == ("-H", "ssh://fixture.invalid")
        if argv[-2:] == ("system", "dial-stdio"):

            async def communicate(request):
                verb, endpoint, _ = request.decode().split("\r\n")[0].split()
                assert verb == "HEAD"
                source = parse_qs(urlsplit(endpoint).query)["path"][0]
                metadata_calls.append(source)
                name = PurePosixPath(source).name
                status, mode, link = 404, None, ""
                if name == present:
                    status, mode = 200, 0o644
                elif source in ("/logs/verifier", "/logs", "/"):
                    status, mode = (
                        (404, None)
                        if companion == "lost-container"
                        else (200, 0x800001ED)
                    )
                    if companion == "unsafe-parent" and source == "/logs/verifier":
                        status, mode, link = 200, 0x080001FF, "/missing"
                elif name == missing:
                    if companion == "directory":
                        status, mode = 200, 0x800001ED
                    elif companion == "link":
                        status, mode, link = 200, 0x080001FF, "/missing"
                    elif companion == "lost-file":
                        status, mode = 200, 0o644
                    elif companion == "transport":
                        status = 500
                header = f"HTTP/1.1 {status} fixture\r\n"
                if mode is not None:
                    data = json.dumps({"mode": mode, "linkTarget": link}).encode()
                    header += (
                        "X-Docker-Container-Path-Stat: "
                        + base64.b64encode(data).decode()
                        + "\r\n"
                    )
                return (header + "\r\n").encode(), b""

            return SimpleNamespace(returncode=0, communicate=communicate)
        assert argv[-1] == "-" and "cp" in argv
        source = argv[-2].split(":", 1)[1]
        copy_calls.append(source)
        name = PurePosixPath(source).name
        stderr = asyncio.StreamReader()
        code = 1
        if name == present or companion in ("directory", "link"):
            content = b'{"reward":1}' if present.endswith("json") else b"1"
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode="w") as archive:
                entry = tarfile.TarInfo(name)
                if name != present:
                    entry.type = (
                        tarfile.DIRTYPE if companion == "directory" else tarfile.SYMTYPE
                    )
                    entry.linkname = "/missing" if companion == "link" else ""
                    archive.addfile(entry)
                else:
                    entry.size = len(content)
                    archive.addfile(entry, io.BytesIO(content))
            os.write(stdout, payload.getvalue())
            code = 0
        else:
            # Text claiming absence is deliberately identical for a true
            # absence and a transport/capture failure of an existing file.
            stderr.feed_data(b"Could not find the file in container\n")
        stderr.feed_eof()
        return SimpleNamespace(
            stderr=stderr, wait=AsyncMock(return_value=code), kill=lambda: None
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    sandbox.download_dir = AsyncMock(
        side_effect=RuntimeError("fixture bulk unavailable")
    )
    sandbox.download_file = backend.download_file
    rewards, error, warnings = await _assess(sandbox, verifier)
    if companion == "absent":
        assert rewards == {"reward": 1.0} and error is None and not warnings
    elif companion in ("directory", "link", "unsafe-parent"):
        assert rewards == {"reward": 0.0} and error is None and warnings
    else:
        assert rewards is None and error is not None and not warnings
    # Both arms must reach the real adapter for the companion. The repaired
    # arm can establish absence without asking cp to copy a nonexistent file.
    if companion == "missing-cli":
        assert "docker CLI unavailable" in error
    else:
        assert f"/logs/verifier/{missing}" in metadata_calls + copy_calls
    if companion == "lost-file":
        assert f"/logs/verifier/{missing}" in copy_calls
