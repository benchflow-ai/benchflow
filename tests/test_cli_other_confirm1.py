"""Guards CONF-1 at 5bb84a1b: live Docker env handoff has no filesystem alias."""

import asyncio
import os
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox.process.docker import DockerProcess


@pytest.mark.parametrize(
    "value", ["dummy-next-role", "dummy quoted value", "dummy-first\nsecond", ""]
)
@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("alias", ["symlink", "hardlink", "directory", "absent"])
async def test_environment_handoff_uses_only_the_live_pipe(
    tmp_path, monkeypatch, remote, alias, value
):
    """5bb84a1b used a shared name that could alias a previous role's file."""
    saved = tmp_path / "dummy-saved"
    saved.write_text("dummy-before")
    legacy = tmp_path / "dummy-env"
    if alias == "symlink":
        legacy.symlink_to(saved)
    elif alias == "hardlink":
        legacy.hardlink_to(saved)
    elif alias == "directory":
        legacy.mkdir()
    # Scope the old implementation's actual write to this disposable fixture.
    monkeypatch.setattr(DockerProcess, "_ENV_PATH", str(legacy), raising=False)
    process = DockerProcess(
        "dummy-project",
        str(tmp_path),
        [],
        global_args=["--context", "dummy"] if remote else [],
    )
    monkeypatch.setattr(process, "_host_env", lambda: dict(os.environ))
    monkeypatch.setenv("BENCHFLOW_ACP_LINE_LIMIT", "0")
    spawn = asyncio.create_subprocess_exec
    calls = []

    async def local_exec(*args, **kwargs):
        calls.append(args)
        # Execute the transport's actual shell under this test container's
        # unprivileged UID; no Docker daemon or host paths are involved.
        return await spawn("/bin/bash", "-c", args[-1], **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", local_exec)
    try:
        await process.start(
            'printf "%s\\n" "$DUMMY_LOGIN"; IFS= read -r line; printf "%s\\n" "$line"',
            env={"DUMMY_LOGIN": value},
        )
        await process.writeline('{"dummy":"protocol"}')
        for line in (value + "\n").encode().splitlines(keepends=True):
            assert await process.readline() == line
        assert await process.readline() == b'{"dummy":"protocol"}\n'
        assert saved.read_text() == "dummy-before"
        assert len(calls) == 1
        assert "dummy-next-role" not in str(calls)
    finally:
        await process.close()


async def test_pipe_handoff_failure_closes_the_started_transport(monkeypatch):
    """5bb84a1b's handoff had a separate writer outside start rollback."""
    process = DockerProcess("dummy-project", "/tmp/dummy-project", [])
    monkeypatch.setattr(process, "_host_env", lambda: {})
    child = AsyncMock()
    child.communicate.return_value = (b"", b"")
    child.stdin.write = lambda data: None
    child.stdin.drain.side_effect = BrokenPipeError("dummy pipe closure")
    child.stderr = None
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", AsyncMock(return_value=child)
    )
    closed = AsyncMock()
    monkeypatch.setattr(process, "close", closed)
    with pytest.raises(BrokenPipeError):
        await process.start("true", env={"DUMMY_LOGIN": "dummy-login"})
    closed.assert_awaited_once()


@pytest.mark.parametrize("startup", ["path", "bash-env"])
async def test_private_handoff_bootstrap_ignores_prior_role_startup(
    tmp_path, monkeypatch, startup
):
    """Guards CONF-1's bootstrap closure at 56977cca with harmless markers."""
    marker = tmp_path / "dummy-bootstrap-ran"
    env = dict(os.environ)
    if startup == "path":
        for name, trusted in (("bash", "/bin/bash"), ("base64", "/usr/bin/base64")):
            script = tmp_path / name
            script.write_text(f'#!/bin/sh\ntouch "{marker}"\nexec {trusted} "$@"\n')
            script.chmod(0o755)
        env["PATH"] = f"{tmp_path}:/usr/bin:/bin"
    else:
        hook = tmp_path / "dummy-bash-env"
        hook.write_text(f'touch "{marker}"\n')
        env["BASH_ENV"] = str(hook)
    process = DockerProcess("dummy-project", str(tmp_path), [])
    monkeypatch.setattr(process, "_host_env", lambda: env)
    monkeypatch.setenv("BENCHFLOW_ACP_LINE_LIMIT", "0")
    spawn = asyncio.create_subprocess_exec

    async def local_exec(*args, **kwargs):
        command = args[args.index(process._service) + 1 :]
        return await spawn(*command, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", local_exec)
    try:
        await process.start(
            'printf "%s\\n" "$DUMMY_LOGIN"', env={"DUMMY_LOGIN": "dummy-next-role"}
        )
        assert await process.readline() == b"dummy-next-role\n"
        assert not marker.exists()
    finally:
        await process.close()
