"""Root's commands find their programs on a trusted PATH while a trial is
verified (V3-CONFIRM-1).

PT's root route gives every exec as root that sets no
PATH of its own the trusted PATH, through the backend's env-file bootstrap,
including the commands a backend's own upload helpers run. The Docker
lifecycle regression is in ``test_verifier_environment_refusals.py``. On PT's
route (merge wave 32b) every root exec already pins ``ROOT_EXEC_PATH``, and
Docker-in-Docker's lower exec takes a route with no default, so V3's DinD
upload test is covered by PT's route tests.
"""

from __future__ import annotations

import pytest

from benchflow.sandbox._base import (
    ROOT_EXEC_PATH,
    SYSTEM_PATH,
    ExecResult,
    with_system_path,
)
from benchflow.sandbox._task_route import as_task_command
from benchflow.sandbox.docker import DockerSandbox


class _Call(list):
    """A transport argv, with the env the exec fed on its stdin."""

    stdin: bytes = b""


def _env_file(call: _Call) -> str:
    """The env the exec's shell sources from stdin (integ b156), as text."""
    return call.stdin.decode()


def _docker():
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._persistent_env = {"PATH": "/agent/bin:/usr/bin:/bin"}
    sandbox.default_user = None
    calls: list[list[str]] = []

    async def transport(command, **kwargs):
        call = _Call(command)
        call.stdin = kwargs.get("stdin_data") or b""
        calls.append(call)
        return ExecResult(stdout="", stderr="", return_code=0)

    sandbox._run_docker_compose_command = transport
    return sandbox, calls


@pytest.mark.asyncio
async def test_root_execs_get_the_trusted_path():
    # PT's route pins root's PATH on every exec; V3's root_commands_on_path
    # block, which only repeated it, is gone (batch 32b review B, B32B-B-3).
    sandbox, calls = _docker()

    await sandbox.exec("chmod +x /tests/test.sh", user="root")
    await sandbox.exec("id", user=0)
    await sandbox.exec("whoami", user="agent")
    # test.sh runs on the task route (PT), where an explicit PATH counts;
    # root plumbing always pins its own.
    with as_task_command(sandbox):
        await sandbox.exec("/tests/test.sh", env={"PATH": "/judged"}, user="root")

    for call in calls[:2]:
        assert call[-1].startswith(with_system_path("", ROOT_EXEC_PATH))
        assert "/agent/bin" not in call[-1]
        assert "LD_PRELOAD=" in call
        assert SYSTEM_PATH in call[-1]
    # The agent's exec and the explicit PATH keep their own (PT's route
    # carries the image PATH as data).
    rest = [_env_file(call) + call[-1] for call in calls[2:]]
    assert any("/judged" in text for text in rest)
    assert any("/agent/bin" in text for text in rest)


@pytest.mark.asyncio
async def test_root_route_cannot_be_replaced_by_persistent_agent_loader_state():
    """Guards the 85b87bc9 merge: persistent agent state never replaces PT's root route."""
    sandbox, calls = _docker()
    sandbox._persistent_env["LD_PRELOAD"] = "/agent/lib/injected.so"
    await sandbox.exec("mkdir -p /tests", user="root")
    call = calls[0]
    assert "LD_PRELOAD=" in call
    assert "/agent/lib/injected.so" not in call[-1]
    assert "/agent/bin" not in call[-1]
    assert call[-1].startswith(with_system_path("", ROOT_EXEC_PATH))


def test_env_file_cleanup_preserves_a_quoted_runtime_route(tmp_path, monkeypatch):
    """Guards f29d30f1: PT's route may itself contain shell quoting."""
    import shlex
    import shutil
    import subprocess

    from benchflow.sandbox import _base

    tools = tmp_path / "root runtime"
    tools.mkdir()
    for tool in ("rm", "base64"):
        (tools / tool).symlink_to(shutil.which(tool))
    monkeypatch.setattr(
        _base,
        "with_system_path",
        lambda command: f"PATH={shlex.quote(str(tools))}; export PATH\n{command}",
    )
    prefix = str(tmp_path / "env-")
    command = _base.wrap_command_with_env_file(
        {"VALUE": "intact"},
        'printf %s "$VALUE"',
        env_path_prefix=prefix,
    )
    done = subprocess.run(
        ["/bin/sh", "-c", command], capture_output=True, text=True, timeout=10
    )
    assert (done.returncode, done.stdout) == (0, "intact"), done.stderr
    assert list(tmp_path.glob("env-*")) == []
