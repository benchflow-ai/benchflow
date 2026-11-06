"""Guard C8 on root's own plumbing: the system shell, or a refusal.

PT-CONF-6's siblings, S2-CONF-1/2 (no backend advertised a trusted shell)
and V3-RECONFIRM-1 (backend helpers that bypass it). Each exec is run here
with planted tools first on the image's PATH; only marker files are written.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox._base import ExecResult
from tests._trust_admission import admitted

TOOLS = (
    "sh",
    "bash",
    "su",
    "rm",
    "ls",
    "readlink",
    "find",
    "sed",
    "mkdir",
    "chmod",
    "cat",
    "sort",
    "stat",
)


def _plant(tmp_path: Path) -> tuple[Path, str]:
    planted = tmp_path / "app/.venv/bin"
    planted.mkdir(parents=True)
    marks = tmp_path / "marks"
    marks.mkdir()
    for tool in TOOLS:
        exe = planted / tool
        exe.write_text(f"#!/bin/sh\n: > {marks}/{tool}\nexit 97\n")
        exe.chmod(0o755)
    return marks, f"{planted}:/usr/local/bin:/usr/bin:/bin"


def _ran(marks: Path) -> list[str]:
    return sorted(p.name for p in marks.iterdir())


def _run(argv, image_path, **env):
    ran = subprocess.run(
        list(argv), env={"PATH": image_path, **env}, capture_output=True, text=True
    )
    return ExecResult(return_code=ran.returncode, stdout=ran.stdout, stderr=ran.stderr)


@pytest.mark.asyncio
async def test_the_path_lock_runs_on_the_system_shell():
    """S2-CONF-1's class: the PATH check's own ls/readlink never come from PATH."""
    from benchflow.sandbox.lockdown import lock_system_path

    calls = []

    async def exec_(command, **kwargs):
        calls.append(kwargs)
        return ExecResult(return_code=0, stdout="", stderr="")

    env = SimpleNamespace(exec=exec_, supports_system_shell=True)
    await lock_system_path(env, "/app")
    assert calls and calls[0].get("system_shell") is True
    assert calls[0].get("user") == "root"


@pytest.mark.asyncio
async def test_the_path_lock_refuses_a_sandbox_without_a_system_shell():
    from benchflow.sandbox.lockdown import lock_system_path

    env = SimpleNamespace(exec=AsyncMock(), supports_system_shell=False)
    with pytest.raises(RuntimeError, match="system PATH"):
        await lock_system_path(env, "/app")
    env.exec.assert_not_called()


@pytest.mark.asyncio
async def test_the_judge_root_exec_refuses_rather_than_use_the_image_path():
    """J1's fallback ran root's script on a shell the image's PATH picked."""
    from benchflow.taskmd.verify import TaskMdVerifierError, root_exec

    sandbox = SimpleNamespace(exec=AsyncMock(), supports_system_shell=False)

    with pytest.raises(TaskMdVerifierError, match="protected root shell execution"):
        await root_exec(sandbox, "true", timeout_sec=5)
    sandbox.exec.assert_not_called()


def test_every_backend_advertises_the_system_shell():
    """S2-CONF-2: Docker, remote Docker, Daytona (direct and DinD), Modal, Apple, AgentCore."""
    from benchflow.sandbox.agentcore import AgentCoreSandbox
    from benchflow.sandbox.apple_container import AppleContainerSandbox
    from benchflow.sandbox.daytona import DaytonaSandbox
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.modal_impl import ModalSandbox
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    for cls in (
        DockerSandbox,
        RemoteDockerSandbox,
        DaytonaSandbox,
        ModalSandbox,
        AppleContainerSandbox,
        AgentCoreSandbox,
    ):
        assert object.__new__(cls).supports_system_shell is True, cls.__name__


@pytest.mark.asyncio
async def test_apple_system_shell_runs_no_planted_tool(tmp_path, monkeypatch):
    from benchflow.sandbox import apple_container

    marks, image_path = _plant(tmp_path)
    sandbox = object.__new__(apple_container.AppleContainerSandbox)
    sandbox._container_name = "bf"
    sandbox.default_user = None
    sandbox._persistent_env = {}
    captured = []

    async def run_cli(*args, timeout=None):
        captured.append(args)
        return _run(args[args.index("bf") + 1 :], image_path)

    monkeypatch.setattr(apple_container, "_run_cli", run_cli)
    result = await sandbox.exec(
        "mkdir -p /tmp/x-c8 && printf ok", user="root", system_shell=True
    )
    assert "LD_PRELOAD=" in captured[0][: captured[0].index("bf")]
    assert result.stdout == "ok", result.stderr
    assert _ran(marks) == []


@pytest.mark.asyncio
async def test_modal_system_shell_empties_loader_values_and_runs_no_planted_tool(
    tmp_path,
):
    pytest.importorskip("modal")
    from benchflow.sandbox.modal_impl import ModalSandbox

    marks, image_path = _plant(tmp_path)
    sandbox = object.__new__(ModalSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    seen = {}

    async def exec_aio(*argv, secrets=(), **kwargs):
        seen["secrets"] = secrets
        result = _run(argv, image_path)
        process = MagicMock()
        process.stdout.read.aio = AsyncMock(return_value=result.stdout)
        process.stderr.read.aio = AsyncMock(return_value=result.stderr)
        process.wait.aio = AsyncMock(return_value=result.return_code)
        return process

    sandbox._sandbox = MagicMock()
    sandbox._sandbox.exec.aio = exec_aio
    result = await sandbox.exec("ls / >/dev/null && printf ok", system_shell=True)
    assert result.stdout == "ok"
    assert seen["secrets"]  # the empty loader values travel as the exec's env
    assert _ran(marks) == []


@pytest.mark.asyncio
async def test_agentcore_system_shell_runs_no_planted_tool(tmp_path, monkeypatch):
    from benchflow.sandbox.agentcore import AgentCoreSandbox

    marks, image_path = _plant(tmp_path)
    sandbox = object.__new__(AgentCoreSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox.runtime_arn = "arn"
    sandbox.runtime_session_id = "s"
    monkeypatch.setattr(
        sandbox,
        "_invoke_command",
        lambda payload, timeout: _run(["/bin/sh", "-c", payload], image_path),
    )
    result = await sandbox.exec("ls / >/dev/null && printf ok", system_shell=True)
    assert result.stdout == "ok", result.stderr
    assert _ran(marks) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["upload_dir", "credential_run"])
async def test_daytona_direct_helpers_ask_for_the_system_shell(step):
    """V3-RECONFIRM-1: Daytona's helpers call _sandbox_exec past exec()."""
    pytest.importorskip("daytona")
    from benchflow.sandbox import daytona_strategies as strategies
    from benchflow.sandbox._base import SYSTEM_SHELL_ROUTE

    env = MagicMock()
    env._sandbox_exec = AsyncMock(
        return_value=ExecResult(return_code=0, stdout="", stderr="")
    )
    env._sdk_upload_dir = AsyncMock()
    if step == "upload_dir":
        strategy = object.__new__(strategies._DaytonaDirect)
        strategy._env = env
        await strategy.upload_dir("/host/x", "/app/x")
    else:
        await strategies._DaytonaCredentialOps(env).run("true")
    calls = env._sandbox_exec.await_args_list
    assert calls and all(c.kwargs.get("route") == SYSTEM_SHELL_ROUTE for c in calls)


@pytest.mark.asyncio
async def test_docker_windows_line_ending_fix_runs_on_the_system_shell(monkeypatch):
    """V3-RECONFIRM-1: the Windows upload's bash/find/sed, as root by path."""
    from benchflow.sandbox import docker as docker_mod

    sandbox = object.__new__(docker_mod.DockerSandbox)
    calls = []

    async def exec_(command, **kwargs):
        calls.append((command, kwargs))
        return ExecResult(return_code=0, stdout="", stderr="")

    sandbox.exec = exec_
    sandbox._copy = AsyncMock()
    sandbox._run_docker_compose_command = AsyncMock()
    monkeypatch.setattr(sys, "platform", "win32")
    await sandbox.upload_dir("/host/x", "/app/x", admission=admitted(sandbox))
    sandbox._run_docker_compose_command.assert_not_called()
    fixes = [kw for cmd, kw in calls if "sed -i" in cmd]
    assert fixes and fixes[0].get("system_shell") is True
