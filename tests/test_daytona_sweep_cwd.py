"""Guards FF-sw-daytona literal working directories at 363d815b."""

import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import SYSTEM_SHELL_ROUTE, ExecResult, ExecRoute
from benchflow.sandbox.daytona import DaytonaSandbox, _load_daytona_sdk


@pytest.mark.parametrize(
    "name",
    [
        "project files",
        "a;b",
        "a'b",
        'a"b',
        "a$(true)",
        "a`true`",
        "a\nb",
        "-project",
        "日本語",
    ],
)
@pytest.mark.parametrize(
    "shell,system_shell", [("bash -c", False), ("sh -c", False), ("bash -c", True)]
)
async def test_cwd_is_one_literal_argument(tmp_path, name, shell, system_shell):
    """Guard 363d815b: execute in the exact accepted directory on both VM shells."""
    _load_daytona_sdk()
    cwd = tmp_path / name
    cwd.mkdir()
    process = SimpleNamespace(create_session=AsyncMock())
    commands = []

    async def execute(session, request, timeout=None):
        commands.append(request.command)
        return SimpleNamespace(cmd_id="dummy")

    process.execute_session_command = execute
    env = DaytonaSandbox.__new__(DaytonaSandbox)
    env._sandbox = SimpleNamespace(process=process)
    env._poll_response = AsyncMock(
        return_value=ExecResult(return_code=0, stdout="", stderr="")
    )
    # Every exec names its route (C8): the system shell's, or a task
    # command as a named user.
    route = (
        SYSTEM_SHELL_ROUTE
        if system_shell
        else ExecRoute(pin=None, task_root=False, root=False)
    )
    await env._sandbox_exec("pwd -P", cwd=str(cwd), shell=shell, route=route)
    result = subprocess.run(["bash", "-c", commands[0]], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(cwd) + "\n"
