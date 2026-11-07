"""Guards FF-sw-launchers: an unresolved identity never becomes root (363d815b)."""

import subprocess
from unittest.mock import MagicMock

import pytest

from benchflow.sandbox._base import SYSTEM_SHELL_ROUTE, ExecRoute
from benchflow.sandbox.agentcore import AgentCoreSandbox
from benchflow.task.config import SandboxConfig

# A named user that is not root, as exec_route gives it (C8).
USER_ROUTE = ExecRoute(pin=None, task_root=False, root=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("user", [424242, "424242", "", -1, "-l", "0:0"])
async def test_unresolved_identity_never_executes(tmp_path, user):
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    env = AgentCoreSandbox(
        environment_dir=tmp_path,
        environment_name="task",
        session_id="s",
        rollout_paths=None,
        task_env_config=SandboxConfig(),
    )
    payloads = []
    env._invoke_command = MagicMock(
        side_effect=lambda payload, timeout: payloads.append(payload)
    )
    try:
        await env._dispatch_command(
            "printf EXECUTED", timeout_sec=10, resolved_user=user, route=USER_ROUTE
        )
    except ValueError:
        return
    result = subprocess.run(
        ["/bin/bash", "-c", payloads[0]], capture_output=True, timeout=5
    )
    assert b"EXECUTED" not in result.stdout
    assert result.returncode != 0


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["modal", "daytona", "daytona-system"])
@pytest.mark.parametrize("user", [424242, "424242", "", -1, "-l"])
async def test_sibling_backends_refuse_unresolved_identity(tmp_path, backend, user):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from benchflow.sandbox.daytona import DaytonaSandbox
    from benchflow.sandbox.modal_impl import ModalSandbox

    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    cls = ModalSandbox if backend == "modal" else DaytonaSandbox
    env = cls(
        environment_dir=tmp_path,
        environment_name="task",
        session_id="s",
        rollout_paths=None,
        task_env_config=SandboxConfig(),
    )
    env._sandbox = MagicMock()
    try:
        if backend == "modal":
            process = MagicMock()
            process.stdout.read.aio = AsyncMock(return_value="")
            process.stderr.read.aio = AsyncMock(return_value="")
            process.wait.aio = AsyncMock(return_value=0)
            env._sandbox.exec.aio = AsyncMock(return_value=process)
            await env.exec("printf EXECUTED", user=user)
            argv = env._sandbox.exec.aio.call_args.args
        else:
            env._sandbox.process.create_session = AsyncMock()
            env._sandbox.process.execute_session_command = AsyncMock(
                return_value=SimpleNamespace(cmd_id="c")
            )
            env._poll_response = AsyncMock()
            await env._sandbox_exec(
                "printf EXECUTED",
                user=user,
                route=SYSTEM_SHELL_ROUTE if backend == "daytona-system" else USER_ROUTE,
            )
            argv = [
                "/bin/bash",
                "-c",
                env._sandbox.process.execute_session_command.call_args.args[1].command,
            ]
    except ValueError:
        return
    result = subprocess.run(argv, capture_output=True, timeout=5)
    assert b"EXECUTED" not in result.stdout
    assert result.returncode != 0


@pytest.mark.parametrize("user", [0, "0", "root", 65534, "65534", "nobody"])
def test_known_accounts_remain_literal_and_missing_getent_fails_closed(tmp_path, user):
    from benchflow.sandbox._su import su_command

    tools = tmp_path / "tools"
    tools.mkdir()
    su = tools / "su"
    su.write_text('#!/bin/sh\nprintf "ACCOUNT=%s\\n" "$1"\n')
    su.chmod(0o755)
    if isinstance(user, int) or user.isdigit():
        getent = tools / "getent"
        getent.write_text('#!/bin/sh\nprintf "nobody:x:65534:65534::/:/bin/sh\\n"\n')
        getent.chmod(0o755)
    command = su_command("printf EXECUTED", user)
    result = subprocess.run(
        ["/bin/bash", "-c", command],
        env={"PATH": str(tools)},
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert result.stdout.startswith(b"ACCOUNT=")
    assert b"ACCOUNT=\n" not in result.stdout
    if isinstance(user, int) or user.isdigit():
        getent.unlink()
        result = subprocess.run(
            ["/bin/bash", "-c", command],
            env={"PATH": str(tools)},
            capture_output=True,
            timeout=5,
        )
        assert result.returncode != 0 and not result.stdout
