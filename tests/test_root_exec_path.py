"""Guard PT C8 (Rust d855e5a4): root plumbing cannot start a planted sh."""

import contextlib
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._task_route import as_task_command
from benchflow.sandbox.daytona_dind import _DaytonaDinD
from benchflow.sandbox.docker import DockerSandbox


def _sandbox(tmp_path, monkeypatch, backend="docker"):
    planted = tmp_path / "agent-bin"
    planted.mkdir()
    for tool in ("sh", "bash", "su", "find", "base64", "rm"):
        executable = planted / tool
        executable.write_text(
            f"#!/bin/sh\nprintf 'PLANTED-{tool}-uid-%s\\n' \"$(/usr/bin/id -u)\"\n"
        )
        executable.chmod(0o755)
    image_path = f"{planted}:/usr/local/bin:/usr/bin:/bin"
    if backend == "docker":
        sandbox = object.__new__(DockerSandbox)
        transport = sandbox
    else:
        # The strategy below the Daytona sandbox, which picks its route (C8).
        from benchflow.sandbox.daytona import DaytonaSandbox

        sandbox = object.__new__(DaytonaSandbox)
        transport = object.__new__(_DaytonaDinD)
        transport._env = sandbox
        sandbox._strategy = transport
    sandbox.default_user = None
    sandbox._persistent_env = {}
    # The image's agent user: uid 1000, as exec_route's lookup would read.
    sandbox._named_uids = (sandbox._exec_generation, {("main", "agent"): 1000})

    async def native_exec(argv, **kwargs):
        # The real transport's constructed argv, launched as Docker launches
        # it with the image environment. No daemon/socket/mount is needed.
        command = argv[argv.index("main") + 1 :]
        # Docker's exec carries the env on stdin.
        data = kwargs.get("stdin_data")
        ran = subprocess.run(
            command,
            env={"PATH": image_path},
            input=None if data is None else data.decode(),
            capture_output=True,
            text=True,
        )
        return ExecResult(
            return_code=ran.returncode, stdout=ran.stdout, stderr=ran.stderr
        )

    monkeypatch.setattr(
        transport,
        "_run_docker_compose_command" if backend == "docker" else "_compose_exec",
        native_exec,
    )
    return sandbox, image_path


@pytest.mark.asyncio
@pytest.mark.parametrize("user", ["root", 0, "0", "0:0", "root:root"])
@pytest.mark.parametrize("backend", ["docker", "dind"])
async def test_root_plumbing_uses_system_shell_and_tools(
    tmp_path, monkeypatch, user, backend
):
    """Guards c2db61b8: the planted shell and tools must never run as root."""
    sandbox, _image_path = _sandbox(tmp_path, monkeypatch, backend)
    result = await sandbox.exec(
        "find /dev/null -maxdepth 0; printf real-sh", user=user, env={"CHECK": "yes"}
    )
    assert result.return_code == 0, result.stderr
    assert "PLANTED" not in result.stdout
    assert result.stdout.endswith("real-sh")


@pytest.mark.asyncio
@pytest.mark.parametrize("user", [None, "agent"])
@pytest.mark.parametrize("backend", ["docker", "dind"])
async def test_task_roles_keep_the_image_path(tmp_path, monkeypatch, user, backend):
    """Guards C8's boundary: oracle/agent commands still see image tools."""
    sandbox, image_path = _sandbox(tmp_path, monkeypatch, backend)
    # The task's own command (C8's task route); BenchFlow's user=None
    # plumbing pins PATH (tests/test_c8_loader_routes.py).
    with as_task_command(sandbox):
        result = await sandbox.exec(
            'printf %s "$PATH"', user=user, env={"CHECK": "yes"}
        )
    assert result.stdout == image_path


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "dind"])
async def test_verifier_keeps_its_explicit_trusted_path(tmp_path, monkeypatch, backend):
    """Guards C8's boundary: verified venv extras remain available."""
    sandbox, _ = _sandbox(tmp_path, monkeypatch, backend)
    trusted = tmp_path / "trusted-bin"
    trusted.mkdir()
    tool = trusted / "verifier-tool"
    tool.write_text("#!/bin/sh\nprintf trusted-verifier")
    tool.chmod(0o755)
    # On the task route, as verifier_core runs test.sh.
    with as_task_command(sandbox):
        result = await sandbox.exec(
            "verifier-tool", user="root", env={"PATH": f"{trusted}:/usr/bin:/bin"}
        )
    assert result.return_code == 0, result.stderr
    assert result.stdout == "trusted-verifier"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["exec", "exec_transient"])
@pytest.mark.parametrize(
    "user,persistent_path",
    [("root", False), ("root", True), (None, False), (None, True)],
)
async def test_native_daytona_bootstrap_and_default_root_role_path(
    tmp_path, monkeypatch, method, user, persistent_path
):
    """Guards C8 at c2db61b8 and effective-env preservation in its fix."""
    from benchflow.sandbox import daytona as day

    _, image_path = _sandbox(tmp_path, monkeypatch)
    day._load_daytona_sdk()
    sandbox = object.__new__(day.DaytonaSandbox)
    sandbox.default_user = "root"
    # Explicit root with a planted task PATH guards PT-CONF-1 at 7a220231.
    chosen_path = "/usr/bin:/bin" if persistent_path and user is None else image_path
    sandbox._persistent_env = {"PATH": chosen_path} if persistent_path else {}
    captured = {}

    async def execute(session_id, request, **kwargs):
        command = request.command
        if os.geteuid() != 0:
            # The SDK runs as root. A non-root unit-test host emulates only
            # that identity switch; the fresh root-container check uses it
            # literally. Shell selection, quoting and PATH remain production.
            for switch in (
                "/bin/su root -s /bin/sh -c",
                "/bin/su root -s /bin/bash -c",
                "su root -s /bin/bash -c",
            ):
                command = command.replace(switch, "/bin/sh -c")
        ran = subprocess.run(
            ["/bin/sh", "-c", command],
            env={"PATH": image_path},
            capture_output=True,
            text=True,
        )
        captured["result"] = ExecResult(
            return_code=ran.returncode, stdout=ran.stdout, stderr=ran.stderr
        )
        return SimpleNamespace(cmd_id="command")

    async def poll(*args, **kwargs):
        return captured["result"]

    sandbox._sandbox = SimpleNamespace(
        process=SimpleNamespace(
            create_session=AsyncMock(),
            execute_session_command=execute,
            delete_session=AsyncMock(),
        )
    )
    sandbox._strategy = day._DaytonaDirect(sandbox)
    sandbox._poll_response = poll
    command = (
        "find /dev/null -maxdepth 0; printf real-sh"
        if user == "root"
        else 'printf %s "$PATH"'
    )
    # A default-root command that keeps its PATH is the task's own (C8's
    # task route); root is plumbing either way.
    with as_task_command(sandbox) if user is None else contextlib.nullcontext():
        result = await getattr(sandbox, method)(command, user=user)
    assert result.return_code == 0, result.stderr
    assert "PLANTED" not in result.stdout
    if user == "root":
        assert result.stdout.endswith("real-sh")
    else:
        assert result.stdout == chosen_path
