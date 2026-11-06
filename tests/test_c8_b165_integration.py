"""Guards b165's a844a8ad merge against bypassing PT fe895b04's root route."""

from __future__ import annotations

import shlex
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import ExecResult
from tests._firewall_tools import audit_image_tools  # noqa: F401

KIT = "/opt/runtime/rt"


def route():
    return SimpleNamespace(
        pin=f"{KIT}/bin:/usr/bin:/bin",
        tools=f"{KIT}/bin:/usr/bin:/bin",
        shell=f"{KIT}/bin/sh",
        bash=f"{KIT}/bin/bash",
        kit=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["script", "identity", "dind"])
async def test_new_identity_and_script_execs_take_the_service_route(kind):
    """Guards the DT2 a844a8ad merge: even supervision starts on PT's kit."""
    if kind == "dind":
        from benchflow.sandbox.daytona_dind import _DaytonaDinD

        env = object.__new__(_DaytonaDinD)
        env._env = SimpleNamespace(exec_route=AsyncMock(return_value=route()))
        env._identity_container_id = AsyncMock(return_value="db-container")
        env._vm_exec = AsyncMock(return_value=ExecResult(return_code=0))
        send = env._vm_exec
    else:
        from benchflow.sandbox.docker import DockerSandbox

        env = object.__new__(DockerSandbox)
        env.exec_route = AsyncMock(return_value=route())
        env._run_docker_compose_command = AsyncMock(
            return_value=ExecResult(return_code=0)
        )
        send = env._run_docker_compose_command
    env.identity_container_config = AsyncMock(
        return_value={"Env": ["LD_PRELOAD=/app/evil.so"]}
    )
    env.script_container_config = env.identity_container_config
    if kind == "script":
        await env.script_runner_exec("printf safe", cwd="/app")
    else:
        await env.identity_exec("printf safe", service="db")
    argv = send.call_args.args[0]
    if kind == "dind":
        # A plain docker exec in the VM's shell (SCRIPT-AGENT-DOCKER-F1).
        argv = shlex.split(argv.partition("; docker ")[2])
    assert f"{KIT}/bin/env" in argv
    assert f"{KIT}/bin/{'bash' if kind == 'script' else 'sh'}" in argv
    # The image's loader value is removed (bare -e NAME), never set, not
    # even empty (SCRIPT-AGENT-DOCKER-F1).
    assert "LD_PRELOAD" in argv
    assert not [a for a in argv if a.startswith("LD_PRELOAD=")]
    if kind != "script":
        chooser = env._env.exec_route if kind == "dind" else env.exec_route
        assert chooser.call_args.kwargs["service"] == "db"


@pytest.mark.asyncio
async def test_bind_probe_uses_the_owning_services_kit():
    """Guards S1 a844a8ad: bind identity checks must not use live stat."""
    from benchflow.sandbox._bind_targets import inspect_targets

    env = SimpleNamespace(
        exec_route=AsyncMock(return_value=route()),
        _docker_cli=AsyncMock(return_value=ExecResult(stdout="7:9\n", return_code=0)),
    )
    inspected = {
        "Mounts": [{"Type": "bind", "Destination": "/data"}],
        "Config": {"Labels": {"com.docker.compose.service": "db"}},
    }
    assert await inspect_targets(env, "c" * 64, inspected) == {"/data": "7:9"}
    argv = env._docker_cli.call_args.args[0]
    assert f"{KIT}/bin/sh" in argv and "/usr/bin/stat" not in argv
    assert env.exec_route.call_args.kwargs["service"] == "db"


@pytest.mark.asyncio
async def test_restore_readiness_probe_uses_the_owning_services_kit():
    """Guards S1 a844a8ad: the release probe runs before the checkpoint opens."""
    from benchflow.sandbox._inert_bootstrap import wait_ready

    env = SimpleNamespace(
        exec_route=AsyncMock(return_value=route()),
        _inspect_container=AsyncMock(
            return_value={"Config": {"Labels": {"com.docker.compose.service": "db"}}}
        ),
        _docker_cli=AsyncMock(return_value=ExecResult(return_code=0)),
    )
    await wait_ready(env, "c" * 64)
    assert f"{KIT}/bin/sh" in env._docker_cli.call_args.args[0]
    assert env.exec_route.call_args.kwargs["service"] == "db"


@pytest.mark.usefixtures("audit_image_tools")
def test_capability_reader_runs_the_kit_python_not_live_python(tmp_path):
    """Guards S3 a844a8ad: a later audit cannot execute a planted Python.

    Since integ's batch 23 the reader runs the Python PT's runtime names
    (``python_command``, the kit's); a python3 on root's PATH never runs."""
    from benchflow.sandbox.firewall_power import _scan_cmd
    from tests.test_firewall_power import Stub

    stub = Stub(tmp_path)
    kit = tmp_path / "rt"
    (kit / "py").mkdir(parents=True)
    py = kit / "py/python3"
    py.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    py.chmod(0o755)
    mark = tmp_path / "planted"
    stub.tool("python3", f"#!/bin/sh\nprintf ran > {shlex.quote(str(mark))}\nexit 1\n")
    out = stub.run(
        _scan_cmd(whole_fs=True, fs_root=str(stub.fs), python_command=str(py))
    )
    assert "caps=python3" in out, out
    assert not mark.exists()


@pytest.mark.asyncio
async def test_root_credential_cleanup_does_not_replace_the_route_path():
    """Guards ACP a844a8ad: root cleanup retains PT's sealed tools."""
    from benchflow.agents.credentials import remove_staged_credentials

    env = SimpleNamespace(exec=AsyncMock(return_value=ExecResult(return_code=0)))
    await remove_staged_credentials(env, None, ["/tmp/login/auth.json"])
    assert all("export PATH=" not in c.args[0] for c in env.exec.call_args_list)


@pytest.mark.asyncio
async def test_inert_image_values_are_not_promoted_to_root_overrides():
    """Guards b165/S1: restored image hooks still pass PT's provenance check."""
    from benchflow.sandbox.docker import DockerSandbox

    env = object.__new__(DockerSandbox)
    env.default_user = None
    env._persistent_env = {}
    env._main_container = "c" * 64
    env._bootstrap_configs = {"main": {"Env": ["LD_PRELOAD=/app/marker.so"]}}
    selected = route()
    selected.pin = None
    selected.task_root = True
    selected.root = True
    selected.explicit_root = True
    selected.root_path = "/usr/bin:/bin"
    env.exec_route = AsyncMock(return_value=selected)
    env._run_docker_container_command = AsyncMock(
        return_value=ExecResult(return_code=0)
    )
    await env.exec("printf safe", user="root")
    # Docker's exec carries the env on stdin.
    sent = env._run_docker_container_command.call_args.kwargs["stdin_data"].decode()
    assert "export _BF_IMAGE_LD_PRELOAD=/app/marker.so\n" in sent
    assert "export LD_PRELOAD=" not in sent


@pytest.mark.asyncio
async def test_firewall_audit_keeps_the_kit_inspectors_and_route():
    """Guards the S3 a844a8ad merge: no live inspectors replace sealed ones."""
    from benchflow.sandbox import firewall_power
    from tests._receipt_fakes import receipt_aware

    commands = []

    class Env:
        supports_system_shell = True
        exec_route = AsyncMock(return_value=route())

        @receipt_aware
        async def exec(self, command, **kwargs):
            commands.append(command)
            return ExecResult(return_code=0, stdout="audit-ok")

    out, failed = await firewall_power._exec(Env(), "printf audit-ok", 10)
    assert not failed and out == "audit-ok"
    assert "/bin/ls" not in commands[0] and "/usr/bin/readlink" not in commands[0]
    assert "PATH=/usr/sbin:/usr/bin:/sbin:/bin" not in commands[0]


@pytest.mark.asyncio
async def test_snapshot_release_keeps_protected_pid1_and_inert_loader_values(
    monkeypatch,
):
    """Guards the S1 a844a8ad merge: release cannot load an image hook as root."""
    import logging

    from benchflow.sandbox import _inert_bootstrap
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.protocol import SandboxImage

    env = object.__new__(DockerSandbox)
    env.default_user = None
    env._persistent_env = {}
    env._main_container = None
    env.logger = logging.getLogger("b165")
    env.task_env_config = SimpleNamespace(network_mode=None)
    env._snapshot_credentials = {}
    env._compose_project = "b165"
    env._main_container_id = AsyncMock(return_value="c" * 64)
    env._inspect_container = AsyncMock(
        return_value={
            "Config": {"Env": ["LD_PRELOAD=/app/marker.so"]},
            "HostConfig": {"NetworkMode": "none"},
            "Mounts": [],
        }
    )
    env._remove_replaced_container = AsyncMock()
    env._snapshot_image_bind_baseline = AsyncMock(return_value={"main": {}})
    env._verify_replacement_bind_targets = AsyncMock()
    env._docker_cli = AsyncMock(return_value=ExecResult(return_code=0, stdout="d" * 64))
    monkeypatch.setattr(_inert_bootstrap, "wait_ready", AsyncMock())
    await env.restore(
        SandboxImage(
            provider="docker", ref="fixture-snapshot", meta={"benchflow_kit": "1"}
        )
    )
    argv = next(
        call.args[0]
        for call in env._docker_cli.call_args_list
        if call.args[0][0] == "run"
    )
    assert argv[argv.index("--entrypoint") + 1] == f"{KIT}/bin/sh"
    assert argv[-2:] == [f"{KIT}/bin/sleep", "infinity"]
    assert "LD_PRELOAD=/app/marker.so" not in argv
    assert "_BF_IMAGE_LD_PRELOAD=/app/marker.so" in argv
    env._verify_replacement_bind_targets.assert_awaited_once()
    assert ["kill", "--signal", "USR1", "d" * 64] in [
        c.args[0] for c in env._docker_cli.call_args_list
    ]


@pytest.mark.asyncio
async def test_script_watchdog_does_not_start_a_second_live_root_bash():
    """Guards DT2 a844a8ad: the watchdog's nested root shell stays on the kit."""
    from tests.test_script_agent import FakeShell, run_fake

    env = FakeShell(marker="0")
    env.exec_route = AsyncMock(return_value=route())
    result, _ = await run_fake(env)
    assert result["return_code"] == 0
    command, _ = env.runner()
    assert f"timeout -s TERM -k 1 30 {KIT}/bin/bash " in command


@pytest.mark.asyncio
async def test_native_healthcheck_does_not_request_a_root_exec_route():
    """Guards b165: a shell-less peer's daemon health needs no runtime kit."""
    from benchflow.sandbox._inert_bootstrap import wait_healthy

    env = SimpleNamespace(
        exec_route=AsyncMock(side_effect=AssertionError("native peer has no shell")),
        _inspect_container=AsyncMock(
            return_value={"State": {"Running": True, "Health": {"Status": "healthy"}}}
        ),
        _docker_cli=AsyncMock(),
    )
    await wait_healthy(
        env,
        "c" * 64,
        {"Healthcheck": {"Test": ["CMD", "/native", "--health"]}},
        native=True,
    )
    env.exec_route.assert_not_awaited()
    env._docker_cli.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_healthcheck_parent_uses_the_owning_services_kit():
    """Guards S1 a844a8ad: the healthcheck's parent is root plumbing."""
    from benchflow.sandbox._inert_bootstrap import wait_healthy

    env = SimpleNamespace(
        exec_route=AsyncMock(return_value=route()),
        _docker_cli=AsyncMock(return_value=ExecResult(return_code=0)),
    )
    await wait_healthy(
        env,
        "c" * 64,
        {
            "Labels": {"com.docker.compose.service": "db"},
            "Healthcheck": {"Test": ["CMD", "/task-health"]},
        },
    )
    argv = env._docker_cli.call_args.args[0]
    assert f"{KIT}/bin/sh" in argv
    assert f"{KIT}/bin/env -- /task-health" in argv[-1]
    assert env.exec_route.call_args.kwargs["service"] == "db"
