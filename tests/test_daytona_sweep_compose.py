"""Guards FF-sw-daytona Compose topology and copy-out at 363d815b."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._compose import COMPOSE_DEFINITION_NAMES
from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDinD
from benchflow.sandbox.lockdown import SandboxDaemonGuardError
from benchflow.task.config import SandboxConfig
from tests.test_daytona_download import _fake_dind_strategy


@pytest.mark.parametrize("filename", COMPOSE_DEFINITION_NAMES)
def test_all_compose_filenames_select_dind(tmp_path, filename):
    """Guard 363d815b: topology discovery and Compose flags agree."""
    (tmp_path / filename).write_text(
        "services:\n  main: {}\n  database:\n    image: dummy\n"
    )
    sandbox = DaytonaSandbox(
        environment_dir=tmp_path,
        environment_name="dummy",
        session_id="dummy",
        rollout_paths=SimpleNamespace(),
        task_env_config=SandboxConfig(docker_image="dummy"),
    )
    assert sandbox._compose_mode
    assert isinstance(sandbox._strategy, _DaytonaDinD)
    assert sandbox._environment_docker_compose_path.name == filename
    sandbox._strategy._use_prebuilt = True
    assert (
        f"{sandbox._strategy._ENVIRONMENT_DIR}/{filename}"
        in sandbox._strategy._compose_file_flags()
    )


@pytest.mark.parametrize(
    "bridge", ["eval0", "test-net", "br0", "x" * 15, "eval;0", "eval'0"]
)
async def test_actual_bridge_is_guarded(bridge, tmp_path):
    """Guard 363d815b: firewall input coverage comes from actual networks."""
    strategy = _DaytonaDinD.__new__(_DaytonaDinD)
    commands = []

    async def execute(command, **kwargs):
        commands.append(command)
        data = [
            {
                "Driver": "bridge",
                "Id": "a" * 64,
                "Options": {"com.docker.network.bridge.name": bridge},
            }
        ]
        return ExecResult(
            return_code=0,
            stdout=json.dumps(data)
            if "network inspect" in command
            else "BF dind-daemon-guard status=0",
            stderr="",
        )

    strategy._vm_exec = execute
    guard = getattr(
        strategy, "_guard_compose_networks", strategy._guard_daemon_from_containers
    )
    await guard()
    assert any("-C INPUT" in command for command in commands)
    # Execute the real generated shell against the existing fake firewall;
    # no kernel firewall or extra capabilities are needed for this proof.
    from tests.test_sdk_lockdown import TestSandboxDaemonGuard

    firewall = TestSandboxDaemonGuard()
    binary_dir = firewall._bin(tmp_path)
    firewall._sh(tmp_path, commands[-1], binary_dir, shell="sh")
    state = firewall._state(tmp_path)
    for family in ("iptables", "ip6tables"):
        rules = state[f"{family}:filter"]["INPUT"]
        for port in (2280, 22220, 22222, 33333):
            assert any(
                rule.startswith(f"-i {bridge} -p tcp --dport {port} ") for rule in rules
            )


@pytest.mark.parametrize(
    "networks",
    [
        [],
        [{"Driver": "macvlan"}],
        [{"Driver": "ipvlan"}],
        [{"Driver": "overlay"}],
        [{"Driver": "bridge"}],
        {"unexpected": "shape"},
    ],
)
async def test_unproven_networks_refuse_startup(networks):
    """Guard 363d815b: unsupported namespaces cannot silently bypass guarding."""
    strategy = _DaytonaDinD.__new__(_DaytonaDinD)

    async def execute(command, **kwargs):
        return ExecResult(
            return_code=0,
            stdout=json.dumps(networks)
            if "network inspect" in command
            else "BF dind-daemon-guard status=0",
            stderr="",
        )

    strategy._vm_exec = execute
    guard = getattr(
        strategy, "_guard_compose_networks", strategy._guard_daemon_from_containers
    )
    with pytest.raises(SandboxDaemonGuardError):
        await guard()


@pytest.mark.parametrize(
    "source", ["/logs/verifier", "/logs/agent", "/artifacts", "/logs/verifier/nested"]
)
@pytest.mark.parametrize("kind", ["file", "directory"])
async def test_copy_out_reads_the_effective_container_mount(tmp_path, source, kind):
    """Guard 363d815b: a valid but stale VM log directory is not evidence."""
    strategy = _fake_dind_strategy()
    strategy._env.logger = logging.getLogger("dummy")
    strategy._compose_download_file = AsyncMock()
    strategy._compose_download_dir = AsyncMock()
    if kind == "file":
        await strategy.download_file(source + "/reward.txt", tmp_path / "reward.txt")
        strategy._compose_download_file.assert_awaited_once()
    else:
        await strategy.download_dir(source, tmp_path)
        strategy._compose_download_dir.assert_awaited_once()
    strategy._env._sdk_download_file.assert_not_awaited()
    strategy._env._sdk_download_dir.assert_not_awaited()
