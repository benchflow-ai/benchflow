"""One NET_ADMIN rule for the compose backends that run the no-web firewall.

The no-web NET_ADMIN change gave no-web LLM agent runs on Docker the NET_ADMIN capability their
sandbox-user iptables firewall needs. Daytona DinD runs the same compose stack
inside its VM but still added the capability only for ``network_mode:
denylist`` (f396c355), and referenced ``docker-compose-net-admin.yaml`` without
uploading it to the VM. Both backends now ask ``compose_needs_net_admin``.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from benchflow.sandbox._compose import (
    COMPOSE_NET_ADMIN_PATH,
    COMPOSE_NO_NETWORK_PATH,
    compose_needs_net_admin,
)
from benchflow.sandbox.daytona_dind import _DaytonaDinD
from benchflow.sandbox.setup import _create_sandbox_environment
from benchflow.task import RolloutPaths, Task

_DIND_NET_ADMIN = f"{_DaytonaDinD._COMPOSE_DIR}/{COMPOSE_NET_ADMIN_PATH.name}"
_DIND_NO_NETWORK = f"{_DaytonaDinD._COMPOSE_DIR}/{COMPOSE_NO_NETWORK_PATH.name}"

# (task network_mode, preserve_agent_network, firewall armed in ``main``)
_RUNS = [
    ("no-network", True, True),
    ("denylist", True, True),
    ("no-network", False, False),
    ("public", True, False),
    ("public", False, False),
]
_RUN_IDS = [
    "agent-no-network",
    "agent-denylist",
    "oracle-no-network",
    "agent-public",
    "oracle-public",
]


def _task(root: Path, network_mode: str) -> Path:
    task_dir = root / f"task-{network_mode}"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Do it.\n")
    denylist = 'blocked_hosts = ["example.com"]\n' if network_mode == "denylist" else ""
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n'
        "[agent]\ntimeout_sec = 1\n"
        "[verifier]\ntimeout_sec = 1\n"
        f'[environment]\nnetwork_mode = "{network_mode}"\n{denylist}'
    )
    environment = task_dir / "environment"
    (environment / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    # A task compose file selects the Daytona DinD strategy.
    (environment / "docker-compose.yaml").write_text("services:\n  main: {}\n")
    return task_dir


def _sandbox(tmp_path: Path, backend: str, network_mode: str, preserve: bool):
    task_dir = _task(tmp_path / backend, network_mode)
    with (
        patch("benchflow.sandbox.daytona._load_daytona_sdk"),
        patch("benchflow.sandbox._sdk_ops.apply"),
    ):
        return _create_sandbox_environment(
            backend,
            Task(task_dir),
            task_dir,
            "rollout",
            RolloutPaths(rollout_dir=tmp_path / backend / "rollout"),
            preserve_agent_network=preserve,
        )


def _dind_flags(tmp_path: Path, network_mode: str, preserve: bool) -> list[str]:
    sandbox = _sandbox(tmp_path, "daytona", network_mode, preserve)
    assert isinstance(sandbox._strategy, _DaytonaDinD)
    return sandbox._strategy._compose_file_flags()


def test_dind_grants_net_admin_to_agent_enforced_no_web(tmp_path):
    """Guards the Daytona DinD half of the no-web NET_ADMIN change against f396c355's denylist-only rule.

    A no-network task run by an LLM agent keeps an open ``main`` and arms the
    sandbox-user iptables firewall inside it, as on Docker.
    """
    flags = _dind_flags(tmp_path, "no-network", preserve=True)

    assert _DIND_NET_ADMIN in flags
    assert _DIND_NO_NETWORK not in flags


@pytest.mark.parametrize(("network_mode", "preserve", "firewall"), _RUNS, ids=_RUN_IDS)
def test_docker_and_dind_apply_the_same_net_admin_rule(
    tmp_path, network_mode, preserve, firewall
):
    """Guards the shared rule: Docker and DinD cannot drift apart again.

    f396c355 wrote the condition twice and the no-web NET_ADMIN change widened only the Docker
    copy; both backends now defer to ``compose_needs_net_admin``.
    """
    docker = _sandbox(tmp_path, "docker", network_mode, preserve)
    dind_flags = _dind_flags(tmp_path, network_mode, preserve)

    assert compose_needs_net_admin(docker.task_env_config) is firewall
    assert (COMPOSE_NET_ADMIN_PATH in docker._docker_compose_paths) is firewall
    assert (_DIND_NET_ADMIN in dind_flags) is firewall


@pytest.mark.parametrize(("network_mode", "preserve", "firewall"), _RUNS, ids=_RUN_IDS)
def test_dind_uploads_every_benchflow_compose_file_it_references(
    tmp_path, network_mode, preserve, firewall
):
    """Guards f396c355's DinD overlay, which was referenced but never uploaded.

    ``docker compose -f`` on a missing file fails the whole ``compose up``.
    """
    uploaded = {path.name for path in _DaytonaDinD._BENCHFLOW_COMPOSE_FILES}
    flags = _dind_flags(tmp_path, network_mode, preserve)
    referenced = {
        Path(flag).name
        for flag in flags
        if flag.startswith(f"{_DaytonaDinD._COMPOSE_DIR}/")
    }

    assert referenced <= uploaded
    assert COMPOSE_NET_ADMIN_PATH.name in uploaded
