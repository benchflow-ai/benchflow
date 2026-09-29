"""No-web agent runs on Docker get what their egress enforcement needs.

Native Claude OAuth on a plain ``network_mode: no-network`` Docker task
failed before any model call. ``_create_sandbox_environment`` opens the container for
LLM agents and BenchFlow enforces no-web at the agent layer with an iptables
sandbox-user firewall, but the Docker backend added its NET_ADMIN overlay only
for ``network_mode: denylist`` (f396c355), so iptables was refused. The
image also lacked ``python3``, and the no-web setup reported only
``python3: not found``.
"""

import subprocess
from pathlib import Path

import pytest

from benchflow.agents.registry import AGENTS
from benchflow.sandbox._compose import COMPOSE_NET_ADMIN_PATH, COMPOSE_NO_NETWORK_PATH
from benchflow.sandbox.egress_denylist import _setup_cmd
from benchflow.sandbox.setup import _create_sandbox_environment
from benchflow.task import RolloutPaths, Task


def _task(root: Path, network_mode: str) -> Path:
    task_dir = root / f"task-{network_mode}"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Do it.\n")
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n'
        "[agent]\ntimeout_sec = 1\n"
        "[verifier]\ntimeout_sec = 1\n"
        f'[environment]\nnetwork_mode = "{network_mode}"\n'
    )
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    return task_dir


def _compose_files(tmp_path, network_mode, *, preserve_agent_network):
    task_dir = _task(tmp_path, network_mode)
    sandbox = _create_sandbox_environment(
        "docker",
        Task(task_dir),
        task_dir,
        "rollout",
        RolloutPaths(rollout_dir=tmp_path / f"rollout-{network_mode}"),
        preserve_agent_network=preserve_agent_network,
    )
    return sandbox._docker_compose_paths


def test_docker_grants_net_admin_to_agent_enforced_no_web(tmp_path):
    """Guards the no-web Docker fix against the denylist-only overlay of f396c355.

    A no-network task run by an LLM agent keeps an open container and arms
    the sandbox-user iptables firewall, so ``main`` needs NET_ADMIN, the
    capability the automatic reviewer's own overlay added for the same reason.
    """
    files = _compose_files(tmp_path, "no-network", preserve_agent_network=True)

    assert COMPOSE_NET_ADMIN_PATH in files
    assert COMPOSE_NO_NETWORK_PATH not in files


@pytest.mark.parametrize(
    ("network_mode", "preserve_agent_network", "no_network"),
    [("no-network", False, True), ("public", True, False), ("public", False, False)],
    ids=["oracle-no-network", "agent-public", "oracle-public"],
)
def test_docker_keeps_net_admin_off_without_an_agent_firewall(
    tmp_path, network_mode, preserve_agent_network, no_network
):
    """Guards the no-web Docker fix: NET_ADMIN only where the firewall arms.

    An oracle run on a no-network task keeps Docker's own network block, and
    public tasks arm no firewall; neither gets the capability.
    """
    files = _compose_files(
        tmp_path, network_mode, preserve_agent_network=preserve_agent_network
    )

    assert COMPOSE_NET_ADMIN_PATH not in files
    assert (COMPOSE_NO_NETWORK_PATH in files) is no_network


def test_no_web_claude_setup_names_the_missing_python3(tmp_path):
    """Guards the no-web Docker fix: a missing python3 is named as a requirement.

    The smoke's ubuntu:24.04 image failed with a bare ``python3: not found``
    from the claude-agent-acp no-web settings command.
    """
    result = subprocess.run(
        ["/bin/bash", "-c", AGENTS["claude-agent-acp"].disallow_web_tools_setup_cmd],
        env={"PATH": str(tmp_path), "BENCHFLOW_AGENT_HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 127
    assert "needs python3 in the task image" in result.stderr


def test_native_oauth_proxy_names_its_runtime_for_no_web_runs(tmp_path):
    """Guards the no-web Docker fix: the proxy check does not blame denylist only.

    The same loopback proxy carries native Claude OAuth on no-web tasks, so
    its missing-runtime error must name that use as well.
    """
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "python3").symlink_to("/usr/bin/true")
    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            _setup_cmd(
                runtime_dir=str(tmp_path / "runtime"), ca_dir=str(tmp_path / "ca")
            ),
        ],
        env={"PATH": str(binaries)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 87
    assert "needs openssl in the task image" in result.stderr
    assert "native Claude OAuth" in result.stderr
