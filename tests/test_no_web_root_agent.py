"""An agent that runs as uid 0 under the no-web policy is held.

Before, ``enforce_agent_egress_firewall`` installed nothing when the run had
no sandbox user, so a root agent (``--sandbox-user none``, and since PERF-A
round 4 ``root`` and ``0`` too) had an open network on a no-web task. Now it
is launched with its own primary group and without NET_ADMIN, NET_RAW,
SETUID or SETGID, and the uid firewall holds uid 0 in that group to
loopback; BenchFlow's own root processes (gid 0) are not held. A
session-factory agent, whose commands run outside that launch, is refused.
The real-Docker checks are in test_cloud_metadata_failclosed_docker.py.
"""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox.lockdown import (
    NO_WEB_ROOT_GID,
    _agent_egress_firewall_cmd,
    build_root_no_web_launch,
    enforce_agent_egress_firewall,
    no_web_root_agent,
)

NO_WEB = {
    "BENCHFLOW_DISALLOW_WEB_TOOLS": "1",
    "BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:4000/v1",
}


def test_the_root_launch_keeps_uid_0_in_its_own_group_without_the_powers():
    launch = build_root_no_web_launch("agent-acp --stdio")

    assert f"--regid={NO_WEB_ROOT_GID} --clear-groups" in launch
    assert "--no-new-privs" in launch
    assert "--bounding-set=-net_admin,-net_raw,-setuid,-setgid" in launch
    assert "--inh-caps=-all --ambient-caps=-all" in launch
    assert "--reuid" not in launch
    assert "agent-acp --stdio" in launch
    # Without a setpriv that can do this, the launch refuses with the reason.
    assert "sandbox startup refused" in launch and "exit 1" in launch
    subprocess.run(["bash", "-n", "-c", launch], check=True)


def test_the_root_rules_match_uid_0_in_the_agents_group_only():
    cmd = _agent_egress_firewall_cmd("root", root_gid=NO_WEB_ROOT_GID)
    owner = f"-m owner --uid-owner 0 --gid-owner {NO_WEB_ROOT_GID}"

    assert f"iptables -I OUTPUT 1 -o lo {owner} -j ACCEPT" in cmd
    assert f"iptables -A OUTPUT {owner} -j REJECT" in cmd
    assert f"ip6tables -A OUTPUT {owner} -j REJECT" in cmd
    # Never a rule for all of uid 0: BenchFlow's own root processes (the
    # in-sandbox model proxy, Daytona's daemon) keep gid 0 and their network.
    assert "--uid-owner 0 -j" not in cmd
    assert "id -u" not in cmd
    subprocess.run(["bash", "-n", "-c", cmd], check=True)


def test_the_non_root_rules_are_unchanged():
    cmd = _agent_egress_firewall_cmd("agent")
    assert "agent_uid=$(id -u agent)" in cmd
    assert '-o lo -m owner --uid-owner "$agent_uid" -j ACCEPT' in cmd
    assert "--gid-owner" not in cmd


@pytest.mark.parametrize(
    ("user", "env", "held"),
    [
        (None, NO_WEB, True),
        (None, {"BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:4000/v1"}, False),
        ("agent", NO_WEB, False),
    ],
)
def test_which_agents_get_the_root_hold(user, env, held):
    assert no_web_root_agent(user, env) is held


@pytest.mark.asyncio
async def test_a_root_agent_under_no_web_gets_the_root_rules():
    """On 5c2d86755 nothing ran: the root agent kept an open network."""
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0))

    await enforce_agent_egress_firewall(env, None, dict(NO_WEB))

    env.exec.assert_awaited_once()
    command = env.exec.await_args.args[0]
    assert f"--uid-owner 0 --gid-owner {NO_WEB_ROOT_GID}" in command
    assert env.exec.await_args.kwargs == {"user": "root", "timeout_sec": 120}


@pytest.mark.asyncio
async def test_a_root_agent_under_no_web_needs_a_loopback_model_route():
    env = MagicMock()
    env.exec = AsyncMock()
    with pytest.raises(RuntimeError, match="loopback provider base URL"):
        await enforce_agent_egress_firewall(
            env,
            None,
            {
                "BENCHFLOW_DISALLOW_WEB_TOOLS": "1",
                "BENCHFLOW_PROVIDER_BASE_URL": "https://api.example.test/v1",
            },
        )
    env.exec.assert_not_called()


@pytest.mark.asyncio
async def test_a_failed_root_firewall_stops_the_trial():
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=86, stdout="", stderr=""))
    with pytest.raises(RuntimeError, match="egress firewall"):
        await enforce_agent_egress_firewall(env, None, dict(NO_WEB))


@pytest.mark.asyncio
async def test_a_root_agent_without_no_web_gets_no_rules():
    env = MagicMock()
    env.exec = AsyncMock()
    await enforce_agent_egress_firewall(
        env, None, {"BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:4000/v1"}
    )
    env.exec.assert_not_called()


class _LaunchObserved(Exception):
    pass


@pytest.mark.asyncio
async def test_a_native_root_turn_under_no_web_uses_the_root_launch(tmp_path):
    from benchflow.native_harness.client import NativeCLIClient
    from benchflow.native_harness.harnesses import native_harness_for

    process = AsyncMock()
    process.start.side_effect = _LaunchObserved()
    process.stderr_tail = ""
    env = MagicMock()
    env.live_process = AsyncMock(return_value=process)
    client = NativeCLIClient(
        env=env,
        harness=native_harness_for("claude-agent-acp"),
        agent="claude-agent-acp",
        launch_env=dict(NO_WEB),
        sandbox_user=None,
        cwd="/tmp/workspace",
        rollout_dir=tmp_path,
    )
    client._kill = AsyncMock()
    with pytest.raises(_LaunchObserved):
        await client.prompt("dummy prompt")
    command = process.start.await_args.kwargs["command"]
    assert f"--regid={NO_WEB_ROOT_GID}" in command
    assert "--bounding-set=-net_admin,-net_raw,-setuid,-setgid" in command


@pytest.mark.asyncio
async def test_a_root_session_factory_agent_under_no_web_is_refused(tmp_path):
    from benchflow.rollout import Role, Rollout, RolloutConfig, Scene
    from benchflow.sandbox.firewall_power import FirewallPowerError

    trial = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task",
            scenes=[
                Scene(roles=[Role(name="agent", agent="claude-agent-acp", model="m")])
            ],
            sandbox_user=None,
        )
    )
    trial._disallow_web_tools = True
    with pytest.raises(FirewallPowerError, match="no-web policy"):
        trial._check_cloud_metadata_protocol("pkg.module:factory")
    # An ACP or native agent, or a non-root user, is not refused here.
    trial._check_cloud_metadata_protocol(None)
    trial._config.sandbox_user = "agent"
    trial._check_cloud_metadata_protocol("pkg.module:factory")
