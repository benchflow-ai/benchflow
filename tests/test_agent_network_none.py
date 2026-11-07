"""``[agent] network = "none"`` holds the agent offline in an online sandbox.

task.md's ``[agent] network = "none"`` under an online ``[sandbox]`` (and
task.toml's ``[agent] network_mode = "no-network"``) was accepted with the plan
line "the agent's phase runs offline in an online sandbox", and nothing at run
time read it: the agent had the sandbox's full network (DT2-F2, the
``net-ip-none`` probe: a forbidden address answered under Python, refused
under the Rust bench). The agent's phase now takes the offline path the
sandbox's own ``no-network`` takes for a model agent: model calls go through
the sandbox-local proxy, the agent user's other traffic is refused by its uid
firewall, and web tools are switched off.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.rollout import Rollout, RolloutConfig, _task_disallows_internet
from benchflow.task import RolloutPaths
from benchflow.task.config import SandboxConfig


def _task(*, agent_mode: str | None, allow_internet: bool = True) -> SimpleNamespace:
    sandbox = SandboxConfig(
        network_mode="no-network" if not allow_internet else "public",
        blocked_urls=[],
        blocked_hosts=[],
        allowed_hosts=[],
        allow_internet=allow_internet,
        skills_dir=None,
        docker_image=None,
        workdir=None,
        env={},
        setup_commands=[],
    )
    return SimpleNamespace(
        name="agent-offline",
        config=SimpleNamespace(
            sandbox=sandbox,
            verifier=SimpleNamespace(env={}),
            agent=SimpleNamespace(
                prompt_prefix=None, timeout_sec=60, network_mode=agent_mode
            ),
        ),
    )


def _fake_sandbox() -> MagicMock:
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
    env.stop = AsyncMock()
    env.download_file = AsyncMock()
    return env


def _fake_acp_connection() -> tuple[Any, Any, Any, str]:
    client = MagicMock()
    client.close = AsyncMock()
    client.session = None
    session = MagicMock()
    session.latest_usage_totals.return_value = None
    return client, session, MagicMock(), "agent"


def _fake_planes(env: Any) -> MagicMock:
    planes = MagicMock()
    planes.extract_usage.return_value = {}
    planes.resolve_locked_paths.return_value = []
    planes.resolve_agent_env.side_effect = lambda _agent, _model, agent_env: dict(
        agent_env or {}
    )
    planes.agent_launch.side_effect = (
        lambda agent, *, disallow_web_tools, disallow_hosted_search=False: agent
    )
    planes.create_environment.return_value = env
    planes.ensure_litellm_runtime = AsyncMock(
        side_effect=lambda **kwargs: (kwargs["agent_env"], None)
    )
    planes.start_egress_denylist = AsyncMock()
    planes.stop_egress_denylist = AsyncMock()
    planes.stop_provider_runtime = AsyncMock()
    planes.install_agent = AsyncMock(return_value=MagicMock())
    planes.write_credential_files = AsyncMock()
    planes.upload_subscription_auth = AsyncMock()
    planes.apply_web_tool_policy = AsyncMock()
    planes.block_cloud_metadata = AsyncMock(
        return_value={"state": "on", "why": "fixture metadata block proved"}
    )
    planes.connect_acp = AsyncMock(side_effect=lambda **k: _fake_acp_connection())
    planes.connect_session_factory = AsyncMock(
        side_effect=lambda **k: (None, _fake_acp_connection()[1], None, "agent")
    )
    return planes


def _rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    task: Any,
    planes: Any,
    sandbox_user: str | None = "agent",
) -> Rollout:
    task_dir = tmp_path / "task"
    task_dir.mkdir(exist_ok=True)
    (task_dir / "instruction.md").write_text("Solve it.\n")
    (task_dir / "task.toml").write_text(
        '[task]\nname = "agent-offline"\n[agent]\nnetwork_mode = "'
        + (task.config.agent.network_mode or "open")
        + '"\n'
    )
    rollout_dir = tmp_path / "rollout"

    def fake_init_rollout(task_path, job_name, rollout_name, jobs_dir):
        for subdir in ("agent", "verifier", "artifacts", "trajectory"):
            (rollout_dir / subdir).mkdir(parents=True, exist_ok=True)
        return (
            task,
            rollout_dir,
            RolloutPaths(rollout_dir=rollout_dir),
            datetime.now(),
            "job",
            "rollout",
        )

    monkeypatch.setattr("benchflow.rollout._init_rollout", fake_init_rollout)
    # This transport-policy unit test has no real container to audit. Keep
    # the production audit call/record path, with a successful audit response.
    monkeypatch.setattr(
        "benchflow.rollout.check_firewall_power",
        AsyncMock(return_value=({"fixture": "admitted"}, None)),
    )
    cfg = RolloutConfig(
        task_path=task_dir,
        agent="claude-agent-acp",
        model="test-model",
        sandbox_user=sandbox_user,
        planes=planes,
        task_digest="digest",
    )
    return Rollout(cfg)


def test_an_offline_agent_in_an_online_sandbox_disallows_internet():
    assert _task_disallows_internet(_task(agent_mode="no-network")) is True
    assert _task_disallows_internet(_task(agent_mode=None)) is False
    assert _task_disallows_internet(_task(agent_mode="allowlist")) is False


def test_task_toml_agent_no_network_is_read(tmp_path):
    from benchflow.task import Task

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        '[task]\nname = "test/agent-offline"\n\n[agent]\nnetwork_mode = "no-network"\n'
    )
    (task_dir / "instruction.md").write_text("Test instruction.")

    task = Task(task_dir)
    assert task.config.sandbox.allow_internet is not False
    assert _task_disallows_internet(task) is True


@pytest.mark.asyncio
async def test_the_agent_connects_offline(tmp_path, monkeypatch):
    env = _fake_sandbox()
    planes = _fake_planes(env)
    rollout = _rollout(
        tmp_path, monkeypatch, task=_task(agent_mode="no-network"), planes=planes
    )

    await rollout.setup()
    await rollout.connect()

    assert rollout._disallow_web_tools is True
    # The model goes through the sandbox-local proxy, so the uid firewall
    # (armed by the ACP runtime from this marker) can refuse everything else.
    for call in planes.ensure_litellm_runtime.await_args_list:
        assert call.kwargs["force_sandbox_local"] is True
    acp_env = planes.connect_acp.await_args.kwargs["agent_env"]
    assert acp_env["BENCHFLOW_DISALLOW_WEB_TOOLS"] == "1"
    # The sandbox itself stays online: installs and the verifier keep it.
    assert planes.create_environment.call_args.kwargs.get("preserve_agent_network") in (
        True,
        None,
    )


@pytest.mark.asyncio
async def test_an_online_agent_is_unchanged(tmp_path, monkeypatch):
    env = _fake_sandbox()
    planes = _fake_planes(env)
    rollout = _rollout(
        tmp_path, monkeypatch, task=_task(agent_mode=None), planes=planes
    )

    await rollout.setup()
    await rollout.connect()

    assert rollout._disallow_web_tools is False
    acp_env = planes.connect_acp.await_args.kwargs["agent_env"]
    assert "BENCHFLOW_DISALLOW_WEB_TOOLS" not in acp_env


@pytest.mark.asyncio
async def test_an_offline_agent_without_a_sandbox_user_is_refused(
    tmp_path, monkeypatch
):
    env = _fake_sandbox()
    planes = _fake_planes(env)
    rollout = _rollout(
        tmp_path,
        monkeypatch,
        task=_task(agent_mode="no-network"),
        planes=planes,
        sandbox_user=None,
    )

    with pytest.raises(ValueError, match="requires a sandbox_user"):
        await rollout.setup()
    planes.create_environment.assert_not_called()
