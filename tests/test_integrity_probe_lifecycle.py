"""The integrity pre-turn probe is recorded on every lifecycle that installs an agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import benchflow.integrity.trial as integrity_trial
from benchflow.rollout import Rollout, RolloutConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("integrity", ["audit", "strict"])
async def test_install_agent_records_the_pre_turn_probe_after_lockdown(
    tmp_path, monkeypatch, integrity
):
    """A manually composed lifecycle (setup, start, install_agent, connect,
    execute, verify) gets the same pre-turn probe as the automatic one: it is
    taken at the end of install_agent, after lockdown, before any turn."""
    config = RolloutConfig.from_legacy(
        task_path=tmp_path / "task",
        agent="claude-agent-acp",
        prompts=[None],
        sandbox_user="agent",
    )
    config.integrity = integrity
    rollout = Rollout(config)
    rollout._env = MagicMock()
    rollout._env.exec = AsyncMock(return_value=MagicMock(stdout="/workspace\n"))
    rollout._rollout_dir = tmp_path / "trial"
    rollout._rollout_dir.mkdir()
    rollout._rollout_paths = MagicMock()
    rollout._task = MagicMock()
    # FIX-49 r4: this unnamed-runner fake has a readable tests folder.
    rollout._task.task_dir = tmp_path / "task"
    (rollout._task.task_dir / "tests").mkdir(parents=True)
    rollout._effective_locked = []

    order: list[str] = []
    for name in (
        "install_agent",
        "write_credential_files",
        "upload_subscription_auth",
        "snapshot_build_config",
        "seed_verifier_workspace",
        "deploy_skills",
    ):
        monkeypatch.setattr(rollout._planes, name, AsyncMock(return_value=MagicMock()))
    monkeypatch.setattr(
        rollout._planes, "setup_sandbox_user", AsyncMock(return_value="/home/agent")
    )

    async def lockdown(*args, **kwargs):
        order.append("lockdown")

    async def probe(target, phase):
        order.append(f"probe:{phase}")

    monkeypatch.setattr(rollout._planes, "lockdown_paths", lockdown)
    monkeypatch.setattr(integrity_trial, "record_access_probe", probe)

    await rollout.install_agent()

    assert order == ["lockdown", "probe:pre"]


@pytest.mark.asyncio
async def test_no_probe_is_taken_with_integrity_off(tmp_path, monkeypatch):
    from benchflow.rollout import _probe_integrity_access

    called = AsyncMock()
    monkeypatch.setattr(integrity_trial, "record_access_probe", called)
    stand_in = MagicMock()
    stand_in._config = RolloutConfig.from_legacy(
        task_path=tmp_path / "task", agent="claude-agent-acp", prompts=[None]
    )
    await _probe_integrity_access(stand_in, "pre")
    called.assert_not_awaited()
