"""Guards the CONF-rollout-1 fix against scripted-output loss at 10b183f4."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.rollout import Rollout
from tests.reachable_fixture import directory
from tests.test_acp_final_reply import RecordingEnv


@pytest.mark.parametrize("agent", ["oracle", "nop"])
async def test_terminal_verification_preserves_scripted_output(
    tmp_path, monkeypatch, agent
):
    import benchflow.rollout as rollout_mod

    monkeypatch.setattr(rollout_mod, "capture_terminal_workspace", AsyncMock())
    monkeypatch.setattr(rollout_mod, "collect_rollout_artifacts", AsyncMock())
    monkeypatch.setattr(rollout_mod, "require_safe_branch_world", lambda _r: None)
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.recovery_ineligible_reason",
        lambda _r: "not eligible",
    )
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.mark_solver_complete", lambda _r: None
    )
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "output.txt").write_text("script-owned marker")

    async def grade(*args, **kwargs):
        return (
            {
                "reward": float(
                    (agent_dir / "output.txt").read_text() == "script-owned marker"
                )
            },
            None,
            None,
        )

    monkeypatch.setattr(rollout_mod, "_verify_rollout", AsyncMock(side_effect=grade))
    rollout = Rollout.__new__(Rollout)
    rollout._config = SimpleNamespace(
        primary_agent=agent, sandbox_user=None, task_path=Path("/tmp/task")
    )
    rollout._env = RecordingEnv()
    rollout._task = SimpleNamespace(config=None)
    rollout._trajectory = [{"type": agent}]
    rollout._rollout_dir = tmp_path
    rollout._rollout_paths = SimpleNamespace(agent_dir=directory(agent_dir))
    rollout._timing = {}
    rollout._planes = object()
    rollout._agent_cwd = "/app"
    rollout._diagnostics = MagicMock()
    assert await rollout.verify() == {"reward": 1.0}
