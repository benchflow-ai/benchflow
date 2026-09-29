"""A built-in `nop` control agent.

Users tried `--agent nop` / `agent="nop"` for the empty control run the docs and
bf.load_job already treat as a control: 'nop' was not an agent, so it
started a sandbox, ran `nop` as a raw ACP command and failed after repeated
ACP initialize timeouts ('ACP initialize timed out', pipe_closed), or fell back to a default Claude
model and failed on a missing API key. `nop` is now a scripted agent like the
oracle: it installs nothing, runs nothing, and the task's verifier scores the
untouched workspace, so a task author can prove the verifier fails on it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import benchflow as bf
from benchflow.agents.registry import is_scripted_agent
from benchflow.evaluation import EvaluationConfig, effective_model
from benchflow.rollout import Rollout, RolloutConfig


def _task(task_dir: Path) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
    (task_dir / "instruction.md").write_text("Write hello.txt.\n")
    return task_dir


def test_nop_is_a_scripted_agent_with_no_model() -> None:
    assert is_scripted_agent("nop") and is_scripted_agent("oracle")
    assert not is_scripted_agent("claude-agent-acp")
    assert effective_model("nop", None) is None
    assert EvaluationConfig(agent="nop").model is None


def test_nop_passes_the_sdk_preflight(tmp_path: Path) -> None:
    from benchflow.runtime import check_agent_names

    check_agent_names(["nop"])  # no "did you mean", no raw-command path


@pytest.mark.asyncio
async def test_a_nop_rollout_runs_no_agent_and_is_verified(
    tmp_path: Path, monkeypatch
) -> None:
    import benchflow.rollout as rollout_module

    task = _task(tmp_path / "t")
    rollout = Rollout(RolloutConfig(task_path=task, agent="nop"))
    trial = tmp_path / "trial"
    trial.mkdir()
    rollout._rollout_dir = trial
    rollout._rollout_name = "t__nop"
    rollout._started_at = datetime.now()
    rollout.setup = AsyncMock()
    rollout.start = AsyncMock()
    rollout.install_agent = AsyncMock()
    rollout._env = AsyncMock()
    rollout._agent_cwd = "/app"

    async def no_agent(*_a, **_k):
        raise AssertionError("nop must not run an agent or solve.sh")

    rollout._run_steps = no_agent
    monkeypatch.setattr(rollout_module, "_run_oracle", no_agent)

    async def verify():
        rollout._rewards = {"reward": 0.0}
        rollout._phase = "verified"
        return rollout._rewards

    async def cleanup():
        rollout._phase = "cleaned"

    rollout.verify = AsyncMock(side_effect=verify)
    rollout.cleanup = AsyncMock(side_effect=cleanup)

    result = await rollout.run()

    rollout.verify.assert_awaited_once()
    assert result.error is None and result.rewards == {"reward": 0.0}
    assert result.agent == "nop" and result.agent_name == "nop"
    assert [e["type"] for e in result.trajectory] == ["nop"]
    saved = json.loads((trial / "result.json").read_text())
    assert saved["agent"] == "nop"
    loaded = bf.load_trial(trial)
    assert loaded.control == "empty" and loaded.assessment == "scored"
