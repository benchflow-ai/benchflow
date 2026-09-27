"""An unscored trial's results.jsonl row has reward null, not 0.0.

A trial that failed on auth has ``rewards: null`` in result.json and is
unscored in bf.load_job, but the Verifiers-shaped results.jsonl row said
``"reward": 0.0``, so any tool reading that file counted an infrastructure
failure as a real zero. The row already writes null for a non-finite reward.
"""

from __future__ import annotations

import json
from pathlib import Path

import benchflow as bf
from benchflow.trajectories.results import write_rollout_results_jsonl


def _row(rollout_dir: Path, rewards, error=None) -> dict:
    rollout_dir.mkdir(parents=True)
    write_rollout_results_jsonl(
        rollout_dir,
        task_name="hello",
        rollout_name="hello__1",
        agent="claude-agent-acp",
        agent_name="claude-agent-acp",
        model="claude-haiku-4-5",
        n_tool_calls=0,
        prompts=["x"],
        trajectory=[],
        partial_trajectory=False,
        rewards=rewards,
        error=error,
        verifier_error=None,
    )
    return json.loads((rollout_dir / "results.jsonl").read_text())


def test_a_missing_reward_is_null_in_the_row(tmp_path: Path) -> None:
    row = _row(tmp_path / "job" / "hello__1", None, error="Invalid credentials")
    assert row["reward"] is None
    assert _row(tmp_path / "job2" / "hello__1", {"reward": 0.0})["reward"] == 0.0
    assert _row(tmp_path / "job3" / "hello__1", {"other": 1.0})["reward"] is None
    # The loader reads the row back as unscored, the same as result.json.
    trial = bf.load_job(tmp_path / "job" / "hello__1" / "results.jsonl").trials[0]
    assert trial.reward is None and trial.assessment == "unscored"
