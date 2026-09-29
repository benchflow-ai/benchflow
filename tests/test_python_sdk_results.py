"""Typed results for Python SDK callers.

A first-time user of ``bf.run()`` got a ``RolloutResult`` with no scalar
reward, no pass flag and no path to its artifacts, while the Agent +
Environment form returned a ``RuntimeResult`` that had all three. A batch run
through ``Evaluation`` returned only counts, so the per-task results were
reachable only through the ``on_result`` callback. These tests pin the
additions that close both gaps.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.models import RolloutResult
from benchflow.rollout._results import _build_rollout_result


class TestRolloutResultConveniences:
    def test_reward_reads_the_canonical_reward_key(self) -> None:
        assert RolloutResult(task_name="t", rewards={"reward": 0.5}).reward == 0.5

    def test_reward_is_none_without_rewards_or_key(self) -> None:
        assert RolloutResult(task_name="t").reward is None
        assert RolloutResult(task_name="t", rewards={"exact": 1.0}).reward is None

    def test_passed_matches_the_score_outcome(self) -> None:
        assert RolloutResult(task_name="t", rewards={"reward": 1.0}).passed is True
        assert RolloutResult(task_name="t", rewards={"reward": 0.0}).passed is False
        errored = RolloutResult(task_name="t", error="boom")
        assert errored.passed is False

    def test_rollout_dir_defaults_to_none(self) -> None:
        assert RolloutResult(task_name="t").rollout_dir is None


def _write_rollout(tmp_path: Path) -> tuple[Path, RolloutResult]:
    rollout_dir = tmp_path / "job" / "hello__abcd1234"
    rollout_dir.mkdir(parents=True)
    trajectory = [
        {"type": "user_message", "text": "hi"},
        {"type": "tool_call", "title": "Write hello.txt", "kind": "edit"},
        {"type": "agent_message", "text": "done"},
    ]
    result = _build_rollout_result(
        rollout_dir,
        task_name="hello",
        rollout_name="hello__abcd1234",
        agent="claude-agent-acp",
        agent_name="claude",
        model="claude-haiku-4-5",
        n_tool_calls=1,
        prompts=["hi"],
        error=None,
        verifier_error=None,
        trajectory=trajectory,
        partial_trajectory=False,
        trajectory_source="acp",
        rewards={"reward": 1.0},
        started_at=datetime(2026, 1, 1, 12, 0, 0),
        timing={"agent": 3.0},
        n_input_tokens=10,
        n_output_tokens=20,
        total_tokens=30,
    )
    return rollout_dir, result


class TestRolloutResultOnDisk:
    def test_build_sets_rollout_dir(self, tmp_path: Path) -> None:
        rollout_dir, result = _write_rollout(tmp_path)
        assert result.rollout_dir == rollout_dir

    def test_load_round_trips_result_json_and_trajectory(self, tmp_path: Path) -> None:
        rollout_dir, built = _write_rollout(tmp_path)
        loaded = RolloutResult.load(rollout_dir)
        assert loaded.task_name == "hello"
        assert loaded.rollout_name == "hello__abcd1234"
        assert loaded.rewards == {"reward": 1.0}
        assert loaded.reward == 1.0
        assert loaded.passed is True
        assert loaded.agent == "claude-agent-acp"
        assert loaded.model == "claude-haiku-4-5"
        assert loaded.n_tool_calls == 1
        assert loaded.n_prompts == 1
        assert loaded.n_input_tokens == 10
        assert loaded.total_tokens == 30
        assert loaded.trajectory_source == "acp"
        assert loaded.started_at == built.started_at
        assert loaded.finished_at == built.finished_at
        assert loaded.rollout_dir == rollout_dir
        assert [e["type"] for e in loaded.trajectory] == [
            "user_message",
            "tool_call",
            "agent_message",
        ]

    def test_load_accepts_the_result_json_path(self, tmp_path: Path) -> None:
        rollout_dir, _ = _write_rollout(tmp_path)
        assert RolloutResult.load(rollout_dir / "result.json").task_name == "hello"

    def test_load_names_the_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"result\.json"):
            RolloutResult.load(tmp_path)

    def test_from_dict_ignores_unknown_keys(self) -> None:
        loaded = RolloutResult.from_dict(
            {"task_name": "t", "rewards": {"reward": 0.0}, "sandbox_id": "x"}
        )
        assert loaded.task_name == "t"
        assert loaded.passed is False


def _make_job(tmp_path: Path, n_tasks: int) -> Evaluation:
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    for i in range(n_tasks):
        (tasks_dir / f"task-{i}").mkdir()
        (tasks_dir / f"task-{i}" / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    cfg = EvaluationConfig(concurrency=1, retry=RetryConfig(max_retries=0))
    return Evaluation(tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=cfg)


class TestEvaluationResultCarriesResults:
    @pytest.mark.asyncio
    async def test_results_and_job_dir(self, tmp_path: Path) -> None:
        job = _make_job(tmp_path, n_tasks=2)
        outcomes = {
            "task-0": RolloutResult(task_name="task-0", rewards={"reward": 1.0}),
            "task-1": RolloutResult(task_name="task-1", error="agent crashed"),
        }

        async def fake_run(task_path, _cfg):
            return outcomes[task_path.name]

        job._run_single_task = AsyncMock(side_effect=fake_run)
        result = await job.run()

        assert result.job_dir == tmp_path / "jobs" / job._job_name
        assert set(result.results) == {"task-0", "task-1"}
        assert result.results["task-0"] is outcomes["task-0"]
        assert result.results["task-0"].passed is True
        assert result.results["task-1"].error == "agent crashed"

    @pytest.mark.asyncio
    async def test_resumed_tasks_are_loaded_from_disk(self, tmp_path: Path) -> None:
        job = _make_job(tmp_path, n_tasks=2)
        done = job._jobs_dir / job._job_name / "task-0__11111111"
        done.mkdir(parents=True)
        (done / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "task-0",
                    "rollout_name": "task-0__11111111",
                    "rewards": {"reward": 1.0},
                    "error": None,
                    "verifier_error": None,
                    "started_at": "2026-01-01 12:00:00",
                    "finished_at": "2026-01-01 12:01:00",
                }
            )
        )

        async def fake_run(task_path, _cfg):
            return RolloutResult(task_name=task_path.name, rewards={"reward": 0.0})

        job._run_single_task = AsyncMock(side_effect=fake_run)
        result = await job.run()

        assert job._run_single_task.await_count == 1
        resumed = result.results["task-0"]
        assert isinstance(resumed, RolloutResult)
        assert resumed.reward == 1.0
        assert resumed.rollout_name == "task-0__11111111"
        assert result.results["task-1"].passed is False


class TestRolloutAttachesItsDirectory:
    @pytest.mark.asyncio
    async def test_results_not_built_from_artifacts_still_carry_rollout_dir(
        self, tmp_path: Path
    ) -> None:
        """Review, recovery and deadline results are constructed directly, not
        through ``_build_rollout_result``; the rollout attaches its directory
        on the way out so ``result.rollout_dir`` is set on every path."""
        from benchflow.rollout import Rollout, RolloutConfig

        task = Path(__file__).parent / "examples" / "hello-world-task"
        rollout = Rollout(RolloutConfig(task_path=task, agent="oracle"))
        rollout._rollout_dir = tmp_path
        direct = RolloutResult(task_name="hello", rollout_name="hello__x")
        rollout._finish_scoring_locked = AsyncMock(return_value=direct)

        result = await rollout._finish_scoring(direct)

        assert result.rollout_dir == tmp_path


def test_evaluation_result_repr_leaves_out_the_config() -> None:
    """``print(job)`` dumped the whole EvaluationConfig, retry policy included,
    burying the counts a caller printed it for."""
    from benchflow.evaluation import EvaluationResult

    text = repr(EvaluationResult(job_name="j", config=EvaluationConfig(), total=2))
    assert "RetryConfig" not in text
    assert "total=2" in text
