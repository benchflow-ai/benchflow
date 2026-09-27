"""Verifier infra errors stay retryable for tasks without verifier recovery.

Guards the ``retry_on_verifier_infra`` behaviour against a regression from
solver-evidence preservation: it prefixed every verifier error with
``[solver-preserved]`` and ``RetryConfig`` refuses to retry that prefix, so a
task that never opted into ``workspace_recovery``/``submission_files`` ended as
a final ``verifier recovery unavailable`` error instead of being retried.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow._utils.scoring import VERIFIER_INFRA, classify_verifier_error
from benchflow._utils.task_authoring import task_digest
from benchflow.evaluation import Evaluation, RetryConfig
from benchflow.models import RolloutResult
from benchflow.review.options import ReviewerConfig
from benchflow.review.resume import (
    ReviewResumeError,
    resume_pending_reviews,
    resume_review,
)
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.rollout import _verifier_recovery as recovery
from benchflow.rollout._review import prepare_terminal_result
from benchflow.task import RolloutPaths, Task

CRASH = "verifier crashed: connection reset"


@pytest.fixture
def plain_task(tmp_path):
    """A task that declares neither workspace_recovery nor submission_files."""
    task = tmp_path / "tasks" / "task"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n[verifier]\ntimeout_sec = 60\n')
    (task / "instruction.md").write_text("Produce a file.")
    return task


def _rollout(task, root, verifier):
    rollout = Rollout(RolloutConfig(task_path=task, task_digest=task_digest(task)))
    rollout._task = Task(task)
    rollout._rollout_dir = root
    rollout._rollout_name = "task__a"
    rollout._started_at = datetime.now()
    rollout._rollout_paths = RolloutPaths(rollout_dir=root)
    rollout._env = SimpleNamespace(exec=AsyncMock())
    rollout._planes = SimpleNamespace(
        harden_before_verify=AsyncMock(), verifier=lambda **_: verifier
    )
    rollout._agent_cwd = "/app"
    rollout._trajectory = [{"type": "agent_message", "content": "done"}]
    return rollout


@pytest.mark.asyncio
async def test_crash_without_recovery_contract_is_retried_like_main(
    plain_task, tmp_path, monkeypatch, job_factory
):
    """A transport crash keeps main's text and is retried by the evaluation loop."""
    root = tmp_path / "jobs" / "job" / "task__a"
    root.mkdir(parents=True)

    async def verify():
        raise ConnectionResetError("connection reset")

    rollout = _rollout(plain_task, root, SimpleNamespace(verify=verify))
    monkeypatch.setattr(
        "benchflow.rollout._publish_trajectory_for_verifier", AsyncMock()
    )
    run_recovery = AsyncMock(side_effect=AssertionError("no recovery contract"))
    monkeypatch.setattr(recovery, "recover_verifier", run_recovery)

    await rollout.verify()
    assert rollout._verifier_error == CRASH
    assert not (root / "solver-complete.json").exists()

    def build_result(*, result_filename="result.json"):
        (root / result_filename).write_text(
            json.dumps(
                {
                    "task_name": "task",
                    "rollout_name": "task__a",
                    "rewards": rollout._rewards,
                    "verifier_error": rollout._verifier_error,
                }
            )
        )
        return RolloutResult(
            task_name="task",
            rollout_name="task__a",
            rewards=rollout._rewards,
            verifier_error=rollout._verifier_error,
        )

    rollout._build_result = Mock(side_effect=build_result)
    rollout._phase = "cleaned"
    result = await rollout._finish_scoring(prepare_terminal_result(rollout))

    run_recovery.assert_not_awaited()
    assert result.verifier_error == CRASH
    assert classify_verifier_error(result.verifier_error) == VERIFIER_INFRA
    assert RetryConfig().should_retry_verifier_error(result.verifier_error)
    assert not (root / "verifier-recovery").exists()
    assert not (root / "verification.json").exists()

    job, tasks_dir = job_factory(n_tasks=1, max_retries=1)
    ok = RolloutResult(task_name="task-0", rewards={"reward": 1.0})
    job._run_single_task = AsyncMock(side_effect=[result, ok])
    assert (await job._run_task(tasks_dir / "task-0")).rewards == {"reward": 1.0}
    assert job._run_single_task.await_count == 2


@pytest.mark.asyncio
async def test_resume_reruns_verifier_crash_without_recovery_contract(
    plain_task, tmp_path
):
    """Evaluation resume re-runs the task instead of pinning a recovery verdict."""
    job_dir = tmp_path / "jobs" / "job"
    root = job_dir / "task__a"
    (root / "trajectory").mkdir(parents=True)
    source = {
        "task_name": "task",
        "task_digest": task_digest(plain_task),
        "purpose": "task",
        "rollout_name": "task__a",
        "agent": "opencode",
        "rewards": None,
        "verifier_error": CRASH,
    }
    # A solver.json that an earlier build wrote for any
    # recoverable-looking verifier error; evaluation resume must still skip it.
    (root / "solver.json").write_text(json.dumps(source))
    (root / "result.json").write_text(json.dumps(source))
    (root / "config.json").write_text(
        json.dumps(
            {
                "task_digest": source["task_digest"],
                "verifier_recovery": {
                    "eligible": False,
                    "reason": "task must declare a validated workspace_recovery=true "
                    "or submission_files contract",
                },
            }
        )
    )
    (root / "prompts.json").write_text('["Produce a file."]')
    (root / "trajectory/acp_trajectory.jsonl").write_text("")

    await resume_pending_reviews(
        job_dir,
        tasks_root=plain_task.parent,
        reviewer=ReviewerConfig(),
        task_names={"task"},
    )

    assert json.loads((root / "result.json").read_text())["verifier_error"] == CRASH
    assert not (root / "verifier-recovery").exists()
    assert not (root / "verification.json").exists()
    job = Evaluation(
        tasks_dir=plain_task.parent, jobs_dir=tmp_path / "jobs", job_name="job"
    )
    assert "task" not in job._get_completed_tasks()


def test_terminal_result_writes_no_solver_snapshot_without_recovery_contract(
    plain_task, tmp_path
):
    """Guards the result files against a leak from solver-evidence preservation.

    prepare_terminal_result also wrote solver.json for an ineligible task
    whose verifier error looked recoverable, which evaluation resume then
    visited. solver.json is written only for automatic review.
    """
    root = tmp_path / "jobs" / "job" / "task__a"
    root.mkdir(parents=True)
    rollout = _rollout(plain_task, root, SimpleNamespace())
    rollout._verifier_error = CRASH
    written = []

    def build_result(*, result_filename="result.json"):
        written.append(result_filename)
        return RolloutResult(task_name="task", verifier_error=CRASH)

    rollout._build_result = Mock(side_effect=build_result)

    assert prepare_terminal_result(rollout).verifier_error == CRASH
    assert written == ["result.json"]


@pytest.mark.asyncio
async def test_resume_leaves_ineligible_trial_as_main_would(plain_task, tmp_path):
    """Guards the resume behaviour against a leak from solver-evidence preservation.

    A solver.json left for an ineligible task made evaluation resume rewrite
    its result.json and let `bench eval score` finish it. Evaluation resume
    must never touch such a trial, and scoring it directly must
    be refused because the task has no automatic review rubric.
    """
    job_dir = tmp_path / "jobs" / "job"
    root = job_dir / "task__a"
    (root / "trajectory").mkdir(parents=True)
    source = {
        "task_name": "task",
        "task_digest": task_digest(plain_task),
        "purpose": "task",
        "rollout_name": "task__a",
        "agent": "opencode",
        "rewards": None,
        "verifier_error": CRASH,
    }
    (root / "solver.json").write_text(json.dumps(source))
    (root / "result.json").write_text(json.dumps(source))
    (root / "config.json").write_text(
        json.dumps(
            {
                "task_digest": source["task_digest"],
                "verifier_recovery": {
                    "eligible": False,
                    "reason": "task must declare a validated workspace_recovery=true "
                    "or submission_files contract",
                },
            }
        )
    )
    (root / "prompts.json").write_text('["Produce a file."]')
    (root / "trajectory/acp_trajectory.jsonl").write_text("")

    def snapshot():
        # The scoring lock is taken before any decision, as before.
        return {
            str(path.relative_to(root)): path.is_file() and path.read_bytes()
            for path in root.rglob("*")
            if path.name != ".scoring.lock"
        }

    before = snapshot()

    await resume_pending_reviews(
        job_dir,
        tasks_root=plain_task.parent,
        reviewer=ReviewerConfig(),
        task_names={"task"},
    )
    with pytest.raises(ReviewResumeError, match="no automatic review rubric"):
        await resume_review(root, tasks_root=plain_task.parent)

    assert snapshot() == before
