"""Regrade reads the attempt the run kept, and a resume's decision supersedes
the earlier one.

Guards the audit fix on branch
``audit-final-attempt-3-regrade-resume`` (findings JOBS-1 and JOBS-8;
the lead's ruling of 2026-11-13). Regrade took every scored attempt of
a task, so it regraded and counted an attempt the run had retried, which no
score uses.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import benchflow as bf
from benchflow._utils.attempt_decision import (
    FINAL,
    RETRIED,
    read_decision,
    record_decision,
)
from benchflow._utils.result_paths import load_task_results
from benchflow._utils.scoring import score_summary_fields
from benchflow.eval_regrade import find_trials
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.models import RolloutResult

IDLE = "Agent idle for 600s (no output); killed"
SANDBOX = "sandbox startup failed: docker compose up exited 1"


def _attempt(job: Path, name: str, *, error, rewards, minute: int) -> RolloutResult:
    """One attempt of task t1, with its folder and result.json."""
    folder = job / name
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(
        json.dumps({"agent": "claude-agent-acp", "task_path": "t1"})
    )
    row = {
        "task_name": "t1",
        "rollout_name": name,
        "agent": "claude-agent-acp",
        "model": "m",
        "rewards": rewards,
        "error": error,
        "started_at": f"2026-11-12T10:{minute:02d}:00",
        "finished_at": f"2026-11-12T10:{minute + 1:02d}:00",
    }
    (folder / "result.json").write_text(json.dumps(row))
    return RolloutResult(
        task_name="t1",
        rollout_name=name,
        error=error,
        rewards=rewards,
        rollout_dir=folder,
    )


def _job(tmp_path: Path) -> Path:
    job = tmp_path / "jobs" / "job"
    job.mkdir(parents=True)
    (job / "evaluation.json").write_text(
        json.dumps(
            {"schema_version": 1, "tasks_dir": "/x", "job_name": "job", "config": {}}
        )
    )
    return job


def _run(tmp_path: Path, attempts: list[RolloutResult]) -> RolloutResult:
    """One run of the job's retry loop for task t1 over *attempts*."""
    tasks = tmp_path / "tasks"
    (tasks / "t1").mkdir(parents=True, exist_ok=True)
    evaluation = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        job_name="job",
        config=EvaluationConfig(agent="oracle", retry=RetryConfig(min_wait_sec=0)),
    )
    queue = list(attempts)

    async def run_single(task_dir, cfg):
        return queue.pop(0)

    async def no_sweep():
        return None

    evaluation._run_single_task = run_single
    evaluation._sweep_docker = no_sweep
    return asyncio.run(evaluation._run_task(tasks / "t1"))


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    async def instantly(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", instantly)


def test_regrade_takes_the_kept_attempt_not_the_retried_one(tmp_path):
    job = _job(tmp_path)
    _attempt(job, "t1__aaaa0001", error=IDLE, rewards={"reward": 0.0}, minute=0)
    _attempt(job, "t1__bbbb0002", error=None, rewards={"reward": 1.0}, minute=5)
    record_decision(job / "t1__aaaa0001", RETRIED, task="t1", attempt=1)
    record_decision(job / "t1__bbbb0002", FINAL, task="t1", attempt=2)

    assert [t.name for t in find_trials(job)] == ["t1__bbbb0002"]


def test_regrade_lists_a_kept_attempt_that_has_no_reward(tmp_path):
    job = _job(tmp_path)
    _attempt(job, "t1__aaaa0001", error=IDLE, rewards={"reward": 1.0}, minute=0)
    _attempt(job, "t1__bbbb0002", error=SANDBOX, rewards=None, minute=5)
    record_decision(job / "t1__aaaa0001", RETRIED, task="t1", attempt=1)
    record_decision(job / "t1__bbbb0002", FINAL, task="t1", attempt=2)

    # The trial is the errored attempt; regrade reports it, not attempt 1.
    assert [t.name for t in find_trials(job)] == ["t1__bbbb0002"]


def test_regrade_reads_a_job_with_no_records_by_the_earlier_rule(tmp_path):
    job = _job(tmp_path)
    _attempt(job, "t1__aaaa0001", error=SANDBOX, rewards=None, minute=0)
    _attempt(job, "t1__bbbb0002", error=None, rewards={"reward": 1.0}, minute=5)

    assert [t.name for t in find_trials(job)] == ["t1__bbbb0002"]


def test_a_resume_that_runs_the_task_again_supersedes_the_earlier_decision(tmp_path):
    job = _job(tmp_path)
    # The first run: attempt 1 goes idle with a reward and is retried,
    # attempt 2 fails at sandbox start, and so does attempt 3, which is kept.
    first = [
        _attempt(job, "t1__aaaa0001", error=IDLE, rewards={"reward": 1.0}, minute=0),
        _attempt(job, "t1__bbbb0002", error=SANDBOX, rewards=None, minute=5),
        _attempt(job, "t1__cccc0003", error=SANDBOX, rewards=None, minute=10),
    ]
    kept = _run(tmp_path, first)

    assert kept.rollout_name == "t1__cccc0003"
    # The trial errored, so a resume finds the task unfinished and runs it.
    assert load_task_results(job) == {}
    before = score_summary_fields(
        load_task_results(job, include_terminal_errors=True).values()
    )
    assert (before["passed"], before["errored"]) == (0, 1)

    # The resume: one more attempt, which passes.
    again = _run(
        tmp_path,
        [_attempt(job, "t1__dddd0004", error=None, rewards={"reward": 1.0}, minute=20)],
    )

    assert again.rollout_name == "t1__dddd0004"
    decisions = {
        name: read_decision(job / name)
        for name in ("t1__aaaa0001", "t1__bbbb0002", "t1__cccc0003", "t1__dddd0004")
    }
    assert [d["decision"] for d in decisions.values()] == [
        RETRIED,
        RETRIED,
        FINAL,
        FINAL,
    ]
    assert [d["sequence"] for d in decisions.values()] == [1, 2, 3, 4]
    # Every reader now takes the resume's attempt.
    assert load_task_results(job)["t1"]["rollout_name"] == "t1__dddd0004"
    after = score_summary_fields(load_task_results(job).values())
    assert (after["passed"], after["errored"]) == (1, 0)
    assert [t.path.name for t in bf.load_job(job).trials] == ["t1__dddd0004"]
    assert [t.name for t in find_trials(job)] == ["t1__dddd0004"]
