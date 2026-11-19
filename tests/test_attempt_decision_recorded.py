"""The retry loop records which attempt it retried and which it kept.

Guards the audit fix on branch ``audit-final-attempt-1-marker``
(finding JOBS-1, the lead's ruling of 2026-11-13): no file said which
attempt of a retried trial was its result, so each reader worked it out by its
own rule. The run's retry decision is now written into each attempt's folder
as ``attempt-decision.json``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from benchflow._utils.attempt_decision import (
    DECISION_FILE,
    FINAL,
    RETRIED,
    read_decision,
    record_decision,
)
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.models import RolloutResult

IDLE = "Agent idle for 600s (no output); killed"
SANDBOX = "sandbox startup failed: docker compose up exited 1"


def _evaluation(tmp_path: Path, results: list[RolloutResult]) -> Evaluation:
    """An Evaluation whose trials are *results*, one per attempt."""
    tasks = tmp_path / "tasks"
    (tasks / "t1").mkdir(parents=True)
    evaluation = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        job_name="job",
        config=EvaluationConfig(agent="oracle", retry=RetryConfig(min_wait_sec=0)),
    )
    queue = list(results)

    async def run_single(task_dir, cfg):
        return queue.pop(0)

    async def no_sweep():
        return None

    evaluation._run_single_task = run_single
    evaluation._sweep_docker = no_sweep
    return evaluation


def _attempt(tmp_path: Path, name: str, *, error, rewards, folder=True):
    trial = tmp_path / "jobs" / "job" / name
    if folder:
        trial.mkdir(parents=True)
    return RolloutResult(
        task_name="t1",
        rollout_name=name,
        error=error,
        rewards=rewards,
        rollout_dir=trial if folder else None,
    )


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    async def instantly(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", instantly)


def test_a_retried_attempt_and_the_kept_one_are_recorded(tmp_path):
    first = _attempt(tmp_path, "t1__aaaa0001", error=IDLE, rewards={"reward": 1.0})
    second = _attempt(tmp_path, "t1__bbbb0002", error=None, rewards={"reward": 0.0})
    evaluation = _evaluation(tmp_path, [first, second])

    result = asyncio.run(evaluation._run_task(tmp_path / "tasks" / "t1"))

    assert result is second
    retried = read_decision(first.rollout_dir)
    kept = read_decision(second.rollout_dir)
    assert (retried["decision"], retried["attempt"], retried["sequence"]) == (
        RETRIED,
        1,
        1,
    )
    assert (kept["decision"], kept["attempt"], kept["sequence"]) == (FINAL, 2, 2)
    assert retried["task"] == kept["task"] == "t1"


def test_a_trial_that_ran_once_is_recorded_as_kept(tmp_path):
    only = _attempt(tmp_path, "t1__aaaa0001", error=None, rewards={"reward": 1.0})
    evaluation = _evaluation(tmp_path, [only])

    asyncio.run(evaluation._run_task(tmp_path / "tasks" / "t1"))

    assert read_decision(only.rollout_dir)["decision"] == FINAL


def test_the_last_attempt_is_kept_when_retries_run_out(tmp_path):
    attempts = [
        _attempt(tmp_path, f"t1__0000000{n}", error=SANDBOX, rewards=None)
        for n in (1, 2, 3)
    ]
    evaluation = _evaluation(tmp_path, list(attempts))

    result = asyncio.run(evaluation._run_task(tmp_path / "tasks" / "t1"))

    assert result is attempts[2]
    assert [read_decision(a.rollout_dir)["decision"] for a in attempts] == [
        RETRIED,
        RETRIED,
        FINAL,
    ]


def test_a_kept_attempt_with_no_folder_is_recorded_when_its_folder_is_made(tmp_path):
    first = _attempt(tmp_path, "t1__aaaa0001", error=IDLE, rewards={"reward": 1.0})
    second = _attempt(
        tmp_path,
        "t1__bbbb0002",
        error="ValueError: bad task",
        rewards=None,
        folder=False,
    )
    evaluation = _evaluation(tmp_path, [first, second])
    job = tmp_path / "jobs" / "job"

    asyncio.run(evaluation._run_task(tmp_path / "tasks" / "t1"))

    assert read_decision(first.rollout_dir)["decision"] == RETRIED
    assert not (job / "t1__bbbb0002").exists()

    payload = {"task_name": "t1", "rollout_name": "t1__bbbb0002", "error": second.error}
    evaluation._record_trial_without_folder(job, "t1", payload)

    kept = read_decision(job / "t1__bbbb0002")
    assert (kept["decision"], kept["attempt"], kept["sequence"]) == (FINAL, 2, 2)


def test_a_later_decision_for_the_task_gets_a_higher_sequence(tmp_path):
    job = tmp_path / "jobs" / "job"
    for name in ("t1__a", "t1__b", "t10__a"):
        (job / name).mkdir(parents=True)

    record_decision(job / "t1__a", FINAL, task="t1", attempt=1)
    record_decision(job / "t10__a", FINAL, task="t10", attempt=1)
    record_decision(job / "t1__b", FINAL, task="t1", attempt=1)

    assert read_decision(job / "t1__a")["sequence"] == 1
    assert read_decision(job / "t10__a")["sequence"] == 1
    assert read_decision(job / "t1__b")["sequence"] == 2


@pytest.mark.parametrize(
    "text", ["", "{", "[]", '{"decision": "maybe", "sequence": 1}']
)
def test_an_unreadable_record_is_no_decision(tmp_path, text):
    (tmp_path / DECISION_FILE).write_text(text)

    assert read_decision(tmp_path) is None


@pytest.mark.parametrize(
    "failure", [OSError("disk full"), ValueError("bad"), RuntimeError("boom")]
)
def test_a_record_that_cannot_be_written_never_fails_the_trial(
    tmp_path, monkeypatch, failure
):
    """Review of #137: the recording is guarded where the run calls it, for a
    kept attempt with a folder and for one whose folder is made afterwards."""
    import benchflow.evaluation as evaluation_module

    def refuse(*args, **kwargs):
        raise failure

    monkeypatch.setattr(evaluation_module, "record_decision", refuse)
    first = _attempt(tmp_path, "t1__aaaa0001", error=IDLE, rewards={"reward": 1.0})
    second = _attempt(
        tmp_path,
        "t1__bbbb0002",
        error="ValueError: bad task",
        rewards=None,
        folder=False,
    )
    evaluation = _evaluation(tmp_path, [first, second])
    job = tmp_path / "jobs" / "job"

    result = asyncio.run(evaluation._run_task(tmp_path / "tasks" / "t1"))
    payload = {"task_name": "t1", "rollout_name": "t1__bbbb0002", "error": second.error}
    evaluation._record_trial_without_folder(job, "t1", payload)

    assert result is second
    assert (job / "t1__bbbb0002" / "result.json").is_file()
    assert read_decision(first.rollout_dir) is None
    assert read_decision(job / "t1__bbbb0002") is None
