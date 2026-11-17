"""Every reader of a job takes the attempt the run kept as the trial's result.

Guards the audit fix on branch ``audit-final-attempt-2-selector``
(finding JOBS-1; the lead's ruling of 2026-11-13: the run's retry
decision is final, and an attempt the run retried is never the trial's result).

The case: attempt 1's agent goes idle, the verifier still gives its partial
work reward 1, and the run retries it; attempt 2 fails at sandbox start and is
the attempt the run keeps. The run's own summary counted the trial as errored
(0%). A resume and ``bf.load_job`` preferred "a scored attempt", read attempt 1
and said passed (100%).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import benchflow as bf
from benchflow._utils.attempt_decision import FINAL, RETRIED, record_decision
from benchflow._utils.result_paths import load_task_results
from benchflow._utils.scoring import score_summary_fields
from benchflow.evaluation import RetryConfig

IDLE_WITH_REWARD = {
    "task_name": "t1",
    "rollout_name": "t1__aaaa0001",
    "agent": "claude-agent-acp",
    "model": "m",
    "rewards": {"reward": 1.0},
    "error": "Agent idle for 600s (no output); killed",
    "started_at": "2026-11-12T10:00:00",
    "finished_at": "2026-11-12T10:20:00",
}
SANDBOX_FAILED = {
    "task_name": "t1",
    "rollout_name": "t1__bbbb0002",
    "agent": "claude-agent-acp",
    "model": "m",
    "rewards": None,
    "error": "sandbox startup failed: docker compose up exited 1",
    "started_at": "2026-11-12T10:21:00",
    "finished_at": "2026-11-12T10:22:00",
}


def _job(tmp_path: Path, *attempts: dict, decisions: bool = True) -> Path:
    """A job folder with one trial folder per attempt, oldest first; with
    *decisions*, every attempt but the last is marked retried."""
    job = tmp_path / "job"
    job.mkdir()
    (job / "evaluation.json").write_text(
        json.dumps(
            {"schema_version": 1, "tasks_dir": "/x", "job_name": "job", "config": {}}
        )
    )
    for index, attempt in enumerate(attempts):
        folder = job / attempt["rollout_name"]
        folder.mkdir()
        (folder / "config.json").write_text(
            json.dumps({"agent": "claude-agent-acp", "task_path": "t1"})
        )
        result = folder / "result.json"
        result.write_text(json.dumps(attempt))
        os.utime(result, (1_800_000_000 + index, 1_800_000_000 + index))
        if decisions:
            last = index == len(attempts) - 1
            record_decision(
                folder, FINAL if last else RETRIED, task="t1", attempt=index + 1
            )
    return job


def test_the_case_is_one_the_run_retries():
    retry = RetryConfig()
    assert retry.should_retry(IDLE_WITH_REWARD["error"])
    assert retry.should_retry(SANDBOX_FAILED["error"])
    # What the run's own summary says: it counts the attempt it kept.
    run_summary = score_summary_fields([SANDBOX_FAILED])
    assert (run_summary["passed"], run_summary["errored"]) == (0, 1)


def test_a_resume_never_reuses_the_attempt_the_run_retried(tmp_path):
    job = _job(tmp_path, IDLE_WITH_REWARD, SANDBOX_FAILED)

    # The trial errored with no reward, so the task is not finished: a resume
    # runs it again. It never becomes "finished, passed" through attempt 1.
    assert load_task_results(job) == {}


def test_a_summary_refresh_agrees_with_the_runs_own_summary(tmp_path):
    job = _job(tmp_path, IDLE_WITH_REWARD, SANDBOX_FAILED)

    latest = load_task_results(job, include_terminal_errors=True)

    assert latest["t1"]["rollout_name"] == "t1__bbbb0002"
    refreshed = score_summary_fields(latest.values())
    run_summary = score_summary_fields([SANDBOX_FAILED])
    for count in ("total", "passed", "failed", "errored", "score"):
        assert refreshed[count] == run_summary[count], count
    assert (refreshed["passed"], refreshed["errored"]) == (0, 1)


def test_load_job_reads_the_kept_attempt(tmp_path):
    job = bf.load_job(_job(tmp_path, IDLE_WITH_REWARD, SANDBOX_FAILED))

    assert [t.path.name for t in job.trials] == ["t1__bbbb0002"]
    assert job.denominators().passed == 0
    # The retried attempt is still listed as an attempt of the trial.
    assert [a.path.name for a in job.trials[0].attempts] == [
        "t1__aaaa0001",
        "t1__bbbb0002",
    ]


def test_a_trial_whose_every_attempt_was_retried_has_no_result_yet(tmp_path):
    job = _job(tmp_path, IDLE_WITH_REWARD, decisions=False)
    record_decision(job / "t1__aaaa0001", RETRIED, task="t1", attempt=1)

    # The run stopped before the next attempt finished.
    assert load_task_results(job, include_terminal_errors=True) == {}
    assert bf.load_job(job).trials == []


def test_a_job_with_no_attempt_records_is_read_by_the_earlier_rule(tmp_path):
    job = _job(tmp_path, IDLE_WITH_REWARD, SANDBOX_FAILED, decisions=False)

    assert load_task_results(job)["t1"]["rollout_name"] == "t1__aaaa0001"
    assert [t.path.name for t in bf.load_job(job).trials] == ["t1__aaaa0001"]
