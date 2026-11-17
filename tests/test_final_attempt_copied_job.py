"""A job folder copied without its evaluation.json and summary.json reads as
the job does.

Guards the fix on branch ``audit-final-attempt-2-selector`` after the
re-review of PR #139 (2026-11-13): regrade honours the run's attempt records
in such a folder, but ``bf.load_job`` and the readers that looked for the
job's files (the training signal, the viewer's catalog, ``bench train
stream``) read every rollout there as a sample of its own, so the attempt
the run retried, with its reward 1, counted as a trial. A folder with
neither the job's files nor its records still holds samples, all kept.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import benchflow as bf
from tests.test_final_attempt_rows_and_folderless import IDLE, SANDBOX, _run
from tests.test_stopped_during_retry import _stopped_during_a_retry


def _copied(job_dir: Path, to: Path) -> Path:
    """*job_dir* copied without the job's own files."""
    shutil.copytree(
        job_dir, to, ignore=shutil.ignore_patterns("evaluation.json", "summary.json")
    )
    return to


def _retried_then_kept(tmp_path: Path) -> Path:
    """A copy of a job whose run retried attempt A (idle, reward 1) of t1
    and kept attempt B (its sandbox did not start)."""
    job_dir = _run(
        tmp_path,
        {
            "t1": [
                ("t1__aaaa0001", {"reward": 1.0}, IDLE),
                ("t1__bbbb0002", None, SANDBOX),
            ]
        },
        retries=1,
    )
    return _copied(job_dir, tmp_path / "copied")


def test_load_job_reads_a_copied_job_as_the_job(tmp_path):
    copied = _retried_then_kept(tmp_path)

    job = bf.load_job(copied)

    assert [t.path.name for t in job.trials] == ["t1__bbbb0002"]
    assert [a.path.name for a in job.trials[0].attempts] == [
        "t1__aaaa0001",
        "t1__bbbb0002",
    ]
    assert job.unfinished() == ([], [])
    every = bf.load_job(copied, attempts="all")
    assert [t.path.name for t in every.trials] == ["t1__aaaa0001", "t1__bbbb0002"]


def test_a_copied_job_stopped_during_a_retry_keeps_its_task_interrupted(tmp_path):
    copied = _copied(_stopped_during_a_retry(tmp_path).job_dir, tmp_path / "copied")

    job = bf.load_job(copied)

    assert [t.task_name for t in job.trials] == ["t0"]
    assert job.unfinished() == (["t1"], [])
    assert [(t.task_name, t.cost_usd) for t in job.retried] == [("t1", 0.5)]
    assert "Cost: $0.5000 over 1 rollout(s)" in job.to_markdown()


def test_the_training_signal_replaces_the_retried_attempt_of_a_copy(tmp_path):
    from benchflow.trajectories.training_signal import replaced_attempts

    copied = _retried_then_kept(tmp_path)
    results = [
        (d, json.loads((d / "result.json").read_text()))
        for d in sorted(copied.glob("t1__*"))
    ]

    assert replaced_attempts(results) == {copied / "t1__aaaa0001"}


def test_the_viewer_does_not_count_the_retried_attempt_of_a_copy(tmp_path):
    from benchflow.trajectories.viewer.catalog import _not_counted

    _retried_then_kept(tmp_path)

    assert _not_counted(tmp_path, ["copied/t1__aaaa0001", "copied/t1__bbbb0002"]) == {
        "copied/t1__aaaa0001": "superseded"
    }


def test_the_stream_reads_the_records_of_a_copy(tmp_path):
    from benchflow.trajectories.rollout_stream import _Scanner

    copied = _retried_then_kept(tmp_path)

    records = {
        Path(r.rollout_path).name: r
        for r in _Scanner(copied, None, lambda message: None).scan()
    }

    assert records["t1__aaaa0001"]._retryable is True
    assert records["t1__bbbb0002"]._retryable is False
    assert records["t1__bbbb0002"].replaces == records["t1__aaaa0001"].rollout_path
