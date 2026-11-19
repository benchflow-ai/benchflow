"""summary.json counts a trial this run finished from the same records as a
trial a resume reloads.

Guards the audit fix on branch ``audit-summary-rows``: a fresh
trial's summary row was rebuilt from the in-memory result plus a short list of
fields copied back from result.json, so a record outside that list (here
``verifier_timeout_info``) was missing from the run's own summary.json and
present after a resume. Seen on a live Docker run: a verifier that ran past
its time limit was scored 0 and summary.json said ``grading_timeouts: 0``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from benchflow._utils.evaluation_results import rollout_result_payload
from benchflow._utils.scoring import score_summary_fields
from benchflow.evaluation import Evaluation
from benchflow.models import RolloutResult

PERSISTED = {
    "task_name": "t",
    "rollout_name": "t__1",
    "agent": "script:ok.sh",
    "model": None,
    "rewards": {"reward": 0.0},
    "error": None,
    "verifier_error": None,
    "started_at": "2026-11-13T06:49:42+00:00",
    "finished_at": "2026-11-13T06:50:53+00:00",
    "timing": {"verifier": 36.5},
    "verifier_timeout_info": {
        "timeout_budget_sec": 30.0,
        "elapsed_sec": 36.5,
        "task_name": "t",
        "scored": True,
    },
}


def _fresh_row(tmp_path: Path) -> dict:
    """The row the run builds for a trial it just finished."""
    trial = tmp_path / "jobs" / "job" / "t__1"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps(PERSISTED))
    result = RolloutResult.from_dict(PERSISTED, rollout_dir=trial)
    row = rollout_result_payload(
        result, source_provenance=None, tasks_dir=tmp_path / "tasks", task_name="t"
    )
    host = SimpleNamespace(_jobs_dir=tmp_path / "jobs", _job_name="job")
    Evaluation._enrich_payload_with_persisted_timing(host, row, result)
    return row


def test_a_fresh_row_holds_every_record_of_its_result(tmp_path):
    row = _fresh_row(tmp_path)

    assert set(PERSISTED) <= set(row)
    assert row["verifier_timeout_info"] == PERSISTED["verifier_timeout_info"]
    assert row["started_at"] == PERSISTED["started_at"]


def test_the_in_memory_result_wins_over_the_persisted_one(tmp_path):
    trial = tmp_path / "jobs" / "job" / "t__1"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps(PERSISTED))
    row = {"rewards": {"reward": 1.0}, "error": None}
    host = SimpleNamespace(_jobs_dir=tmp_path / "jobs", _job_name="job")

    Evaluation._enrich_payload_with_persisted_timing(
        host, row, SimpleNamespace(rollout_name="t__1")
    )

    assert row["rewards"] == {"reward": 1.0}
    assert row["agent"] == "script:ok.sh"


def test_a_grading_timeout_is_counted_the_same_fresh_and_resumed(tmp_path):
    fresh = score_summary_fields([_fresh_row(tmp_path)])
    resumed = score_summary_fields([PERSISTED])

    assert fresh["grading_timeouts"] == resumed["grading_timeouts"] == 1
    assert fresh["failed"] == resumed["failed"] == 1
