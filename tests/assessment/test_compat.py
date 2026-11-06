"""The sidecar never changes what a 0.9 reader reports.

``result.json`` (its ``rewards``, and its own ``assessment`` field, which
withholds a reward while pending), ``regrade``'s trial discovery, the
``benchflow.rollout-stream.v1`` records and the job export must read the same
before and after a full assessment lives beside a rollout. A missing sidecar
is not a pending review, and a label is never a reward.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import benchflow as bf
from benchflow import assessment as a
from benchflow._utils.scoring import assessment_status, extract_reward
from benchflow.eval_regrade import find_trials
from tests.assessment._fixtures import OPERATOR, REVIEWER, make_rollout, world


def _job_with_other_trial(w) -> Path:
    job = w.first.root.parent
    make_rollout(w.base, "other-task__eeee5555", reward=1.0)
    (job / "evaluation.json").write_text("{}")
    (job / "summary.json").write_text(json.dumps({"total": 3}))
    return job


def _readers(job: Path) -> dict[str, Any]:
    """Everything a 0.9 reader reports about the job, as JSON-comparable values."""
    results = {}
    for trial in sorted(job.iterdir()):
        path = trial / "result.json"
        if path.is_file():
            data = json.loads(path.read_text())
            results[trial.name] = {
                "bytes": path.read_bytes().decode(),
                "reward": extract_reward(data),
                "assessment_status": assessment_status(data),
                "trial": bf.load_trial(trial).to_json_dict(include_trajectory=False),
            }
    return {
        "job": bf.load_job(job).to_json_dict(),
        "stream": [
            record.to_json()
            for record in bf.stream_rollouts(job, follow=False, group_size=2)
        ],
        "regrade_trials": [str(p) for p in find_trials(job)],
        "results": results,
    }


def test_a_full_assessment_leaves_every_09_reader_unchanged(tmp_path: Path) -> None:
    w = world(tmp_path)
    job = _job_with_other_trial(w)
    before = _readers(job)
    assert before["results"][w.first.root.name]["assessment_status"] is None

    w.answered("left")
    a.reduce(w.sidecar)
    a.consume(w.sidecar, label_revision=1, consumer_id="ledger", consumer_receipt="row")
    target = a.load(w.sidecar).effective
    a.correct(
        w.sidecar,
        supersedes_event_id=target,
        choice="tie",
        reason="",
        correction_reason="c",
        actor=REVIEWER,
    )
    a.reduce(w.sidecar)
    a.withdraw(
        w.sidecar,
        label_revision=1,
        reason="superseded",
        disposition="queued_compensation",
        actor=OPERATOR,
    )
    assert (w.sidecar / "assessment-label.v1.json").is_file()

    after = _readers(job)
    assert after == before
    for result in after["results"].values():
        assert "assessment" not in json.loads(result["bytes"])


def test_a_sidecar_with_a_pending_review_is_not_a_pending_trial(tmp_path: Path) -> None:
    """Guards the contract's rule that a sidecar never stands for a pending assessment."""
    w = world(tmp_path)
    job = _job_with_other_trial(w)
    before = _readers(job)
    w.create()  # awaiting release: a human review is not even possible yet
    w.release()
    w.assign()  # awaiting the human
    assert _readers(job) == before


def test_the_assessment_package_imports_no_reward_or_run_code() -> None:
    script = (
        "import sys, benchflow.assessment, benchflow.assessment.schema;"
        "print('\\n'.join(sorted(m for m in sys.modules if m.startswith('benchflow'))))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    ).stdout.split()
    outside = [m for m in loaded if not m.startswith("benchflow.assessment")]
    assert outside == ["benchflow", "benchflow.errors"], outside
