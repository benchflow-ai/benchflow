"""``bench train token-coverage`` counts the job's rollouts, not its reviewers.

On a job with automatic rubric review, token-coverage reported one rollout
more per reviewed trial: it found result.json files with ``rglob`` and so
counted the reviewer's own run under ``<trial>/reviews/``. ``bf.load_job``
and ``bench eval inspect`` already skip reviewer runs through
``iter_task_result_paths``; token-coverage now uses the same discovery.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner


def _result(folder: Path, **extra) -> None:
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(
        json.dumps({"task_name": "t", "rewards": {"reward": 1.0}, **extra})
    )


def test_token_coverage_ignores_nested_reviewer_runs(tmp_path: Path) -> None:
    from benchflow.cli.main import app

    job = tmp_path / "job"
    trial = job / "t__abc"
    _result(trial, usage_tracking={"endpoint_kind": "agent_native"})
    _result(
        trial / "reviews" / "r1" / "runtime" / "t__abc" / "x" / "run" / "review-t__abc",
        purpose="reviewer",
    )

    out = CliRunner().invoke(app, ["train", "token-coverage", str(job), "--json"])

    assert out.exit_code == 0, out.output
    report = json.loads(out.output)
    assert report["rollouts"] == 1
    assert [r["rollout"] for r in report["per_rollout"]] == ["t__abc"]


def test_token_coverage_of_one_rollout_dir(tmp_path: Path) -> None:
    from benchflow.cli.main import app

    trial = tmp_path / "t__abc"
    _result(trial)

    out = CliRunner().invoke(app, ["train", "token-coverage", str(trial), "--json"])

    assert json.loads(out.output)["rollouts"] == 1
