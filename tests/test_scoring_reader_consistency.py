"""Guards scoring regressions present at commit 764325430."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow._utils.scoring import score_summary_fields
from benchflow.batch import Results
from benchflow.cli.rescore import _refresh_job_artifacts
from benchflow.jobs import load_job
from benchflow.models import RolloutResult
from benchflow.review.outcome import ScoringResult, scoring_error
from benchflow.review.persistence import commit_scoring_result


def _solver(job: Path, *, timeout: bool = False) -> Path:
    trial = job / "task__one"
    (trial / "trajectory").mkdir(parents=True)
    source = {
        "task_name": "task",
        "rollout_name": trial.name,
        "agent": "codex-acp",
        "agent_name": "codex-acp",
        "model": "azure/test",
        "n_tool_calls": 1,
        "rewards": {"reward": 1.0},
        "verifier_error": None,
        "error": "Agent prompt exceeded wall-clock budget 60s" if timeout else None,
        "error_category": "timeout" if timeout else None,
        "partial_trajectory": timeout,
        "started_at": "2026-10-26T00:00:00",
        "finished_at": "2026-10-26T00:01:00",
        "timing": {"total": 60.0},
    }
    (trial / "solver.json").write_text(json.dumps(source))
    (trial / "trajectory/acp_trajectory.jsonl").write_text("")
    (trial / "prompts.json").write_text("[]")
    (trial / "config.json").write_text('{"agent": "codex-acp"}')
    return trial


def _complete(*, reward: float, tests: bool = True, blockers: bool = True):
    return ScoringResult(
        status="complete",
        passed=tests and blockers,
        tests_pass=tests,
        all_blockers_pass=blockers,
        failed_blockers=[] if blockers else ["validity"],
        verifier_reward=1.0 if tests else 0.0,
        rubric_reward=reward,
        reviewer_run="reviews/fixture",
    )


@pytest.mark.parametrize(
    ("quality", "tests", "blockers"),
    [
        (0.6, True, True),
        (0.0, True, True),
        (1.0, True, True),
        (1.0, False, True),
        (1.0, True, False),
    ],
)
def test_completed_verdict_survives_all_job_readers(tmp_path, quality, tests, blockers):
    """Guards commit 764325430: rubric gate success is independent of quality."""
    job_path = tmp_path / "job"
    trial_path = _solver(job_path)
    scoring = _complete(reward=quality, tests=tests, blockers=blockers)
    result = commit_scoring_result(trial_path, scoring)
    expected_pass = tests and blockers
    batch = Results([RolloutResult.from_dict(result)])
    assert batch.denominators().passed == int(expected_pass)
    assert batch.mean_reward == (quality if expected_pass else 0.0)
    for job in (load_job(job_path), load_job(trial_path / "results.jsonl")):
        trial = job.trials[0]
        assert trial.scoring == scoring
        assert trial.passed is expected_pass
        assert trial.assessment == "scored"
        assert trial.reward == (quality if expected_pass else 0.0)
        denominators = job.denominators()
        assert denominators.passed == denominators.clean_passed == int(expected_pass)
        assert denominators.pass_rate_attempted == float(expected_pass)
        assert job.solve_rates(ks=[1]).get(1).pass_at_k == float(expected_pass)


def test_incomplete_timeout_keeps_evidence_without_exporting_a_score(tmp_path):
    """Guards commit 764325430: a verifier pass cannot complete a missing review."""
    job_path = tmp_path / "job"
    trial_path = _solver(job_path, timeout=True)
    original = (trial_path / "solver.json").read_bytes()
    # A preceding completed score must not survive an unsuccessful rescore.
    commit_scoring_result(trial_path, _complete(reward=0.6))
    result = commit_scoring_result(
        trial_path,
        scoring_error(
            "Solver trajectory is incomplete", tests_pass=True, verifier_reward=1.0
        ),
    )
    assert (trial_path / "solver.json").read_bytes() == original
    assert result["rewards"] == {"reward": 1.0}
    assert result["error_category"] == "timeout"
    batch = Results([RolloutResult.from_dict(result)])
    assert batch[0].reward is None
    assert batch.denominators().scored == 0
    assert batch.mean_reward == 0.0
    assert batch.mean_reward_scored_only is None
    assert not (trial_path / "rewards.jsonl").exists()
    row = json.loads((trial_path / "results.jsonl").read_text())
    assert row["reward"] is None and row["score"] is None
    assert row["passed"] is None
    assert "reward" not in row["metrics"]
    trainer = json.loads((trial_path / "trainer/verifiers.jsonl").read_text())
    # The older ORS schema requires a number plus an explicit invalid flag.
    assert trainer["reward"] == 0.0
    assert trainer["info"]["reward_valid"] is False
    for job in (load_job(job_path), load_job(trial_path / "results.jsonl")):
        trial = job.trials[0]
        assert trial.reward is None and not trial.passed
        assert trial.assessment != "scored"
        assert job.solve_rates(ks=[1]).get(1).pass_at_k == 0.0
        assert job.denominators().scored == 0
        assert job.denominators().mean_reward_scored_only is None

    summary = {"job_name": "job", **score_summary_fields([result])}
    (job_path / "summary.json").write_text(json.dumps(summary))
    _refresh_job_artifacts(job_path)
    refreshed = json.loads((job_path / "summary.json").read_text())
    assert refreshed["passed"] == 0
    assert refreshed["solve_rates"]["pass_at_k"]["1"] == 0.0


@pytest.mark.parametrize("passed", [True, False])
def test_rescore_refreshes_plan_score_in_both_summaries(tmp_path, passed):
    """Guards commit 764325430: a rescored partial job retains its plan denominator."""
    job_path = tmp_path / "job"
    trial_path = _solver(job_path)
    commit_scoring_result(trial_path, _complete(reward=0.6, blockers=passed))
    summary = {
        "job_name": "job",
        "planned": 2,
        "ran": 1,
        "not_run": 1,
        "not_run_tasks": ["unfinished"],
        "score_of_plan": 0.0 if passed else 0.5,
    }
    for path in (job_path / "summary.json", tmp_path / "summary.json"):
        path.write_text(json.dumps(summary))
    _refresh_job_artifacts(job_path)
    for path in (job_path / "summary.json", tmp_path / "summary.json"):
        refreshed = json.loads(path.read_text())
        assert refreshed["score_of_plan"] == (0.5 if passed else 0.0)
        assert refreshed["planned"] == 2 and refreshed["ran"] == 1
        assert refreshed["not_run_tasks"] == ["unfinished"]
