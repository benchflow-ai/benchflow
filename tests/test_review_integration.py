"""Automatic task-rubric integration and artifact consistency tests."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from benchflow._utils.evaluation_results import rubric_review_summary
from benchflow._utils.task_authoring import task_digest
from benchflow.evaluation import Evaluation, EvaluationConfig
from benchflow.models import RolloutResult
from benchflow.review.config import load_rubric
from benchflow.review.integration import integrate_task_rubric_review
from benchflow.review.lifecycle import AutomaticRubricReview
from benchflow.review.policy import RubricReviewConfig
from benchflow.review.resume import resume_incomplete_rubric_reviews
from benchflow.review.runner import ReviewReport, TrialReview
from benchflow.review.scoring import score_weighted_review
from benchflow.review.workspace import _CAPTURE_SCRIPT


def _rubric(task: Path) -> Path:
    verifier = task / "verifier"
    verifier.mkdir(parents=True)
    path = verifier / "rubric.json"
    path.write_text(
        json.dumps(
            {
                "criteria": [
                    {
                        "name": "required_output",
                        "blocker": 1,
                        "weight": 10,
                        "description": "Required output exists",
                        "guidance": "Inspect the final workspace output.",
                    },
                    {
                        "name": "quality",
                        "blocker": 0,
                        "weight": 3,
                        "description": "Output quality",
                        "guidance": "Judge the result quality.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_automatic_review_is_default_only_for_tasks_with_rubrics(tmp_path):
    """Guards the rubric final-score PR's zero-configuration activation policy."""

    plain_task = tmp_path / "plain"
    plain_task.mkdir()
    reviewed_task = tmp_path / "reviewed"
    _rubric(reviewed_task)
    policy = RubricReviewConfig()

    assert policy.enabled is True
    assert policy.agent == "codex-acp"
    assert policy.model == "openai/gpt-5.6-sol"
    assert policy.reasoning_effort == "xhigh"
    assert AutomaticRubricReview.discover(plain_task, policy) is None
    assert AutomaticRubricReview.discover(reviewed_task, policy) is not None


def _source_rollout(tmp_path: Path, *, reward: float = 1.0):
    rollout = tmp_path / "jobs" / "task__abc123"
    (rollout / "artifacts" / "workspace").mkdir(parents=True)
    (rollout / "artifacts" / "workspace" / "answer.txt").write_text("42")
    (rollout / "artifacts" / "workspace-manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "copied_file_count": 1,
                "copied_bytes": 2,
                "copied_files": [{"path": "answer.txt", "size": 2}],
            }
        )
    )
    (rollout / "trajectory").mkdir()
    (rollout / "trajectory" / "acp_trajectory.jsonl").write_text("")
    (rollout / "prompts.json").write_text('["solve"]')
    (rollout / "timing.json").write_text('{"total": 1.0}')
    result_data = {
        "task_name": "task",
        "rollout_name": "task__abc123",
        "rewards": {"reward": reward, "test_count": 3},
        "agent": "gemini",
        "agent_name": "gemini",
        "model": "test-model",
        "agent_result": {"usage_source": "unavailable"},
        "error": None,
        "verifier_error": None,
        "timing": {"total": 1.0},
    }
    (rollout / "result.json").write_text(json.dumps(result_data))
    (rollout / "rewards.jsonl").write_text('{"value": 1.0}\n')
    result = RolloutResult(
        task_name="task",
        rollout_name="task__abc123",
        rewards={"reward": reward, "test_count": 3},
        trajectory=[],
        agent="gemini",
        agent_name="gemini",
        model="test-model",
        started_at=datetime.now(),
    )
    return rollout, result


def _fake_review(
    out_dir: Path,
    *,
    rubric_path: Path,
    blocker: str = "pass",
    deterministic_pass: bool = True,
) -> tuple[ReviewReport, TrialReview]:
    rubric = load_rubric(rubric_path)
    checks = {
        "required_output": {"outcome": blocker, "explanation": "inspected"},
        "quality": {"score": 2, "explanation": "complete"},
    }
    leaf = out_dir / "runtime" / "reviewer__xyz"
    (leaf / "trajectory").mkdir(parents=True)
    (leaf / "config.json").write_text('{"agent":"codex-acp"}')
    (leaf / "result.json").write_text(
        json.dumps(
            {
                "rollout_name": "reviewer__xyz",
                "agent_name": "codex",
                "agent_result": {"total_tokens": 100, "cost_usd": 0.01},
                "timing": {"total": 2.0},
                "error": None,
                "verifier_error": None,
            }
        )
    )
    (leaf / "timing.json").write_text('{"total":2.0}')
    trial = TrialReview(
        trial_name="task__abc123",
        source_rollout="source",
        review_valid=True,
        summary="reviewed",
        checks=checks,
        reviewer_rollout=str(leaf),
        rubric_path=str(rubric_path),
        rubric_contract=rubric.contract,
        criteria=[criterion.name for criterion in rubric.criteria],
        criterion_metadata=[criterion.metadata() for criterion in rubric.criteria],
        scoring=score_weighted_review(
            rubric, checks, deterministic_pass=deterministic_pass
        ),
    )
    report = ReviewReport(
        path="source",
        rubric_path=str(rubric_path),
        criteria=trial.criteria,
        agent="codex-acp",
        model="openai/gpt-5.6-sol",
        reasoning_effort="xhigh",
        environment="docker",
        trials=[trial],
    )
    return report, trial


@pytest.mark.asyncio
async def test_weighted_review_updates_every_score_artifact(tmp_path, monkeypatch):
    """Guards the rubric final-score PR's canonical artifact synchronization."""

    task = tmp_path / "tasks" / "task"
    rubric_path = _rubric(task)
    rollout, result = _source_rollout(tmp_path)
    async def fake_run_reviews(_path, **kwargs):
        report, _trial = _fake_review(
            kwargs["out_dir"], rubric_path=kwargs["rubric_path"]
        )
        report_path = kwargs["out_dir"] / "review_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict()))
        return report, report_path

    monkeypatch.setattr("benchflow.review.integration.run_reviews", fake_run_reviews)
    policy = RubricReviewConfig()
    updated = await integrate_task_rubric_review(
        result,
        rollout_dir=rollout,
        task_path=task,
        rubric_path=rubric_path,
        source_environment="docker",
        policy=policy,
        workspace_error=None,
    )

    assert updated.rewards == {
        "reward": 1.0,
        "test_count": 3,
        "pass_at_1": 1.0,
        "score": 1.0,
        "verifier_reward": 1.0,
        "weighted_points": 6.0,
        "max_weighted_points": 6.0,
        "failed_blockers": [],
    }
    persisted = json.loads((rollout / "result.json").read_text())
    assert persisted["final_score"]["pass"] is True
    assert persisted["final_score"]["score"] == 1.0
    assert persisted["rubric_review"]["reviewer"]["reasoning_effort"] == "xhigh"
    reward_event = json.loads((rollout / "rewards.jsonl").read_text())
    assert reward_event["value"] == 1.0
    training = json.loads((rollout / "results.jsonl").read_text())
    assert training["reward"] == 1.0
    assert training["score"] == 1.0
    assert training["metrics"]["pass_at_1"] == 1.0
    assert (rollout / "trainer" / "verifiers.jsonl").is_file()
    assert not list((rollout / "review").rglob("result.json"))
    assert (rollout / "review" / "reviewer-result.json").is_file()


@pytest.mark.asyncio
async def test_blocker_failure_zeros_reward_pass_and_score(tmp_path, monkeypatch):
    """Guards the rubric final-score PR's blocker gate."""

    task = tmp_path / "tasks" / "task"
    rubric_path = _rubric(task)
    rollout, result = _source_rollout(tmp_path)

    async def fake_run_reviews(_path, **kwargs):
        report, _trial = _fake_review(
            kwargs["out_dir"],
            rubric_path=kwargs["rubric_path"],
            blocker="fail",
        )
        return report, kwargs["out_dir"] / "review_report.json"

    monkeypatch.setattr("benchflow.review.integration.run_reviews", fake_run_reviews)
    updated = await integrate_task_rubric_review(
        result,
        rollout_dir=rollout,
        task_path=task,
        rubric_path=rubric_path,
        source_environment="docker",
        policy=RubricReviewConfig(max_retries=0),
        workspace_error=None,
    )

    assert updated.rewards is not None
    assert updated.final_score is not None
    assert updated.rewards["reward"] == 0.0
    assert updated.rewards["pass_at_1"] == 0.0
    assert updated.rewards["score"] == 0.0
    assert updated.final_score["weighted_points"] == 6
    assert updated.final_score["failed_blockers"] == ["required_output"]


@pytest.mark.asyncio
async def test_review_failure_removes_stale_reward_and_fails_closed(tmp_path):
    """Guards the rubric final-score PR's unscored failure contract."""

    task = tmp_path / "tasks" / "task"
    rubric_path = _rubric(task)
    rollout, result = _source_rollout(tmp_path)
    baseline = rollout / "artifacts" / "workspace-baseline.json"
    baseline.write_text('{"private": "inventory"}')

    updated = await integrate_task_rubric_review(
        result,
        rollout_dir=rollout,
        task_path=task,
        rubric_path=rubric_path,
        source_environment="docker",
        policy=RubricReviewConfig(max_retries=0),
        workspace_error="workspace export unavailable",
    )

    assert updated.rewards is None
    assert updated.final_score is not None
    assert updated.final_score["status"] == "error"
    assert updated.verifier_error == (
        "rubric review failed: workspace export unavailable"
    )
    assert not (rollout / "rewards.jsonl").exists()
    assert not baseline.exists()
    persisted = json.loads((rollout / "result.json").read_text())
    assert persisted["rewards"] is None
    assert persisted["rubric_review"]["deterministic_verifier_rewards"] == {
        "reward": 1.0,
        "test_count": 3,
    }


@pytest.mark.asyncio
async def test_invalid_review_retains_last_attempt_for_diagnostics(
    tmp_path, monkeypatch
):
    """Guards the rubric final-score PR's invalid-review diagnostics."""

    task = tmp_path / "tasks" / "task"
    rubric_path = _rubric(task)
    rollout, result = _source_rollout(tmp_path)

    async def fake_run_reviews(_path, **kwargs):
        report, trial = _fake_review(
            kwargs["out_dir"], rubric_path=kwargs["rubric_path"]
        )
        trial.review_valid = False
        trial.error = "review result did not satisfy the rubric schema"
        return report, kwargs["out_dir"] / "review_report.json"

    monkeypatch.setattr("benchflow.review.integration.run_reviews", fake_run_reviews)
    updated = await integrate_task_rubric_review(
        result,
        rollout_dir=rollout,
        task_path=task,
        rubric_path=rubric_path,
        source_environment="docker",
        policy=RubricReviewConfig(max_retries=0),
        workspace_error=None,
    )

    assert updated.rewards is None
    assert updated.rubric_review is not None
    assert updated.rubric_review["status"] == "error"
    assert updated.rubric_review["artifact"] == "review/review-report.json"
    assert (rollout / "review" / "reviewer-result.json").is_file()


@pytest.mark.asyncio
async def test_resume_finishes_review_without_rerunning_source(tmp_path, monkeypatch):
    """Guards the rubric final-score PR's process-crash recovery path."""

    task = tmp_path / "tasks" / "task"
    _rubric(task)
    rollout, _result = _source_rollout(tmp_path)
    persisted = json.loads((rollout / "result.json").read_text())
    persisted["task_digest"] = task_digest(task)
    (rollout / "result.json").write_text(json.dumps(persisted))

    async def fake_run_reviews(_path, **kwargs):
        return _fake_review(
            kwargs["out_dir"], rubric_path=kwargs["rubric_path"]
        )[0], kwargs["out_dir"] / "review_report.json"

    monkeypatch.setattr("benchflow.review.integration.run_reviews", fake_run_reviews)
    evaluation = Evaluation(
        tasks_dir=task,
        jobs_dir=tmp_path,
        job_name="jobs",
        config=EvaluationConfig(rubric_review=RubricReviewConfig(max_retries=0)),
    )

    completed = evaluation._get_completed_tasks()
    refreshed = await resume_incomplete_rubric_reviews(
        completed,
        tasks_dir=task,
        job_dir=tmp_path / "jobs",
        source_environment="docker",
        policy=evaluation._config.rubric_review,
    )

    assert refreshed["task"]["final_score"]["pass"] is True
    assert refreshed["task"]["rewards"]["verifier_reward"] == 1.0
    assert (rollout / "review" / "reviewer-result.json").is_file()


@pytest.mark.asyncio
async def test_resume_rejects_changed_task_bytes(tmp_path):
    """Guards the rubric final-score PR's crash-resume provenance gate."""

    task = tmp_path / "tasks" / "task"
    _rubric(task)
    rollout, _result = _source_rollout(tmp_path)
    persisted = json.loads((rollout / "result.json").read_text())
    persisted["task_digest"] = "sha256:" + "0" * 64
    (rollout / "result.json").write_text(json.dumps(persisted))
    evaluation = Evaluation(
        tasks_dir=task,
        jobs_dir=tmp_path,
        job_name="jobs",
        config=EvaluationConfig(rubric_review=RubricReviewConfig()),
    )

    completed = await resume_incomplete_rubric_reviews(
        evaluation._get_completed_tasks(),
        tasks_dir=task,
        job_dir=tmp_path / "jobs",
        source_environment="docker",
        policy=evaluation._config.rubric_review,
    )

    assert completed == {}


def test_workspace_capture_exports_delta_and_excludes_credentials(tmp_path):
    """Guards the rubric final-score PR's output-only evidence boundary."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "input.bin").write_bytes(b"unchanged")
    (workspace / ".codex").mkdir()
    (workspace / ".codex" / "auth.json").write_text("secret")
    (workspace / ".daytona" / "sessions").mkdir(parents=True)
    (workspace / ".daytona" / "sessions" / "output.log").write_text("runtime")
    (workspace / "project" / ".ssh").mkdir(parents=True)
    (workspace / "project" / ".ssh" / "id_rsa").write_text("nested secret")
    baseline = tmp_path / "artifacts" / "workspace-baseline.json"
    destination = tmp_path / "artifacts" / "workspace"
    manifest = tmp_path / "artifacts" / "workspace-manifest.json"

    def capture(mode: str) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-",
                mode,
                str(workspace),
                str(baseline),
                str(destination),
                str(manifest),
            ],
            input=_CAPTURE_SCRIPT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    capture("baseline")
    (workspace / "answer.txt").write_text("new output")
    (workspace / "input.bin").write_bytes(b"changed")
    (workspace / ".env").write_text("API_KEY=secret")
    capture("final")

    assert (destination / "answer.txt").read_text() == "new output"
    assert (destination / "input.bin").read_bytes() == b"changed"
    assert not (destination / ".env").exists()
    assert not (destination / ".codex").exists()
    assert not (destination / ".daytona").exists()
    assert not (destination / "project" / ".ssh").exists()
    captured = json.loads(manifest.read_text())
    assert [item["path"] for item in captured["copied_files"]] == [
        "answer.txt",
        "input.bin",
    ]
    assert {item["path"] for item in captured["excluded_paths"]} >= {
        ".codex",
        ".daytona",
        ".env",
        "project/.ssh",
    }


def test_job_summary_separates_pass_at_one_quality_and_reviewer_usage():
    """Guards the rubric final-score PR's job-level score and usage split."""

    summary = rubric_review_summary(
        {
            "pass": {
                "final_score": {"pass": True, "score": 0.8},
                "rubric_review": {
                    "status": "complete",
                    "reviewer": {
                        "agent_result": {"total_tokens": 100, "cost_usd": 0.02}
                    },
                },
            },
            "blocked": {
                "final_score": {"pass": False, "score": 0.0},
                "rubric_review": {
                    "status": "complete",
                    "reviewer": {
                        "agent_result": {"total_tokens": 50, "cost_usd": 0.01}
                    },
                },
            },
            "plain": {"rewards": {"reward": 1.0}},
        }
    )["rubric_review"]

    assert summary["required"] == 2
    assert summary["completed"] == 2
    assert summary["pass_at_1"] == 0.5
    assert summary["mean_score"] == 0.4
    assert summary["reviewer_total_tokens"] == 150
    assert summary["reviewer_total_cost_usd"] == 0.03
