"""Integrate a task's weighted rubric into one canonical rollout result.

The reviewer itself remains an ordinary isolated rollout. This module owns
the narrow bridge back to the source rollout: retrying the reviewer, retaining
its audit artifacts under non-conflicting names, applying blocker gates, and
refreshing every score-bearing source artifact exactly once.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow.models import RolloutResult
from benchflow.review.config import ReviewRubricError, load_rubric
from benchflow.review.policy import RubricReviewConfig
from benchflow.review.runner import ReviewReport, TrialReview, run_reviews
from benchflow.review.workspace import (
    WORKSPACE_BASELINE_FILENAME,
    read_workspace_manifest,
)
from benchflow.rollout._results import finalize_rubric_review_artifacts

logger = logging.getLogger(__name__)

REVIEW_ARTIFACT_DIRNAME = "review"


def _discard_workspace_baseline(rollout_dir: Path) -> None:
    """Keep the internal pre-agent inventory out of published artifacts."""

    path = rollout_dir / "artifacts" / WORKSPACE_BASELINE_FILENAME
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not discard rubric workspace baseline %s: %s", path, exc)


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_persisted_trajectory(rollout_dir: Path) -> list[dict[str, Any]]:
    path = rollout_dir / "trajectory" / "acp_trajectory.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _persisted_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def load_persisted_rollout_result(rollout_dir: Path) -> RolloutResult:
    """Rehydrate the result fields needed to finish an interrupted review."""

    payload = _read_object(rollout_dir / "result.json")
    if payload is None:
        raise RuntimeError(f"missing or unreadable source result: {rollout_dir}")
    task_name = payload.get("task_name")
    if not isinstance(task_name, str) or not task_name:
        raise RuntimeError("persisted source result has no task_name")
    agent_result = payload.get("agent_result")
    if not isinstance(agent_result, dict):
        agent_result = {}
    rewards = payload.get("rewards")
    return RolloutResult(
        task_name=task_name,
        rollout_name=str(payload.get("rollout_name") or rollout_dir.name),
        rewards=dict(rewards) if isinstance(rewards, dict) else None,
        trajectory=_read_persisted_trajectory(rollout_dir),
        agent=str(payload.get("agent") or ""),
        agent_name=str(payload.get("agent_name") or ""),
        model=(str(payload["model"]) if payload.get("model") is not None else None),
        n_tool_calls=int(agent_result.get("n_tool_calls") or 0),
        n_skill_invocations=int(agent_result.get("n_skill_invocations") or 0),
        n_prompts=int(agent_result.get("n_prompts") or 0),
        n_input_tokens=agent_result.get("n_input_tokens"),
        n_output_tokens=agent_result.get("n_output_tokens"),
        n_cache_read_tokens=agent_result.get("n_cache_read_tokens"),
        n_cache_creation_tokens=agent_result.get("n_cache_creation_tokens"),
        total_tokens=agent_result.get("total_tokens"),
        cost_usd=agent_result.get("cost_usd"),
        usage_source=agent_result.get("usage_source") or "unavailable",
        price_source=agent_result.get("price_source"),
        usage_details=agent_result.get("usage_details"),
        error=payload.get("error"),
        error_category=payload.get("error_category"),
        verifier_error=payload.get("verifier_error"),
        verifier_error_category=payload.get("verifier_error_category"),
        export_error=payload.get("export_error"),
        partial_trajectory=bool(payload.get("partial_trajectory", False)),
        trajectory_source=payload.get("trajectory_source"),
        source_provenance=(
            dict(payload["source"])
            if isinstance(payload.get("source"), dict)
            else None
        ),
        final_score=(
            dict(payload["final_score"])
            if isinstance(payload.get("final_score"), dict)
            else None
        ),
        rubric_review=(
            dict(payload["rubric_review"])
            if isinstance(payload.get("rubric_review"), dict)
            else None
        ),
        started_at=_persisted_datetime(payload.get("started_at")),
        finished_at=_persisted_datetime(payload.get("finished_at")),
    )


def _copy_tree_without_links(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        ignore=lambda directory, names: {
            name for name in names if (Path(directory) / name).is_symlink()
        },
    )


def _retain_reviewer_artifacts(
    rollout_dir: Path,
    report: ReviewReport,
    trial: TrialReview,
) -> tuple[str, dict[str, Any] | None]:
    """Copy reviewer evidence under names that cannot pollute job discovery."""

    destination = rollout_dir / REVIEW_ARTIFACT_DIRNAME
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    report_data = report.to_dict()
    report_data["path"] = "."
    for trial_data in report_data.get("trials", []):
        if isinstance(trial_data, dict):
            trial_data["source_rollout"] = "."
            trial_data["reviewer_rollout"] = "review/"
    (destination / "review-report.json").write_text(
        json.dumps(report_data, indent=2) + "\n", encoding="utf-8"
    )

    reviewer_result: dict[str, Any] | None = None
    if trial.reviewer_rollout:
        leaf = Path(trial.reviewer_rollout)
        file_mapping = {
            "config.json": "reviewer-config.json",
            "result.json": "reviewer-result.json",
            "timing.json": "reviewer-timing.json",
            "prompts.json": "reviewer-prompts.json",
            "rewards.jsonl": "reviewer-rewards.jsonl",
            "results.jsonl": "reviewer-training-results.jsonl",
        }
        for source_name, destination_name in file_mapping.items():
            source = leaf / source_name
            if source.is_file() and not source.is_symlink():
                shutil.copy2(source, destination / destination_name)
        for dirname in ("agent", "verifier", "trajectory", "trainer", "artifacts"):
            _copy_tree_without_links(leaf / dirname, destination / dirname)
        reviewer_result = _read_object(leaf / "result.json")
    return "review/review-report.json", reviewer_result


def _reviewer_metadata(
    policy: RubricReviewConfig,
    *,
    environment: str,
    reviewer_result: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = {
        **policy.to_config_artifact(),
        "environment": environment,
    }
    if reviewer_result is not None:
        metadata.update(
            {
                "rollout_name": reviewer_result.get("rollout_name"),
                "agent_name": reviewer_result.get("agent_name"),
                "agent_result": reviewer_result.get("agent_result"),
                "timing": reviewer_result.get("timing"),
                "error": reviewer_result.get("error"),
                "verifier_error": reviewer_result.get("verifier_error"),
            }
        )
    return metadata


@dataclass(frozen=True)
class _ReviewerOutcome:
    """Reviewer retry result after durable evidence retention."""

    trial: TrialReview | None
    attempts: list[dict[str, Any]]
    artifact_path: str | None = None
    reviewer_result: dict[str, Any] | None = None
    artifact_error: str | None = None


async def _run_reviewer_with_retries(
    *,
    rollout_dir: Path,
    task_path: Path,
    rubric_path: Path,
    environment: str,
    policy: RubricReviewConfig,
) -> _ReviewerOutcome:
    """Run bounded attempts and retain the final attempt before temp cleanup."""

    attempts: list[dict[str, Any]] = []
    selected: tuple[ReviewReport, TrialReview] | None = None
    with tempfile.TemporaryDirectory(prefix="benchflow-auto-review-") as temporary:
        review_root = Path(temporary)
        for attempt in range(1, policy.max_retries + 2):
            attempt_dir = review_root / f"attempt-{attempt}"
            try:
                report, _ = await run_reviews(
                    rollout_dir,
                    agent=policy.agent,
                    model=policy.model,
                    reasoning_effort=policy.reasoning_effort,
                    environment=environment,
                    rubric_path=rubric_path,
                    agent_env=policy.agent_env,
                    concurrency=1,
                    timeout_sec=policy.timeout_sec,
                    open_network=policy.allow_open_network,
                    tasks_root=task_path.parent,
                    out_dir=attempt_dir,
                )
                trial = report.trials[0]
                attempts.append(
                    {
                        "attempt": attempt,
                        "valid": trial.review_valid,
                        "error": trial.error,
                    }
                )
                selected = report, trial
                if trial.review_valid:
                    break
            except Exception as exc:
                attempts.append(
                    {"attempt": attempt, "valid": False, "error": str(exc)}
                )

        if selected is None:
            return _ReviewerOutcome(trial=None, attempts=attempts)
        report, trial = selected
        try:
            artifact_path, reviewer_result = _retain_reviewer_artifacts(
                rollout_dir, report, trial
            )
        except OSError as exc:
            return _ReviewerOutcome(
                trial=trial,
                attempts=attempts,
                artifact_error=str(exc),
            )
        return _ReviewerOutcome(
            trial=trial,
            attempts=attempts,
            artifact_path=artifact_path,
            reviewer_result=reviewer_result,
        )


def _attach_reviewer_evidence(
    payload: dict[str, Any],
    outcome: _ReviewerOutcome,
    *,
    policy: RubricReviewConfig,
    environment: str,
) -> None:
    """Project one retained reviewer attempt into the source result payload."""

    payload["attempts"] = outcome.attempts
    if outcome.artifact_error is not None:
        payload["artifact_error"] = outcome.artifact_error
    trial = outcome.trial
    if trial is None:
        return
    payload.update(
        {
            "summary": trial.summary,
            "checks": trial.checks,
            "reviewer": _reviewer_metadata(
                policy,
                environment=environment,
                reviewer_result=outcome.reviewer_result,
            ),
        }
    )
    if outcome.artifact_path is not None:
        payload["artifact"] = outcome.artifact_path


def _base_review_payload(
    *,
    policy: RubricReviewConfig,
    rubric_path: Path,
    original_rewards: dict[str, Any] | None,
    workspace_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": "pending",
        "rubric": {"path": str(rubric_path)},
        "reviewer": policy.to_config_artifact(),
        "workspace": workspace_manifest,
        "deterministic_verifier_rewards": original_rewards,
        "attempts": [],
    }


def _finish_review_error(
    rollout_dir: Path,
    result: RolloutResult,
    *,
    payload: dict[str, Any],
    message: str,
    started_at: datetime,
) -> RolloutResult:
    payload.update({"status": "error", "error": message})
    prior_error = result.verifier_error
    verifier_error = (
        f"{prior_error}; rubric review failed: {message}"
        if prior_error
        else f"rubric review failed: {message}"
    )
    final_score = {
        "status": "error",
        "pass": None,
        "pass_at_1": None,
        "score": None,
        "weighted_points": None,
        "max_weighted_points": None,
    }
    return finalize_rubric_review_artifacts(
        rollout_dir,
        result,
        rewards=None,
        final_score=final_score,
        rubric_review=payload,
        verifier_error=verifier_error,
        review_elapsed_sec=(datetime.now() - started_at).total_seconds(),
    )


def mark_task_rubric_review_error(
    result: RolloutResult,
    *,
    rollout_dir: Path,
    rubric_path: Path,
    policy: RubricReviewConfig,
    message: str,
) -> RolloutResult:
    """Fail closed when orchestration escapes the normal review error path."""

    started_at = datetime.now()
    _discard_workspace_baseline(rollout_dir)
    payload = _base_review_payload(
        policy=policy,
        rubric_path=rubric_path,
        original_rewards=(dict(result.rewards) if result.rewards is not None else None),
        workspace_manifest=read_workspace_manifest(rollout_dir),
    )
    return _finish_review_error(
        rollout_dir,
        result,
        payload=payload,
        message=message,
        started_at=started_at,
    )


def _finish_valid_review(
    rollout_dir: Path,
    result: RolloutResult,
    *,
    payload: dict[str, Any],
    trial: TrialReview,
    original_rewards: dict[str, Any] | None,
    started_at: datetime,
) -> RolloutResult:
    """Integrate one structurally valid legacy or weighted reviewer result."""

    payload["status"] = "complete"
    if trial.scoring is None:
        payload["status"] = "complete_unweighted"
        payload["note"] = (
            "Legacy rubric reviewed but not integrated into weighted final scoring"
        )
        final_score = {
            "status": "not_weighted",
            "pass": None,
            "pass_at_1": None,
            "score": None,
            "weighted_points": None,
            "max_weighted_points": None,
        }
        return finalize_rubric_review_artifacts(
            rollout_dir,
            result,
            rewards=original_rewards,
            final_score=final_score,
            rubric_review=payload,
            verifier_error=result.verifier_error,
            review_elapsed_sec=(datetime.now() - started_at).total_seconds(),
        )

    scoring = trial.scoring
    payload["scoring"] = scoring.to_dict()
    rewards = dict(original_rewards or {})
    verifier_reward = rewards.get("reward")
    rewards.update(
        {
            "reward": scoring.reward,
            "pass_at_1": float(scoring.passed),
            "score": scoring.score,
            "verifier_reward": verifier_reward,
            "weighted_points": float(scoring.weighted_points),
            "max_weighted_points": float(scoring.max_weighted_points),
            "failed_blockers": list(scoring.failed_blockers),
        }
    )
    final_score = {
        "status": "complete",
        "pass": scoring.passed,
        "pass_at_1": int(scoring.passed),
        "score": scoring.score,
        "weighted_points": scoring.weighted_points,
        "max_weighted_points": scoring.max_weighted_points,
        "raw_quality": scoring.raw_quality,
        "deterministic_pass": scoring.deterministic_pass,
        "all_blockers_pass": scoring.all_blockers_pass,
        "failed_blockers": list(scoring.failed_blockers),
    }
    return finalize_rubric_review_artifacts(
        rollout_dir,
        result,
        rewards=rewards,
        final_score=final_score,
        rubric_review=payload,
        verifier_error=result.verifier_error,
        review_elapsed_sec=(datetime.now() - started_at).total_seconds(),
    )


async def integrate_task_rubric_review(
    result: RolloutResult,
    *,
    rollout_dir: Path,
    task_path: Path,
    rubric_path: Path,
    source_environment: str,
    policy: RubricReviewConfig,
    workspace_error: str | None,
) -> RolloutResult:
    """Run and integrate one automatic rubric review, failing closed on error."""

    started_at = datetime.now()
    _discard_workspace_baseline(rollout_dir)
    original_rewards = dict(result.rewards) if result.rewards is not None else None
    workspace_manifest = read_workspace_manifest(rollout_dir)
    payload = _base_review_payload(
        policy=policy,
        rubric_path=rubric_path,
        original_rewards=original_rewards,
        workspace_manifest=workspace_manifest,
    )
    if workspace_error:
        return _finish_review_error(
            rollout_dir,
            result,
            payload=payload,
            message=workspace_error,
            started_at=started_at,
        )
    if workspace_manifest is None:
        return _finish_review_error(
            rollout_dir,
            result,
            payload=payload,
            message="final workspace manifest is missing or unreadable",
            started_at=started_at,
        )
    try:
        rubric = load_rubric(rubric_path)
    except ReviewRubricError as exc:
        return _finish_review_error(
            rollout_dir,
            result,
            payload=payload,
            message=str(exc),
            started_at=started_at,
        )

    payload["rubric"] = {
        "path": str(rubric_path),
        "contract": rubric.contract,
        "criteria": [criterion.metadata() for criterion in rubric.criteria],
    }
    environment = policy.environment or source_environment
    outcome = await _run_reviewer_with_retries(
        rollout_dir=rollout_dir,
        task_path=task_path,
        rubric_path=rubric_path,
        environment=environment,
        policy=policy,
    )
    _attach_reviewer_evidence(
        payload, outcome, policy=policy, environment=environment
    )
    trial = outcome.trial
    if trial is None or not trial.review_valid or outcome.artifact_error is not None:
        last_error = outcome.attempts[-1].get("error") if outcome.attempts else None
        message = str(outcome.artifact_error or last_error or "invalid review")
        return _finish_review_error(
            rollout_dir,
            result,
            payload=payload,
            message=message,
            started_at=started_at,
        )

    return _finish_valid_review(
        rollout_dir,
        result,
        payload=payload,
        trial=trial,
        original_rewards=original_rewards,
        started_at=started_at,
    )
