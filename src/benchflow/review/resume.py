"""Retry terminal rubric scoring from durable solver evidence.

The solver snapshot is the authority for provenance and deterministic reward.
A previous final result is only consulted to avoid rejudging a completed trial.
No function in this module starts a solver. Declared workspace recovery may
retry deterministic verification before rubric review.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
from typing import Any

from benchflow._utils.task_authoring import task_digest
from benchflow.review.options import ReviewerConfig
from benchflow.review.outcome import scoring_from_result

logger = logging.getLogger(__name__)


class ReviewResumeError(ValueError):
    """Saved evidence cannot safely be used to finish the requested scoring."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ReviewResumeError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReviewResumeError(f"Expected a JSON object in {path}")
    return payload


def _trusted_task(rollout_dir: Path, solver: dict[str, Any], tasks_root: Path) -> Path:
    """Resolve only inside an operator-supplied root and pin every task file."""
    recorded = solver.get("task_name")
    if not isinstance(recorded, str) or not recorded:
        raise ReviewResumeError("solver.json has no task_name")
    name = PurePosixPath(recorded.replace("\\", "/")).name
    root = tasks_root.resolve(strict=True)
    candidate = root if root.name == name else (root / name).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_dir():
        raise ReviewResumeError(f"Task {name!r} is not inside --tasks-root {root}")
    expected = solver.get("task_digest")
    if not isinstance(expected, str) or not expected.startswith("sha256:"):
        raise ReviewResumeError("solver.json has no valid task_digest")
    if task_digest(candidate) != expected:
        raise ReviewResumeError("Task digest mismatch: restore the exact solver task")
    for filename in ("config.json", "result.json"):
        path = rollout_dir / filename
        if path.is_file():
            recorded_digest = _read_object(path).get("task_digest")
            if recorded_digest is not None and recorded_digest != expected:
                raise ReviewResumeError(
                    f"{filename} and solver.json task digests differ"
                )
    return candidate


def _reviewer_options(
    rollout_dir: Path, override: ReviewerConfig | None
) -> ReviewerConfig:
    """Retain the saved runtime while resolving secrets in the current process."""
    config_path = rollout_dir / "config.json"
    review = (
        _read_object(config_path).get("review", {}) if config_path.is_file() else {}
    )
    if not isinstance(review, dict):
        raise ReviewResumeError("config.json review must be an object")
    saved = review.get("reviewer", {})
    if not isinstance(saved, dict):
        raise ReviewResumeError("config.json review.reviewer must be an object")
    # Public config records credential names, never values. They are supplied
    # through the current environment or explicit private reviewer overrides.
    saved = {key: value for key, value in saved.items() if key != "agent_env_keys"}
    if override is not None:
        fields = override.model_dump(exclude_unset=True)
        if "agent_env" in fields:
            fields["agent_env"] = {
                **ReviewerConfig.model_validate(saved).agent_env,
                **override.agent_env,
            }
        saved.update(fields)
    return ReviewerConfig.model_validate(saved)


async def resume_review(
    rollout_dir: str | Path,
    *,
    tasks_root: str | Path,
    reviewer: ReviewerConfig | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Finish a saved trial's review, preserving the original solver record.

    Completed verdicts, including valid failures, are returned unchanged unless
    ``force`` explicitly requests a new scoring revision. ``tasks_root`` is a
    trusted local task collection or the original task directory itself.
    """
    from benchflow.review.automatic import finish_review, prepare_review
    from benchflow.review.persistence import commit_scoring_result, scoring_lock
    from benchflow.rollout._verifier_recovery import (
        needs_verifier_recovery,
        recorded_recovery_ineligible_reason,
        resume_verification,
    )
    from benchflow.task import Task

    path = Path(rollout_dir).resolve(strict=True)
    with scoring_lock(path):
        current_path = path / "result.json"
        if current_path.is_file():
            current = _read_object(current_path)
            scoring = scoring_from_result(current)
            if scoring is not None and scoring.status == "complete" and not force:
                return current
        solver = _read_object(path / "solver.json")
        if solver.get("purpose", "task") != "task":
            raise ReviewResumeError(
                "Reviewer child runs cannot be resumed as task trials"
            )
        task_path = _trusted_task(path, solver, Path(tasks_root))
        await resume_verification(path, task_path)
        plan = prepare_review(task_path, _reviewer_options(path, reviewer))
        if plan is None:
            verifier = Task(task_path).config.verifier
            # Without review, only an eligible recovery contract has scoring
            # to finish; any other trial is refused as before.
            if recorded_recovery_ineligible_reason(path) is not None or not (
                verifier.workspace_recovery
                or needs_verifier_recovery(solver.get("verifier_error"))
            ):
                raise ReviewResumeError(
                    "The original task has no automatic review rubric"
                )
            return commit_scoring_result(path, None)
        review = _read_object(path / "config.json").get("review", {})
        if not isinstance(review, dict):
            raise ReviewResumeError("config.json review must be an object")
        expected_rubric = review.get("rubric_sha256")
        if (
            expected_rubric is not None
            and hashlib.sha256(plan.rubric_path.read_bytes()).hexdigest()
            != expected_rubric
        ):
            raise ReviewResumeError(
                "Rubric digest differs from the original review plan"
            )
        scoring = await finish_review(plan, path)
        return commit_scoring_result(path, scoring)


async def resume_pending_reviews(
    job_dir: Path,
    *,
    tasks_root: Path,
    reviewer: ReviewerConfig,
    task_names: set[str],
) -> None:
    """Resume at most one unfinished scoring attempt per selected task.

    Direct children are task trials; nested reviewer runs are never candidates.
    A completed score always wins over an orphaned retry, matching evaluation
    resume's precedence for durable scores. The shared reviewer runtime owns
    concurrency limits; resumed reviews do not occupy solver slots.

    A trial whose resume fails is logged and left with its saved result for a
    later ``bench eval score``; it cannot cancel the other trials' reviews or
    abort the evaluation (#1134).
    """
    from benchflow.rollout._verifier_recovery import (
        recorded_recovery_ineligible_reason,
    )

    best: dict[str, tuple[tuple[bool, float, str], Path]] = {}
    for snapshot in job_dir.glob("*/solver.json"):
        solver = _read_object(snapshot)
        task_name = solver.get("task_name")
        if task_name not in task_names or solver.get("purpose", "task") != "task":
            continue
        config_path = snapshot.parent / "config.json"
        config = _read_object(config_path) if config_path.is_file() else {}
        reviewed = "review" in config
        if (
            not reviewed
            and recorded_recovery_ineligible_reason(snapshot.parent) is not None
        ):
            # Neither review nor recovery owns this trial; its original result
            # stays for the ordinary rerun, as before.
            continue
        result_path = snapshot.parent / "result.json"
        result_payload = _read_object(result_path) if result_path.is_file() else {}
        scoring = scoring_from_result(result_payload)
        complete = scoring is not None and scoring.status == "complete"
        if not reviewed and scoring is None:
            # Verifier recovery without a review commits no scoring block: a
            # reward with no verifier error is its final verdict. A reviewed
            # trial is complete only once its review is: a result without
            # scoring would carry the unreviewed verifier reward.
            complete = result_payload.get("rewards") is not None and not (
                result_payload.get("verifier_error")
            )
        rank = (complete, snapshot.stat().st_mtime, str(snapshot))
        previous = best.get(task_name)
        if previous is None or rank > previous[0]:
            best[task_name] = (rank, snapshot.parent)

    async def resume_one(path: Path) -> None:
        try:
            await resume_review(path, tasks_root=tasks_root, reviewer=reviewer)
        except ReviewResumeError as exc:
            logger.warning(
                "Could not resume scoring for %s: %s; its saved result is kept",
                path.name,
                exc,
            )
        except Exception:
            logger.exception(
                "Could not resume scoring for %s; its saved result is kept", path.name
            )

    async with asyncio.TaskGroup() as group:
        for (complete, _, _), path in best.values():
            if not complete:
                group.create_task(resume_one(path))
