"""Resume an interrupted automatic review without rerunning its source task."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from benchflow.review.config import find_task_rubric
from benchflow.review.integration import (
    integrate_task_rubric_review,
    load_persisted_rollout_result,
    mark_task_rubric_review_error,
)
from benchflow.review.policy import RubricReviewConfig
from benchflow.review.runner import task_digest_issue
from benchflow.review.workspace import read_workspace_manifest

logger = logging.getLogger(__name__)


def _task_directory(tasks_dir: Path, task_name: str) -> Path | None:
    """Resolve an evaluation-owned task name without trusting result paths."""

    root = tasks_dir.resolve()
    if root.name == task_name:
        return root
    if Path(task_name).name != task_name:
        return None
    candidate = (root / task_name).resolve()
    return candidate if root in candidate.parents and candidate.is_dir() else None


def _rollout_directory(job_dir: Path, payload: dict[str, Any]) -> Path | None:
    rollout_name = payload.get("rollout_name")
    if not isinstance(rollout_name, str) or Path(rollout_name).name != rollout_name:
        return None
    root = job_dir.resolve()
    candidate = (root / rollout_name).resolve()
    if root not in candidate.parents or not (candidate / "result.json").is_file():
        return None
    return candidate


def _read_result(rollout_dir: Path) -> dict[str, Any]:
    try:
        payload = json.loads((rollout_dir / "result.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"rubric review resume did not produce a result in {rollout_dir}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"rubric review resume did not produce an object in {rollout_dir}"
        )
    return payload


def _pending_review_inputs(
    *,
    task_name: str,
    payload: dict[str, Any],
    tasks_dir: Path,
    job_dir: Path,
) -> tuple[Path, Path, Path] | None:
    """Return exact, durable review inputs or signal a source rerun."""

    task_dir = _task_directory(tasks_dir, task_name)
    if task_dir is None:
        return None
    rubric_path = find_task_rubric(task_dir)
    if rubric_path is None or isinstance(payload.get("final_score"), dict):
        return None
    rollout_dir = _rollout_directory(job_dir, payload)
    if rollout_dir is None:
        raise RuntimeError("source rollout directory cannot be resolved safely")
    provenance_error = task_digest_issue(rollout_dir, task_dir)
    if provenance_error is not None:
        raise RuntimeError(provenance_error)
    if read_workspace_manifest(rollout_dir) is None:
        raise RuntimeError("final workspace evidence is missing or unreadable")
    return task_dir, rubric_path, rollout_dir


async def resume_incomplete_rubric_reviews(
    completed: dict[str, dict[str, Any]],
    *,
    tasks_dir: Path,
    job_dir: Path,
    source_environment: str,
    policy: RubricReviewConfig,
) -> dict[str, dict[str, Any]]:
    """Finish durable review phases and discard unsafe source reuse candidates."""

    if not policy.enabled:
        return completed
    refreshed = dict(completed)
    for task_name, payload in completed.items():
        task_dir = _task_directory(tasks_dir, task_name)
        if task_dir is None:
            continue
        rubric_path = find_task_rubric(task_dir)
        if rubric_path is None or isinstance(payload.get("final_score"), dict):
            continue
        try:
            inputs = _pending_review_inputs(
                task_name=task_name,
                payload=payload,
                tasks_dir=tasks_dir,
                job_dir=job_dir,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.info(
                "Re-running source task because its interrupted rubric review "
                "cannot be resumed safely: %s (%s)",
                task_name,
                exc,
            )
            refreshed.pop(task_name, None)
            continue
        if inputs is None:  # defensive; pending status was established above
            continue
        task_dir, rubric_path, rollout_dir = inputs
        logger.info(
            "Resuming interrupted rubric review without re-running source task: %s",
            task_name,
        )
        result = load_persisted_rollout_result(rollout_dir)
        try:
            await integrate_task_rubric_review(
                result,
                rollout_dir=rollout_dir,
                task_path=task_dir,
                rubric_path=rubric_path,
                source_environment=source_environment,
                policy=policy,
                workspace_error=None,
            )
        except Exception as exc:
            logger.exception(
                "Interrupted rubric review resume failed for %s", task_name
            )
            mark_task_rubric_review_error(
                result,
                rollout_dir=rollout_dir,
                rubric_path=rubric_path,
                policy=policy,
                message=f"resume failed: {exc}",
            )
        refreshed[task_name] = _read_result(rollout_dir)
    return refreshed
