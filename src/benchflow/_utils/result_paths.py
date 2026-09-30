"""Discover task results without counting nested review/evidence runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from benchflow._utils.scoring import assessment_status, assessment_withholds_score

logger = logging.getLogger(__name__)

# Files an Evaluation (or branch) job writes in its folder: evaluation.json
# when it starts, summary.json when it ends (the only one in older jobs).
_JOB_RECORDS = ("evaluation.json", "summary.json")


def holds_attempts(folder: Path) -> bool:
    """Whether results of one task in ``folder`` are attempts of one trial.

    An Evaluation job runs each task once and retries it, on an
    infrastructure error, in the same folder (a resume re-runs it there
    too), so its results for one task are attempts, and one of them is the
    trial's result. Any other folder, such as a ``bf.run_batch`` job or
    ``bf.run`` calls sharing a ``job_name``, holds independent rollouts:
    each is a sample of its own.
    """
    return any((folder / name).is_file() for name in _JOB_RECORDS)


def attempt_rank(result_path: Path, *, scored: bool) -> tuple[bool, float, str]:
    """How attempts of one trial are ordered; the highest is the trial's result.

    A scored attempt wins, then the newest, the rule an Evaluation's resume
    and ``summary.json`` use (:func:`load_task_results`).
    """
    try:
        mtime = result_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (scored, mtime, str(result_path))


def iter_task_result_paths(root: Path) -> list[Path]:
    """Return result files at trial boundaries, excluding reviewer children.

    A result below a trial's result or solver checkpoint is an artifact of that
    parent, not an additional trial. In particular, captured files must remain
    excluded while the parent awaits review and has no final result yet. The
    explicit role also covers orphaned reviewer runs. Corrupt task files are
    left to callers' existing error handling.
    """
    paths = sorted(root.rglob("result.json"))
    trial_dirs = {path.parent for path in paths}
    trial_dirs.update(path.parent for path in root.rglob("solver.json"))
    selected: list[Path] = []
    for path in paths:
        if any(parent in trial_dirs for parent in path.parent.parents):
            continue
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError):
            result = None
        if isinstance(result, dict) and result.get("purpose") == "reviewer":
            continue
        selected.append(path)
    return selected


def load_task_results(root: Path) -> dict[str, dict[str, Any]]:
    """Load one durable result per task, preferring scored then newer trials.

    Evaluation resume and scoring-only summary refresh must select identical
    records. Malformed files remain logged and skipped as in evaluation resume;
    a malformed reward envelope remains visible as an errored trial. A result
    that declares an outcome assessment finished executing and is durable even
    while unscored: resume must never re-run an episode awaiting assessment.
    """
    best: dict[str, tuple[tuple[bool, float, str], dict[str, Any]]] = {}
    for path in iter_task_result_paths(root):
        try:
            result = json.loads(path.read_text())
            task = result["task_name"]
            rewards = result.get("rewards")
            if (
                rewards is None
                and not result.get("verifier_error")
                and result.get("scoring") is None
                and assessment_status(result) is None
            ):
                continue
            if rewards is not None and not isinstance(rewards, dict):
                logger.warning(
                    "Malformed rewards field in %s for task %r: "
                    "expected dict or null, got %s %r — "
                    "treating as no reward (task will count as errored)",
                    path,
                    task,
                    type(rewards).__name__,
                    rewards,
                )
            scored = rewards is not None and not assessment_withholds_score(result)
            rank = (scored, path.stat().st_mtime, str(path))
            previous = best.get(task)
            if previous is None or rank >= previous[0]:
                best[task] = (rank, result)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.debug("Skipping corrupt result file %s: %s", path, exc)
    return {task: result for task, (_, result) in best.items()}
