"""Discover task results without counting nested review/evidence runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    a malformed reward envelope remains visible as an errored trial.
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
            rank = (rewards is not None, path.stat().st_mtime, str(path))
            previous = best.get(task)
            if previous is None or rank >= previous[0]:
                best[task] = (rank, result)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.debug("Skipping corrupt result file %s: %s", path, exc)
    return {task: result for task, (_, result) in best.items()}
