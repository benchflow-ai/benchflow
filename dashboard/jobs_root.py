"""Shared jobs-root resolution for dashboard serve and generate."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

JOBS_ROOT_ENV = "BENCHFLOW_DASHBOARD_JOBS_ROOT"
_ROLLOUT_ARTIFACTS = {"result.json", "config.json", "timing.json", "prompts.json"}


def jobs_tree_has_rollouts(jobs: Path) -> bool:
    if not jobs.is_dir():
        return False
    with contextlib.suppress(Exception):
        return any(
            p.name in _ROLLOUT_ARTIFACTS for p in jobs.rglob("*") if p.is_file()
        )
    return False


def remembered_jobs_root(data_path: Path) -> Path | None:
    if not data_path.is_file():
        return None
    with contextlib.suppress(Exception):
        data = json.loads(data_path.read_text())
        raw = ((data.get("jobs") or {}).get("source") or {}).get("path")
        if not raw:
            return None
        remembered = Path(str(raw)).expanduser().resolve()
        if remembered.is_dir() and jobs_tree_has_rollouts(remembered):
            return remembered
    return None


def dashboard_jobs_root(*, root: Path, data_path: Path) -> Path:
    """Return the jobs tree the dashboard should mirror."""
    raw = os.environ.get(JOBS_ROOT_ENV)
    if not raw:
        local_jobs = root / "jobs"
        if jobs_tree_has_rollouts(local_jobs):
            return local_jobs
        return remembered_jobs_root(data_path) or local_jobs
    candidate = Path(raw).expanduser()
    if candidate.name != "jobs" and (candidate / "jobs").is_dir():
        candidate = candidate / "jobs"
    return candidate.resolve()
