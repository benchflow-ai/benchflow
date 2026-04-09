"""Download benchmark task repos if not present under .ref/."""

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

TASK_REPOS = {
    "skillsbench": {
        "repo": "https://github.com/benchflow-ai/skillsbench.git",
        "subdir": "tasks",
    },
    "terminal-bench-2": {
        "repo": "https://github.com/harbor-framework/terminal-bench-2.git",
    },
}


def _repo_root() -> Path:
    """Find the repo root via .git directory."""
    d = Path.cwd()
    while d != d.parent:
        if (d / ".git").exists():
            return d
        d = d.parent
    return Path.cwd()


def ensure_tasks(benchmark: str) -> Path:
    """Clone task repo into .ref/ if target directory is missing.

    Always resolves paths relative to the repo root, so it works
    regardless of the caller's working directory.
    """
    if benchmark not in TASK_REPOS:
        raise ValueError(f"Unknown benchmark: {benchmark!r}. Available: {sorted(TASK_REPOS)}")

    info = TASK_REPOS[benchmark]
    root = _repo_root()
    target = root / ".ref" / benchmark / (info.get("subdir") or "")

    if target.exists() and any(target.iterdir()):
        return target

    logger.info("Downloading %s tasks from %s...", benchmark, info["repo"])
    target.parent.mkdir(parents=True, exist_ok=True)
    clone_dir = target.parent / "_clone"

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", info["repo"], str(clone_dir)],
            check=True,
        )
        if info.get("subdir"):
            (clone_dir / info["subdir"]).rename(target)
        else:
            shutil.rmtree(clone_dir / ".git")
            clone_dir.rename(target)
    finally:
        if clone_dir.exists():
            shutil.rmtree(clone_dir, ignore_errors=True)

    logger.info("Downloaded %d tasks to %s", sum(1 for d in target.iterdir() if d.is_dir()), target)
    return target
