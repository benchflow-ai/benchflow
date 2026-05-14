"""Resolve and cache benchmark task datasets.

Datasets are referenced with two fields (inspired by Vercel's project config):

    source:
      repo: org/repo          # GitHub repository (org/repo)
      path: sub/dir           # optional subpath within the repo
      ref: main               # optional branch/tag (default: repo default)

The repo is cloned once into ``.cache/datasets/org/repo/`` and reused on
subsequent calls.

Generated benchmarks (e.g. ``programbench``) clone the upstream repo and
run a local generator script to produce task directories.
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Source:
    """A benchmark dataset source — identifies a repo and optional subpath."""

    repo: str
    path: str | None = None
    ref: str | None = None

    def resolve(self) -> Path:
        """Clone the repo (if needed) and return the local filesystem path."""
        return resolve_source(self.repo, self.path, self.ref)


def _repo_root() -> Path:
    """Find the repo root via .git directory."""
    d = Path.cwd()
    while d != d.parent:
        if (d / ".git").exists():
            return d
        d = d.parent
    return Path.cwd()


def _cache_dir() -> Path:
    """Return the local cache directory for cloned dataset repos."""
    return _repo_root() / ".cache" / "datasets"


def _clone_repo(org: str, repo: str, ref: str | None = None) -> Path:
    """Clone a GitHub repo into the cache if not already present.

    If the cache exists but is on a different ref, fetches and checks out
    the requested ref.

    Returns the path to the cloned repo root.
    """
    cache = _cache_dir() / org / repo
    if cache.exists() and (cache / ".git").exists():
        if ref:
            # Ensure the cached clone is on the requested ref.
            current = subprocess.run(
                ["git", "-C", str(cache), "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            if current != ref:
                logger.info("Switching %s/%s from %s to %s", org, repo, current, ref)
                subprocess.run(
                    ["git", "-C", str(cache), "fetch", "--depth", "1", "origin", ref],
                    check=True,
                )
                subprocess.run(
                    ["git", "-C", str(cache), "checkout", ref],
                    check=True,
                )
        return cache

    url = f"https://github.com/{org}/{repo}.git"
    logger.info("Cloning %s/%s from %s ...", org, repo, url)
    cache.parent.mkdir(parents=True, exist_ok=True)
    clone_tmp = cache.parent / f"_{repo}_clone"

    try:
        if clone_tmp.exists():
            shutil.rmtree(clone_tmp)
        cmd = ["git", "clone", "--depth", "1"]
        if ref:
            cmd.extend(["--branch", ref])
        cmd.extend([url, str(clone_tmp)])
        subprocess.run(cmd, check=True)
        if cache.exists():
            shutil.rmtree(cache)
        clone_tmp.rename(cache)
    finally:
        if clone_tmp.exists():
            shutil.rmtree(clone_tmp, ignore_errors=True)

    return cache


def resolve_source(repo: str, path: str | None = None, ref: str | None = None) -> Path:
    """Resolve a dataset source to a local filesystem path.

    Args:
        repo: GitHub repository as ``org/repo`` (e.g. ``benchflow-ai/benchmarks``).
        path: Optional subpath within the repo (e.g. ``terminal-bench-2``).
        ref: Optional branch or tag to clone (e.g. ``main``, ``v2.0``).

    Returns:
        Path to the resolved directory on the local filesystem.
    """
    parts = repo.split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid repo format: {repo!r}. Expected 'org/repo' (e.g. 'benchflow-ai/benchmarks')."
        )
    org, repo_name = parts
    root = _clone_repo(org, repo_name, ref)

    if path:
        target = root / path
        if not target.exists():
            raise FileNotFoundError(
                f"Path {path!r} not found in {org}/{repo_name}. "
                f"Available: {[p.name for p in root.iterdir() if p.is_dir() and p.name != '.git']}"
            )
        return target
    return root


# ---------------------------------------------------------------------------
# Generated benchmarks
# ---------------------------------------------------------------------------

# Benchmarks whose tasks are produced by a local generator script rather
# than cloned from a pre-built dataset repo.
_GENERATED_BENCHMARKS: dict[str, dict[str, str]] = {
    "programbench": {
        "repo": "facebookresearch/programbench",
        "subdir": "tasks",
    },
}


def _ensure_generated(benchmark: str) -> Path:
    """Clone upstream repo, run the generator, return the tasks directory."""
    info = _GENERATED_BENCHMARKS[benchmark]
    root = _repo_root()
    target = root / "benchmarks" / benchmark / (info.get("subdir") or "")

    if target.exists() and any(target.iterdir()):
        return target

    repo = info["repo"]
    logger.info("Generating %s tasks from %s...", benchmark, repo)
    clone_dir = root / "benchmarks" / benchmark / "_clone"
    clone_dir.parent.mkdir(parents=True, exist_ok=True)

    # Generate into a temp directory and rename atomically so partial
    # failures never leave a cached target that looks complete.
    staging = root / "benchmarks" / benchmark / "_gen_staging"
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                f"https://github.com/{repo}.git",
                str(clone_dir),
            ],
            check=True,
        )
        pb_tasks = clone_dir / "src" / "programbench" / "data" / "tasks"
        if not pb_tasks.is_dir():
            raise FileNotFoundError(
                f"ProgramBench tasks directory not found at {pb_tasks}"
            )

        import importlib
        import sys

        gen_path = root / "benchmarks" / "programbench"
        if str(gen_path.parent) not in sys.path:
            sys.path.insert(0, str(gen_path.parent))
        generate = importlib.import_module("programbench.benchflow")

        staging.mkdir(parents=True, exist_ok=True)
        generated = generate.generate_all(pb_tasks, staging)

        # Atomic swap: only expose target once generation fully succeeds
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(target)
        logger.info("Generated %d tasks to %s", len(generated), target)
    finally:
        if clone_dir.exists():
            shutil.rmtree(clone_dir, ignore_errors=True)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    return target


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

# Legacy aliases for ensure_tasks("shortname") callers.
TASK_ALIASES: dict[str, tuple[str, str | None, str | None]] = {
    "skillsbench": ("benchflow-ai/skillsbench", "main", "tasks"),
    "terminal-bench-2": ("harbor-framework/terminal-bench-2", None, None),
    "harvey-lab": ("harveyai/harvey-labs", "main", "tasks"),
}

# Old dict shape kept for imports that reference TASK_REPOS.
TASK_REPOS = {
    name: {
        "repo": f"https://github.com/{org_repo}.git",
        **({"ref": branch} if branch else {}),
        **({"subdir": subdir} if subdir else {}),
    }
    for name, (org_repo, branch, subdir) in TASK_ALIASES.items()
}


def ensure_tasks(benchmark: str) -> Path:
    """Clone task repo if not present. Supports aliases, generated benchmarks, and org/repo strings."""
    if benchmark in _GENERATED_BENCHMARKS:
        return _ensure_generated(benchmark)
    if benchmark in TASK_ALIASES:
        org_repo, ref, path = TASK_ALIASES[benchmark]
        return resolve_source(org_repo, path, ref)
    return resolve_source(benchmark)
