"""Resolve and cache benchmark task datasets.

Datasets are referenced with two fields (inspired by Vercel's project config):

    source:
      repo: org/repo          # GitHub repository (org/repo)
      path: sub/dir           # optional subpath within the repo
      ref: main               # optional branch/tag (default: repo default)

The repo is cloned once into ``.cache/datasets/org/repo/`` and reused on
subsequent calls.
"""

import logging
import os
import shutil
import subprocess
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_LOCK_TIMEOUT_SEC = 300.0


@dataclass(frozen=True, slots=True)
class Source:
    """A benchmark dataset source — identifies a repo and optional subpath."""

    repo: str
    path: str | None = None
    ref: str | None = None

    def resolve(self) -> Path:
        """Clone the repo (if needed) and return the local filesystem path."""
        return resolve_source(self.repo, self.path, self.ref)


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    """A resolved source path plus reproducible audit metadata."""

    path: Path
    provenance: dict[str, Any]


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


def _lock_is_stale(lock_path: Path) -> bool:
    try:
        pid = int(lock_path.read_text().strip())
    except (OSError, ValueError):
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    return False


@contextmanager
def _repo_cache_lock(org: str, repo: str):
    """Serialize mutation of the shared git checkout for one repo."""
    lock_dir = _cache_dir() / org
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f".{repo}.lock"
    deadline = time.monotonic() + _LOCK_TIMEOUT_SEC
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode("ascii"))
            except OSError:
                os.close(fd)
                fd = None
                with suppress(FileNotFoundError):
                    lock_path.unlink()
                raise
        except FileExistsError as e:
            if _lock_is_stale(lock_path):
                with suppress(FileNotFoundError):
                    lock_path.unlink()
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for source cache lock {lock_path}"
                ) from e
            time.sleep(0.1)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        with suppress(FileNotFoundError):
            lock_path.unlink()


def _looks_like_commit_sha(ref: str) -> bool:
    return len(ref) == 40 and all(ch in "0123456789abcdefABCDEF" for ch in ref)


def _checkout_fetched_ref(repo_root: Path, ref: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo_root), "fetch", "--depth", "1", "origin", ref],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "checkout", "--detach", "FETCH_HEAD"],
        check=True,
    )


def _read_resolved_sha(repo_root: Path, repo: str) -> str:
    resolved_sha = _git_stdout(repo_root, "rev-parse", "HEAD")
    if not resolved_sha:
        raise RuntimeError(f"Unable to read resolved git SHA for {repo}")
    return resolved_sha


def _snapshot_repo_root(
    repo_root: Path, *, org: str, repo_name: str, resolved_sha: str
) -> Path:
    snapshot = _cache_dir() / org / f"{repo_name}__snapshots" / resolved_sha
    if (snapshot / ".git").exists():
        return snapshot
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "worktree",
                "add",
                "--detach",
                str(snapshot),
                resolved_sha,
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        if (snapshot / ".git").exists():
            return snapshot
        raise
    return snapshot


def _clone_repo_unlocked(org: str, repo: str, ref: str | None = None) -> Path:
    cache = _cache_dir() / org / repo
    if cache.exists() and (cache / ".git").exists():
        if ref:
            if _looks_like_commit_sha(ref):
                head = _git_stdout(cache, "rev-parse", "HEAD")
                if head == ref:
                    return cache
            logger.info("Refreshing %s/%s at %s", org, repo, ref)
            _checkout_fetched_ref(cache, ref)
        return cache

    url = f"https://github.com/{org}/{repo}.git"
    logger.info("Cloning %s/%s from %s ...", org, repo, url)
    cache.parent.mkdir(parents=True, exist_ok=True)
    clone_tmp = cache.parent / f"_{repo}_clone"

    try:
        if clone_tmp.exists():
            shutil.rmtree(clone_tmp)
        cmd = ["git", "clone", "--depth", "1"]
        if ref and not _looks_like_commit_sha(ref):
            cmd.extend(["--branch", ref])
        cmd.extend([url, str(clone_tmp)])
        subprocess.run(cmd, check=True)
        if ref and _looks_like_commit_sha(ref):
            _checkout_fetched_ref(clone_tmp, ref)
        if cache.exists():
            shutil.rmtree(cache)
        clone_tmp.rename(cache)
    finally:
        if clone_tmp.exists():
            shutil.rmtree(clone_tmp, ignore_errors=True)

    return cache


def _clone_repo(org: str, repo: str, ref: str | None = None) -> Path:
    """Clone a GitHub repo into the cache if not already present.

    If the cache exists but is on a different ref, fetches and checks out
    the requested ref.

    Returns the path to the cloned repo root.
    """
    with _repo_cache_lock(org, repo):
        return _clone_repo_unlocked(org, repo, ref)


def _resolve_repo_path(root: Path, path: str, repo_label: str) -> Path:
    requested = Path(path)
    if requested.is_absolute():
        raise ValueError(
            f"Source path {path!r} must be relative to repository root for {repo_label}"
        )
    target = root / requested
    if not target.exists():
        raise FileNotFoundError(
            f"Path {path!r} not found in {repo_label}. "
            f"Available: {[p.name for p in root.iterdir() if p.is_dir() and p.name != '.git']}"
        )
    root_resolved = root.resolve(strict=True)
    target_resolved = target.resolve(strict=True)
    if target_resolved != root_resolved and not target_resolved.is_relative_to(
        root_resolved
    ):
        raise ValueError(
            f"Source path {path!r} escapes repository root for {repo_label}"
        )
    return target_resolved


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
        return _resolve_repo_path(root, path, f"{org}/{repo_name}")
    return root


def _git_stdout(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def task_file_hashes(task_path: Path) -> dict[str, str]:
    """Return deterministic SHA-256 hashes for regular files under a task dir."""
    if task_path.is_symlink():
        raise ValueError(f"Task path {task_path} must not be a symlink")
    hashes: dict[str, str] = {}
    for path in sorted(task_path.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Task path {task_path} contains symlink {path}")
        if not path.is_file():
            continue
        rel_parts = path.relative_to(task_path).parts
        if ".git" in rel_parts or "__pycache__" in rel_parts:
            continue
        rel = Path(*rel_parts).as_posix()
        hashes[rel] = f"sha256:{sha256(path.read_bytes()).hexdigest()}"
    return hashes


def _source_provenance(
    *,
    repo: str,
    requested_ref: str | None,
    source_path: str | None,
    local_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    resolved_sha = _read_resolved_sha(repo_root, repo)
    status = _git_stdout(repo_root, "status", "--porcelain")
    if status is None:
        raise RuntimeError(f"Unable to read git status for {repo} at {resolved_sha}")
    provenance: dict[str, Any] = {
        "type": "github",
        "repo": repo,
        "requested_ref": requested_ref,
        "resolved_sha": resolved_sha,
        "path": source_path or "",
        "local_path": str(local_path),
        "dirty": bool(status),
        "file_hashes": task_file_hashes(local_path)
        if (local_path / "task.toml").is_file()
        else {},
    }
    return provenance


def resolve_source_with_metadata(
    repo: str, path: str | None = None, ref: str | None = None
) -> ResolvedSource:
    """Resolve a dataset source and retain source/audit metadata."""
    parts = repo.split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid repo format: {repo!r}. Expected 'org/repo' (e.g. 'benchflow-ai/benchmarks')."
        )
    org, repo_name = parts
    with _repo_cache_lock(org, repo_name):
        root = _clone_repo_unlocked(org, repo_name, ref)
        target = root
        canonical_source_path = ""
        if path:
            target = _resolve_repo_path(root, path, f"{org}/{repo_name}")
            canonical_source_path = target.relative_to(
                root.resolve(strict=True)
            ).as_posix()
        resolved_sha = _read_resolved_sha(root, f"{org}/{repo_name}")
        snapshot_root = _snapshot_repo_root(
            root,
            org=org,
            repo_name=repo_name,
            resolved_sha=resolved_sha,
        )

    target = (
        snapshot_root / canonical_source_path
        if canonical_source_path
        else snapshot_root
    )

    return ResolvedSource(
        path=target,
        provenance=_source_provenance(
            repo=f"{org}/{repo_name}",
            requested_ref=ref,
            source_path=canonical_source_path,
            local_path=target,
            repo_root=snapshot_root,
        ),
    )


def task_source_provenance(
    source_provenance: dict[str, Any] | None, task_path: Path
) -> dict[str, Any] | None:
    """Return per-task provenance derived from a source directory provenance block."""
    if not source_provenance:
        return infer_task_source_provenance(task_path)
    provenance = dict(source_provenance)
    base_local_raw = provenance.get("local_path")
    source_path = str(provenance.get("path") or "").strip("/")
    task_source_path = source_path
    if isinstance(base_local_raw, str) and base_local_raw:
        base_local = Path(base_local_raw).resolve(strict=True)
        task_resolved = task_path.resolve(strict=True)
        if task_resolved != base_local and not task_resolved.is_relative_to(base_local):
            raise ValueError(
                f"Task path {task_path} is outside source local_path {base_local}"
            )
        try:
            rel = task_resolved.relative_to(base_local).as_posix()
        except ValueError:
            rel = ""
        if rel and rel != ".":
            task_source_path = f"{source_path}/{rel}" if source_path else rel
    provenance["path"] = task_source_path
    provenance["local_path"] = str(task_path)
    provenance["file_hashes"] = task_file_hashes(task_path)
    return provenance


def _repo_slug_from_git_root(repo_root: Path) -> str | None:
    remote = _git_stdout(repo_root, "remote", "get-url", "origin")
    if not remote:
        return None
    normalized = remote.strip()
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if normalized.startswith("git@github.com:"):
        return normalized.removeprefix("git@github.com:")
    marker = "github.com/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    return None


def infer_task_source_provenance(task_path: Path) -> dict[str, Any] | None:
    """Infer github source provenance for tasks under repo or dataset cache paths."""
    try:
        task_resolved = task_path.resolve(strict=True)
    except OSError:
        return None

    cache_root = _cache_dir()
    try:
        cache_rel = task_resolved.relative_to(cache_root.resolve(strict=False))
    except ValueError:
        cache_rel = None

    if cache_rel is not None and len(cache_rel.parts) >= 3:
        org, snapshot_dir, resolved_sha, *rest = cache_rel.parts
        if snapshot_dir.endswith("__snapshots") and _is_hex(resolved_sha):
            repo_name = snapshot_dir.removesuffix("__snapshots")
            snapshot_root = cache_root / org / snapshot_dir / resolved_sha
            source_path = "/".join(rest)
            return {
                "type": "github",
                "repo": f"{org}/{repo_name}",
                "requested_ref": None,
                "resolved_sha": resolved_sha,
                "path": source_path,
                "local_path": str(task_resolved),
                "dirty": bool(
                    _git_stdout(
                        snapshot_root,
                        "status",
                        "--porcelain",
                        "--",
                        *rest,
                    )
                ),
                "file_hashes": task_file_hashes(task_resolved),
            }

    repo_root = _repo_root()
    try:
        rel = task_resolved.relative_to(repo_root.resolve(strict=False))
    except ValueError:
        return None

    repo_slug = _repo_slug_from_git_root(repo_root)
    resolved_sha = _git_stdout(repo_root, "rev-parse", "HEAD")
    if not repo_slug or not resolved_sha:
        return None

    return {
        "type": "github",
        "repo": repo_slug,
        "requested_ref": _git_stdout(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "resolved_sha": resolved_sha,
        "path": rel.as_posix(),
        "local_path": str(task_resolved),
        "dirty": bool(
            _git_stdout(repo_root, "status", "--porcelain", "--", rel.as_posix())
        ),
        "file_hashes": task_file_hashes(task_resolved),
    }


def _is_hex(value: str) -> bool:
    return len(value) in {40, 64} and all(
        ch in "0123456789abcdef" for ch in value.lower()
    )


# ---------------------------------------------------------------------------
# Benchmark aliases
# ---------------------------------------------------------------------------

# Aliases for ensure_tasks("shortname") callers.
# Format: (org/repo, ref, subpath)
TASK_ALIASES: dict[str, tuple[str, str | None, str | None]] = {
    "skillsbench": ("benchflow-ai/skillsbench", "main", "tasks"),
    "programbench": (
        "facebookresearch/programbench",
        "main",
        "src/programbench/data/tasks",
    ),
    "harvey-lab": ("benchflow-ai/benchmarks", "main", "datasets/harvey-lab/tasks"),
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
    """Clone task repo if not present. Supports aliases and org/repo strings."""
    if benchmark in TASK_ALIASES:
        org_repo, ref, path = TASK_ALIASES[benchmark]
        return resolve_source(org_repo, path, ref)
    return resolve_source(benchmark)
