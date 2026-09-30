"""Resolve and cache benchmark task datasets.

Datasets are referenced with two fields (inspired by Vercel's project config):

    source:
      repo: org/repo          # GitHub repository (org/repo)
      path: sub/dir           # optional subpath within the repo
      ref: main               # optional branch/tag (default: repo default)

The repo is cloned once into ``.cache/datasets/org/repo/`` and reused on
subsequent calls. A source with a ``path`` fetches only that path: the clone
downloads no file contents up front (``--filter=blob:none``) and checks out
the path alone (``--sparse``), so one SkillsBench task costs a few megabytes,
not the whole repository. Each later path is added to the same checkout. A
path whose directory holds no BenchFlow task (a foreign benchmark that a
source adapter converts from files elsewhere in the repo), a symlinked path,
a source with no path, and git older than 2.26 get the whole repository, as
before.
"""

import difflib
import functools
import logging
import os
import posixpath
import re
import shutil
import subprocess
import sys
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
            os.write(fd, str(os.getpid()).encode("ascii"))
        except FileExistsError as e:
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
    # Through _git_quiet: in a blob-less clone the checkout fetches file
    # contents on demand, which prints progress on a terminal despite --quiet.
    _git_quiet(repo_root, "fetch", "--quiet", "--depth", "1", "origin", ref)
    _git_quiet(repo_root, "checkout", "--quiet", "--detach", "FETCH_HEAD")


def _read_resolved_sha(repo_root: Path, repo: str) -> str:
    resolved_sha = _git_stdout(repo_root, "rev-parse", "HEAD")
    if not resolved_sha:
        raise RuntimeError(f"Unable to read resolved git SHA for {repo}")
    return resolved_sha


def _snapshot_repo_root(
    repo_root: Path,
    *,
    org: str,
    repo_name: str,
    resolved_sha: str,
    path: str | None = None,
) -> Path:
    """The checkout of ``resolved_sha`` that runs read, one per commit.

    It holds ``path`` when ``repo_root`` is a sparse checkout, and the whole
    tree otherwise.
    """
    snapshot = _cache_dir() / org / f"{repo_name}__snapshots" / resolved_sha
    if (snapshot / ".git").exists():
        _widen_sparse_checkout(snapshot, path)
        return snapshot
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    sparse = _is_sparse_checkout(repo_root)
    cmd = [
        "git",
        "-C",
        str(repo_root),
        "worktree",
        "add",
        # Same reason as the clone's --quiet: "Preparing worktree" and
        # the "Updating files: n%" checkout progress are console noise.
        "--quiet",
        "--detach",
    ]
    if sparse:
        # Check out nothing yet: git before 2.36 does not copy the sparse
        # patterns into a new worktree, so a plain add would fetch and
        # check out the whole repository.
        cmd.append("--no-checkout")
    cmd.extend([str(snapshot), resolved_sha])
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        if (snapshot / ".git").exists():
            return snapshot
        raise
    if sparse:
        sparse_path = _sparse_source_path(path)
        _git_quiet(
            snapshot,
            "sparse-checkout",
            "set",
            "--",
            *([sparse_path] if sparse_path else []),
        )
        _git_quiet(snapshot, "read-tree", "-mu", "HEAD")
    return snapshot


def _repo_url(org: str, repo: str) -> str:
    return f"https://github.com/{org}/{repo}.git"


@functools.cache
def _git_can_sparse() -> bool:
    """Whether git has ``sparse-checkout add`` (2.26+); older git clones fully.

    Assumes it does when the version cannot be read.
    """
    try:
        out = subprocess.run(
            ["git", "version"], capture_output=True, text=True, check=False
        ).stdout
    except Exception:
        # No readable version: the clone itself reports what is wrong with git.
        return True
    match = re.search(r"(\d+)\.(\d+)", out or "")
    if match is None:
        return True
    return (int(match.group(1)), int(match.group(2))) >= (2, 26)


def _git_quiet(root: Path, *args: str) -> None:
    """Run a git command in ``root`` without its progress output.

    A blob-less clone fetches file contents on demand, and git shows that
    fetch's progress on a terminal ("Receiving objects: ...") whatever the
    command's own flags. Its stderr is therefore captured, and replayed only
    when the command fails.
    """
    cmd = ["git", "-C", str(root), *args]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, cmd, stderr=result.stderr
        )


def _sparse_source_path(path: str | None) -> str | None:
    """The repo-relative directory a sparse checkout should hold, or None.

    None when ``path`` is empty or cannot name a directory inside the repo
    (absolute, climbing out with ``..``, or under ``.git``): those are left
    to ``_resolve_repo_path``, which reports them.
    """
    if not path or path.startswith("/"):
        return None
    normalized = posixpath.normpath(path)
    parts = normalized.split("/")
    if normalized == "." or parts[0] == ".." or ".git" in parts:
        return None
    return normalized


def _is_sparse_checkout(root: Path) -> bool:
    """Whether the checkout at ``root`` is sparse (made for a source path)."""
    git_path = root / ".git"
    if not git_path.exists():
        return False
    if git_path.is_dir() and not (git_path / "info" / "sparse-checkout").is_file():
        # A full clone, including every cache made before sparse clones.
        return False
    return _git_stdout(root, "config", "--bool", "core.sparseCheckout") == "true"


_SYMLINK_MODE = "120000"


def _tree_entry(root: Path, path: str) -> tuple[str, str] | None:
    """``(mode, type)`` of ``path`` in HEAD's tree, or None when it is absent.

    Reads only trees, which a blob-less clone holds for the whole repo.
    """
    listing = _git_stdout(root, "ls-tree", "-z", "HEAD", "--", path)
    for record in (listing or "").split("\0"):
        meta, _, name = record.partition("\t")
        fields = meta.split()
        if name == path and len(fields) >= 2:
            return fields[0], fields[1]
    return None


def _holds_native_tasks(directory: Path) -> bool:
    """Whether ``directory`` is, or directly holds, BenchFlow task packages."""
    from benchflow.task.discovery import (
        contains_immediate_task,
        is_task_dir,
        resolve_task_collection_root,
    )

    root = resolve_task_collection_root(directory)
    return is_task_dir(root) or contains_immediate_task(root)


def _widen_sparse_checkout(root: Path, path: str | None) -> None:
    """Make a sparse checkout at ``root`` hold ``path``; a full one is left as is.

    ``path`` None asks for the whole repository, and so do a path through a
    symlink and a directory that holds no BenchFlow task (a foreign
    benchmark, which a source adapter converts from files elsewhere in the
    repo): those get the full checkout that every source had before sparse
    clones. A path that is missing or is a file is left to
    ``_resolve_repo_path`` to report.
    """
    if not _is_sparse_checkout(root):
        return
    if path is None or posixpath.normpath(path) == ".":
        _check_out_everything(root)
        return
    sparse_path = _sparse_source_path(path)
    if sparse_path is None:
        return
    parts = sparse_path.split("/")
    prefixes = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
    entries = [_tree_entry(root, prefix) for prefix in prefixes]
    if any(entry is not None and entry[0] == _SYMLINK_MODE for entry in entries):
        _check_out_everything(root)
        return
    if entries[-1] is None or entries[-1][1] != "tree":
        return
    # Only the two calls below write to the working tree, so only they are
    # skipped for a path the patterns already hold (another process may be
    # reading it). The checks around them read the tree or the index, and the
    # native-task verdict has to run either way: a foreign folder that happens
    # to sit under an already-added pattern still needs the whole repository,
    # because its source adapter reads files outside the path.
    if not _sparse_covers(root, sparse_path):
        _git_quiet(root, "sparse-checkout", "add", "--", sparse_path)
        # The clone checks out nothing (--no-checkout), so fill the working
        # tree from HEAD under the new patterns; on a filled one this changes
        # nothing.
        _git_quiet(root, "read-tree", "-mu", "HEAD")
    if not _holds_native_tasks(root / sparse_path):
        _check_out_everything(root)


def _sparse_covers(root: Path, sparse_path: str) -> bool:
    """Whether the checkout's patterns already hold ``sparse_path`` whole.

    Checking first keeps a resolve of an already fetched path from touching
    the checkout at all (another process may be reading it).
    """
    listing = _git_stdout(root, "sparse-checkout", "list") or ""
    for pattern in listing.splitlines():
        folder = pattern.strip().strip("/")
        if not folder or folder.startswith("!") or "*" in folder:
            continue
        if sparse_path == folder or sparse_path.startswith(f"{folder}/"):
            return True
    return False


def _check_out_everything(root: Path) -> None:
    """Turn a sparse checkout into a full one, as clones were before."""
    _git_quiet(root, "sparse-checkout", "disable")
    _git_quiet(root, "read-tree", "-mu", "HEAD")


def _clone_repo_unlocked(
    org: str, repo: str, ref: str | None = None, path: str | None = None
) -> Path:
    cache = _cache_dir() / org / repo
    if cache.exists() and (cache / ".git").exists():
        if ref:
            logger.info("Refreshing %s/%s at %s", org, repo, ref)
            _checkout_fetched_ref(cache, ref)
        _widen_sparse_checkout(cache, path)
        return cache

    url = _repo_url(org, repo)
    sparse = _sparse_source_path(path) is not None and _git_can_sparse()
    if sparse:
        logger.info("Fetching %s from %s/%s (%s) ...", path, org, repo, url)
    else:
        logger.info("Cloning %s/%s from %s ...", org, repo, url)
    cache.parent.mkdir(parents=True, exist_ok=True)
    clone_tmp = cache.parent / f"_{repo}_clone"

    try:
        if clone_tmp.exists():
            shutil.rmtree(clone_tmp)
        # --quiet: raw clone progress ("remote: Counting objects: …") on the
        # console pre-dashboard is noise; the "Cloning …" log line above is
        # the one-line summary. Errors still print (git keeps stderr on fail).
        cmd = ["git", "clone", "--quiet", "--depth", "1"]
        if sparse:
            # Only what the path needs: no file contents up front, and no
            # checkout until _widen_sparse_checkout has added the path (a
            # checkout here would fetch the top-level files with progress).
            cmd.extend(["--filter=blob:none", "--sparse", "--no-checkout"])
        if ref and not _looks_like_commit_sha(ref):
            cmd.extend(["--branch", ref])
        cmd.extend([url, str(clone_tmp)])
        subprocess.run(cmd, check=True)
        if ref and _looks_like_commit_sha(ref):
            _checkout_fetched_ref(clone_tmp, ref)
        if sparse:
            _widen_sparse_checkout(clone_tmp, path)
        if cache.exists():
            shutil.rmtree(cache)
        clone_tmp.rename(cache)
    finally:
        if clone_tmp.exists():
            shutil.rmtree(clone_tmp, ignore_errors=True)

    return cache


def _clone_repo(
    org: str, repo: str, ref: str | None = None, path: str | None = None
) -> Path:
    """Clone a GitHub repo into the cache if not already present.

    If the cache exists but is on a different ref, fetches and checks out
    the requested ref. With ``path``, only that path is fetched (see the
    module docstring).

    Returns the path to the cloned repo root.
    """
    with _repo_cache_lock(org, repo):
        return _clone_repo_unlocked(org, repo, ref, path)


def _resolve_repo_path(root: Path, path: str, repo_label: str) -> Path:
    requested = Path(path)
    if requested.is_absolute():
        raise ValueError(
            f"Source path {path!r} must be relative to repository root for {repo_label}"
        )
    # ``.git`` is the clone's VCS metadata, not a task source. It exists and is a
    # directory inside the root, so it would otherwise pass the checks below and
    # then resolve to zero task hashes downstream. Reject it (and anything under
    # it) at the boundary with a clear message instead.
    if ".git" in requested.parts:
        raise ValueError(
            f"Source path {path!r} must resolve to a task directory inside "
            f"{repo_label}; .git is the clone metadata, not a task source"
        )
    sparse = _is_sparse_checkout(root)
    normalized = _sparse_source_path(path)
    if sparse and normalized:
        # `docs/../tasks/x` needs docs/ on disk, which a sparse checkout skips.
        requested = Path(normalized)
    target = root / requested
    if not target.exists():
        entry = _tree_entry(root, normalized) if sparse and normalized else None
        if entry is not None and entry[1] == "blob":
            # A sparse checkout leaves the file out; it is still not a task source.
            raise ValueError(
                f"Source path {path!r} must resolve to a directory inside "
                f"{repo_label}, not a file"
            )
        raise FileNotFoundError(
            _missing_path_message(root, path, repo_label, sparse=sparse)
        )
    root_resolved = root.resolve(strict=True)
    target_resolved = target.resolve(strict=True)
    if target_resolved != root_resolved and not target_resolved.is_relative_to(
        root_resolved
    ):
        raise ValueError(
            f"Source path {path!r} escapes repository root for {repo_label}"
        )
    git_metadata = root_resolved / ".git"
    if target_resolved == git_metadata or target_resolved.is_relative_to(git_metadata):
        raise ValueError(
            f"Source path {path!r} must resolve to a task directory inside "
            f"{repo_label}; .git is the clone metadata, not a task source"
        )
    # A regular file (e.g. ``README.md``) exists and stays within the root, but a
    # task source must be a directory. Reject non-directories with a friendly
    # message rather than letting task resolution fail with zero hashes later.
    if not target_resolved.is_dir():
        raise ValueError(
            f"Source path {path!r} must resolve to a directory inside "
            f"{repo_label}, not a file"
        )
    return target_resolved


_MAX_LISTED_ENTRIES = 12


def _directory_entries(root: Path, rel: str, *, sparse: bool) -> list[str] | None:
    """Directory names directly under ``rel`` (``""`` is the root), or None.

    A sparse checkout reads them from the tree, since most directories are
    not checked out.
    """
    if sparse:
        args = ["ls-tree", "-z", "-d", "--name-only", "HEAD"]
        if rel:
            args += ["--", f"{rel}/"]
        listing = _git_stdout(root, *args)
        if listing is None or (rel and not listing):
            return None
        return sorted(posixpath.basename(name) for name in listing.split("\0") if name)
    directory = root / rel if rel else root
    if not directory.is_dir():
        return None
    return sorted(p.name for p in directory.iterdir() if p.is_dir())


def _missing_path_message(
    root: Path, path: str, repo_label: str, *, sparse: bool
) -> str:
    """List the deepest existing folder on ``path``'s way, and suggest a close match."""
    normalized = _sparse_source_path(path)
    parts = normalized.split("/") if normalized else []
    parent_parts = parts[:-1]
    entries: list[str] | None = None
    while True:
        entries = _directory_entries(root, "/".join(parent_parts), sparse=sparse)
        if entries is not None or not parent_parts:
            break
        parent_parts.pop()
    names = [name for name in entries or [] if name != ".git"]
    prefix = "/".join(parent_parts)
    shown = [f"{prefix}/{name}" if prefix else name for name in names]
    listing = shown[:_MAX_LISTED_ENTRIES]
    more = len(shown) - len(listing)
    message = f"Path {path!r} not found in {repo_label}."
    wanted = parts[len(parent_parts)] if len(parts) > len(parent_parts) else ""
    close = difflib.get_close_matches(wanted, names, n=1) if wanted else []
    if close:
        guess = f"{prefix}/{close[0]}" if prefix else close[0]
        message += f" Did you mean {guess!r}?"
    message += f" Available: {listing}"
    if more:
        message += f" and {more} more"
    return message


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
    root = _clone_repo(org, repo_name, ref, path)

    if path:
        return _resolve_repo_path(root, path, f"{org}/{repo_name}")
    return root


def _git_stdout(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        # `ls-tree -z` hands back raw path bytes rather than git's ASCII
        # C-quoting, so a non-ASCII name under a non-UTF-8 LC_CTYPE would
        # raise out of here instead of being read back as a path.
        errors="surrogateescape",
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
        if rel_parts == (".benchflow-source.json",):
            continue
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
        if (local_path / "task.toml").is_file() or (local_path / "task.md").is_file()
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
        root = _clone_repo_unlocked(org, repo_name, ref, path)
        target = root
        canonical_source_path = ""
        if path:
            target = _resolve_repo_path(root, path, f"{org}/{repo_name}")
            canonical_source_path = target.relative_to(
                root.resolve(strict=True)
            ).as_posix()
            # `.` is a spelling of the whole repository, like passing no path
            # at all; Path.relative_to renders that as ".". Record it the same
            # way as a pathless source, so two spellings of one source do not
            # produce different provenance or different per-task paths.
            if canonical_source_path == ".":
                canonical_source_path = ""
        resolved_sha = _read_resolved_sha(root, f"{org}/{repo_name}")
        snapshot_root = _snapshot_repo_root(
            root,
            org=org,
            repo_name=repo_name,
            resolved_sha=resolved_sha,
            path=canonical_source_path or None,
        )
        target = (
            snapshot_root / canonical_source_path
            if canonical_source_path
            else snapshot_root
        )
        # Under the lock: another resolve may add a path to this sparse
        # snapshot, and git status must not see that half written.
        provenance = _source_provenance(
            repo=f"{org}/{repo_name}",
            requested_ref=ref,
            source_path=canonical_source_path,
            local_path=target,
            repo_root=snapshot_root,
        )

    return ResolvedSource(path=target, provenance=provenance)


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
    """Infer github source provenance for tasks under repo or dataset cache paths.

    Best effort: a task whose files cannot be hashed (a symlinked
    ``environment/`` shared with another task, say) gets no inferred
    provenance instead of an error that would abort the whole batch.
    """
    try:
        return _infer_task_source_provenance(task_path)
    except ValueError as exc:
        if "symlink" not in str(exc):
            raise
        logger.warning("No source provenance recorded for %s: %s", task_path, exc)
        return None


def _infer_task_source_provenance(task_path: Path) -> dict[str, Any] | None:
    try:
        task_resolved = task_path.resolve(strict=True)
    except OSError:
        return None

    try:
        from benchflow._utils.hf_datasets import load_source_sidecar

        sidecar_source = load_source_sidecar(task_resolved)
    except ImportError:
        sidecar_source = None
    if sidecar_source is not None:
        return task_source_provenance(sidecar_source, task_resolved)

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

    # Non-snapshot resolve_source() cache layout:
    # .cache/datasets/<org>/<repo>/<subpath...>. Attribute provenance to the
    # cached benchmark repo's git remote/HEAD instead of falling through to
    # the BenchFlow worktree (#492).
    if cache_rel is not None and len(cache_rel.parts) >= 2:
        org, repo_name, *rest = cache_rel.parts
        if not repo_name.endswith("__snapshots") and not repo_name.startswith("_"):
            cached_repo_root = cache_root / org / repo_name
            if (cached_repo_root / ".git").exists():
                resolved_sha = _git_stdout(cached_repo_root, "rev-parse", "HEAD")
                if resolved_sha:
                    source_path = "/".join(rest)
                    rel_for_status = source_path or "."
                    status = _git_stdout(
                        cached_repo_root,
                        "status",
                        "--porcelain",
                        "--",
                        rel_for_status,
                    )
                    return {
                        "type": "github",
                        "repo": f"{org}/{repo_name}",
                        "requested_ref": _git_stdout(
                            cached_repo_root,
                            "rev-parse",
                            "--abbrev-ref",
                            "HEAD",
                        ),
                        "resolved_sha": resolved_sha,
                        "path": source_path,
                        "local_path": str(task_resolved),
                        "dirty": bool(status),
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


# Benchmark aliases

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


def ensure_tasks(benchmark: str) -> Path:
    """Clone task repo if not present. Supports aliases and org/repo strings."""
    if benchmark in TASK_ALIASES:
        org_repo, ref, path = TASK_ALIASES[benchmark]
        return resolve_source(org_repo, path, ref)
    return resolve_source(benchmark)
