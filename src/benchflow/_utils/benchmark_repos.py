"""Resolve and cache benchmark task datasets.

Datasets are referenced with two fields (inspired by Vercel's project config):

    source:
      repo: org/repo          # GitHub repository (org/repo)
      path: sub/dir           # optional subpath within the repo
      ref: main               # optional branch/tag (default: repo default)

The repo is cloned once into ``.cache/datasets-v2/org/repo/`` and reused on
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
import fcntl
import functools
import logging
import os
import posixpath
import re
import shutil
import subprocess
import time
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from benchflow.errors import UserError

logger = logging.getLogger(__name__)
_LOCK_TIMEOUT_SEC = 300.0
_CACHE_LOCK_FDS: ContextVar[tuple[int, ...]] = ContextVar("cache_lock_fds", default=())


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


_FULL_COMMIT_ID = re.compile(r"[0-9a-f]{40}")


def _checked_out_at(repo_root: Path, ref: str) -> bool:
    """True when ``ref`` is a full commit id and ``repo_root``'s HEAD is it."""
    if not _FULL_COMMIT_ID.fullmatch(ref):
        return False
    return _git_stdout(repo_root, "rev-parse", "HEAD") == ref


def _cache_dir() -> Path:
    """The cache owned by the protocol whose Git children inherit flock.

    Pre-fix children may still mutate the old ``datasets`` tree after their
    parent dies. Neither its PID file nor a free flock proves they stopped.
    Keep all new locks, clones, staging trees and snapshots separate; never
    migrate or reclaim the old namespace during an upgrade.
    """
    return _repo_root() / ".cache" / "datasets-v2"


# One folder name of `org/repo`: what GitHub allows in either, plus `_`.
_REPO_PART_RE = re.compile(r"[A-Za-z0-9_.-]+")


def _check_repo_parts(org: str, repo: str) -> None:
    """Refuse an ``org`` or ``repo`` that is not one plain folder name.

    Both become folders under the cache, and a dataset registry is
    downloaded, untrusted data. ``../..`` made the cache the git root of
    the user's own project, so the refresh fetched and checked out their
    origin's default branch there and ran their files as tasks.
    """
    for part in (org, repo):
        if (
            not _REPO_PART_RE.fullmatch(part)
            or part in {".", ".."}
            or part.startswith("-")
        ):
            name = f"{org}/{repo}"
            raise ValueError(
                f"Invalid repo {name!r}: expected 'org/repo' in GitHub name "
                f"characters (e.g. 'benchflow-ai/benchmarks')."
            )


def _make_cache_dir(cache: Path) -> None:
    """Create the source cache, keeping a new one out of the enclosing git repo.

    The cache sits at that repository's root, so one ``git add .`` committed
    the datasets (SkillsBench's whole task folder is about 1.4 GB;
    DF-p1-first-run-15). When BenchFlow creates the cache folder here, it
    writes a ``.gitignore`` holding ``*``, as pytest and ruff do in theirs,
    which ignores the folder and itself. A folder that already exists, made
    by an earlier BenchFlow or by the user, is never written into.
    """
    try:
        cache.mkdir(parents=True)
    except FileExistsError:
        return
    marker = cache / ".gitignore"
    with suppress(OSError):
        marker.write_text(
            "# Created by BenchFlow: its source cache stays out of git.\n*\n"
        )


def _cache_path(*parts: str) -> Path:
    """``_cache_dir()/<parts>``, refused when it resolves outside the cache.

    Names are checked by ``_check_repo_parts``; this also stops a symlink
    planted inside the cache from sending a clone, a lock or a snapshot to a
    folder BenchFlow does not own.
    """
    base = _cache_dir()
    path = base.joinpath(*parts)
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"Source cache path {path} resolves outside {base}")
    return path


def _lock_holder(lock_path: Path) -> str:
    """The pid a lock file names, for the timeout message; "" when unreadable."""
    with suppress(OSError, UnicodeDecodeError):
        text = lock_path.read_text(encoding="ascii").strip()
        if text.isdigit():
            return f" (held by pid {text})"
    return ""


def _dead_lock_owner(fd: int) -> bool:
    """Only a proven-dead PID permits takeover of a legacy O_EXCL claim."""
    try:
        value = os.pread(fd, 64, 0).decode("ascii").strip()
        if not value.isdigit() or int(value) <= 0:
            return False
        os.kill(int(value), 0)
    except ProcessLookupError:
        return True
    except (OSError, UnicodeDecodeError, ValueError, OverflowError):
        pass
    return False


@contextmanager
def _repo_cache_lock(org: str, repo: str):
    """Serialize mutation of the shared git checkout for one repo.

    Exclusive creation respects older BenchFlow's PID-file claims, including
    the interval before their PID is written. An existing claim is reclaimed
    only when its PID is proven dead AND its flock is free. Git children
    inherit the flock, so parent death alone cannot admit a second mutator.
    """
    _check_repo_parts(org, repo)
    _make_cache_dir(_cache_dir())
    lock_dir = _cache_path(org)
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f".{repo}.lock"
    deadline = time.monotonic() + _LOCK_TIMEOUT_SEC
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
            created = True
        except FileExistsError:
            try:
                fd = os.open(lock_path, os.O_RDWR)
            except FileNotFoundError:
                continue
            created = False
        try:
            # Do not briefly flock an empty, newly created claim ahead of
            # its creator. It has not written its PID or acquired flock yet.
            if not created and not _dead_lock_owner(fd):
                raise BlockingIOError("live or indeterminate legacy claim")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not created and not _dead_lock_owner(fd):
                raise BlockingIOError("live or indeterminate legacy claim")
        except BlockingIOError as e:
            os.close(fd)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for source cache lock {lock_path}"
                    f"{_lock_holder(lock_path)}"
                ) from e
            time.sleep(0.1)
            continue
        except BaseException:
            os.close(fd)
            raise
        try:
            held, named = os.fstat(fd), os.stat(lock_path)
            current = (held.st_dev, held.st_ino) == (named.st_dev, named.st_ino)
        except FileNotFoundError:
            current = False
        except BaseException:
            os.close(fd)
            raise
        if current:
            break
        os.close(fd)
    token = _CACHE_LOCK_FDS.set((*_CACHE_LOCK_FDS.get(), fd))
    try:
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode("ascii"))
        yield
    finally:
        try:
            with suppress(FileNotFoundError):
                held, named = os.fstat(fd), os.stat(lock_path)
                if (held.st_dev, held.st_ino) == (named.st_dev, named.st_ino):
                    lock_path.unlink()
        finally:
            _CACHE_LOCK_FDS.reset(token)
            os.close(fd)


def _looks_like_commit_sha(ref: str) -> bool:
    # 40 hex digits for SHA-1 repositories, 64 for SHA-256 ones.
    return len(ref) in (40, 64) and all(ch in "0123456789abcdefABCDEF" for ch in ref)


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
    snapshot = _cache_path(org, f"{repo_name}__snapshots", resolved_sha)
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
        subprocess.run(cmd, check=True, pass_fds=_CACHE_LOCK_FDS.get())
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
        if sparse_path:
            _add_compose_build_contexts(snapshot, sparse_path)
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
        pass_fds=_CACHE_LOCK_FDS.get(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    if result.returncode != 0:
        # Git's last line goes in the error (GitCommandError), not the raw
        # stream: the CLI prints it once, with the next step.
        hint = _git_said(result.stderr, result.returncode)[1]
        if (
            args
            and args[0] == "fetch"
            and any(
                marker in (result.stderr or "").lower()
                for marker in ("couldn't find remote ref", "not our ref")
            )
        ):
            hint = _MISSING_REF_NEXT
        error_type = GitRemoteCommandError if hint else GitCommandError
        error = error_type(result.returncode, cmd, stderr=result.stderr)
        error.hint = hint
        raise error


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
        return
    _add_compose_build_contexts(root, sparse_path)


_COMPOSE_NAMES = (
    "docker-compose.yaml",
    "docker-compose.yml",
    "compose.yaml",
    "compose.yml",
)


def _compose_build_contexts(text: str) -> list[str]:
    """Read long, short and flow-style build contexts, including Compose tags.

    Nodes preserve !override/!reset without constructing tagged objects or
    interpreting interpolation. Compose still validates the document itself.
    """
    import yaml

    def field(
        node: yaml.Node | None, name: str, seen: frozenset[int] = frozenset()
    ) -> yaml.Node | None:
        if not isinstance(node, yaml.MappingNode) or id(node) in seen:
            return None
        for key, value in node.value:
            if isinstance(key, yaml.ScalarNode) and key.value == name:
                return value
        # YAML merges can supply the build/context fields through an anchor.
        # Follow only mapping nodes and bound cycles without constructing values.
        for key, value in node.value:
            if key.tag != "tag:yaml.org,2002:merge":
                continue
            merged = value.value if isinstance(value, yaml.SequenceNode) else [value]
            for item in merged:
                inherited = field(item, name, seen | {id(node)})
                if inherited is not None:
                    return inherited
        return None

    try:
        services = field(yaml.compose(text, Loader=yaml.SafeLoader), "services")
    except yaml.YAMLError:
        return []
    names: dict[str, None] = {}
    pending = [services]
    seen: set[int] = set()
    while pending:
        service_map = pending.pop()
        if not isinstance(service_map, yaml.MappingNode) or id(service_map) in seen:
            continue
        seen.add(id(service_map))
        for key, service in service_map.value:
            if key.tag == "tag:yaml.org,2002:merge":
                pending.extend(
                    service.value
                    if isinstance(service, yaml.SequenceNode)
                    else [service]
                )
            elif isinstance(key, yaml.ScalarNode):
                names.setdefault(key.value, None)
    contexts: list[str] = []
    for name in names:
        build = field(field(services, name), "build")
        context = (
            field(build, "context") if isinstance(build, yaml.MappingNode) else build
        )
        if (
            isinstance(context, yaml.ScalarNode)
            and context.tag == "tag:yaml.org,2002:str"
        ):
            contexts.append(context.value)
    return contexts


def _add_compose_build_contexts(root: Path, sparse_path: str) -> None:
    """Check out the build contexts that tasks under ``sparse_path`` name outside it.

    A task's compose file may build a service from a folder the dataset
    shares with others (robohub's ``context: ../../../../runtimes/base``).
    A sparse checkout of the dataset alone leaves that folder out, and every
    trial then fails in ``docker compose build`` with "unable to prepare
    context". YAML nodes handle Compose tags without interpreting them; a
    context that leaves the repository, or is a URL, is left to compose.
    """
    base = (root / sparse_path).resolve()
    repo = root.resolve()
    extra: set[str] = set()
    for name in _COMPOSE_NAMES:
        for compose in base.rglob(name):
            try:
                text = compose.read_text(errors="replace")
            except OSError:
                continue
            for raw in _compose_build_contexts(text):
                value = raw.strip()
                if not value or "://" in value or value.startswith(("$", "/")):
                    continue
                target = (compose.parent / value).resolve()
                if not target.is_relative_to(repo) or target.is_relative_to(base):
                    continue
                rel = target.relative_to(repo).as_posix()
                if _sparse_source_path(rel) is not None:
                    extra.add(rel)
    missing = sorted(p for p in extra if not _sparse_covers(root, p))
    if missing:
        _git_quiet(root, "sparse-checkout", "add", "--", *missing)
        _git_quiet(root, "read-tree", "-mu", "HEAD")


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


# git's own words for a repository it cannot see: GitHub answers an unknown
# or private repository with an authentication request, which git turns into
# a credential prompt (or, with no terminal, "could not read Username").
_REPO_NOT_FOUND_MARKERS = (
    "repository not found",
    "could not read username",
    "terminal prompts disabled",
    "authentication failed",
)
# What git says when this machine cannot reach the remote at all.
_OFFLINE_MARKERS = (
    "could not resolve host",
    "failed to connect",
    "network is unreachable",
    "connection timed out",
    "temporary failure in name resolution",
)
_OFFLINE_NEXT = (
    "check this machine's network (or its proxy), then run the command again"
)
_MISSING_REF_NEXT = (
    "check the source ref (branch, tag or commit) exists in the remote repository"
)


def _git_said(stderr: object, returncode: int) -> tuple[str, str | None]:
    """Git's last line, said plainly, and the next step when it was offline."""
    said = str(stderr or "").strip()
    last = said.splitlines()[-1] if said else f"git exited with {returncode}"
    if any(marker in said.lower() for marker in _OFFLINE_MARKERS):
        return f"could not reach GitHub (no network?): {last}", _OFFLINE_NEXT
    return last, None


class GitCommandError(subprocess.CalledProcessError):
    """A git command failed: says git's own last line, never Python's repr of
    the command. Internal checkout and cache failures retain crash diagnostics."""

    hint: str | None = None

    def __str__(self) -> str:
        return _git_said(self.stderr, self.returncode)[0]


class GitCloneError(RuntimeError):
    """``git clone`` failed for a reason other than a missing repository."""

    hint: str | None = None


class GitRemoteCommandError(GitCommandError, UserError):
    """A connection failure, or a requested ref missing from the remote."""


class GitRemoteCloneError(GitCloneError, UserError):
    """A connection failure, or a requested branch missing from the remote."""


class RepositoryNotFoundError(ValueError, UserError):
    """A --source-repo GitHub does not show: misspelled, or private."""

    def __init__(self, org: str, repo: str) -> None:
        super().__init__(
            f"repository {org}/{repo} was not found on GitHub, or it is private"
        )
        self.hint = (
            f"check the name (https://github.com/{org}/{repo}); a private "
            "repository needs a git login on this machine (`gh auth login`)"
        )


def _run_clone(cmd: list[str], org: str, repo: str) -> None:
    """Run ``git clone`` with no credential prompt; name why it failed.

    A misspelled repository used to print git's ``fatal: could not read
    Username`` and a CalledProcessError repr with the whole argv
    (DF-p1-first-run-7). Now it is a :class:`RepositoryNotFoundError`, and any
    other failure is git's last line.
    """
    try:
        subprocess.run(
            cmd,
            check=True,
            pass_fds=_CACHE_LOCK_FDS.get(),
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
    except subprocess.CalledProcessError as exc:
        said = str(exc.stderr or "").strip()
        if any(marker in said.lower() for marker in _REPO_NOT_FOUND_MARKERS):
            raise RepositoryNotFoundError(org, repo) from None
        last, hint = _git_said(said, exc.returncode)
        if "--branch" in cmd and re.search(
            r"remote branch .+ not found in upstream origin", said, re.IGNORECASE
        ):
            hint = _MISSING_REF_NEXT
        error_type = GitRemoteCloneError if hint else GitCloneError
        error = error_type(f"git could not clone {org}/{repo}: {last}")
        error.hint = hint
        raise error from exc


def _check_out_everything(root: Path) -> None:
    """Turn a sparse checkout into a full one, as clones were before."""
    _git_quiet(root, "sparse-checkout", "disable")
    _git_quiet(root, "read-tree", "-mu", "HEAD")


def _clone_repo_unlocked(
    org: str, repo: str, ref: str | None = None, path: str | None = None
) -> Path:
    _check_repo_parts(org, repo)
    if ref and ref.startswith("-"):
        # git would read it as an option (`--upload-pack=...` runs a command).
        raise ValueError(
            f"Invalid ref {ref!r} for {org}/{repo}: a ref cannot start with '-'"
        )
    cache = _cache_path(org, repo)
    if cache.exists() and (cache / ".git").exists():
        # A full commit id names fixed content: when the cache is already
        # checked out at it there is nothing to fetch. `bench agent list`
        # reads the agent catalog at a pinned commit, and fetched it from
        # GitHub on every run before this (about 0.4 s and a network round
        # trip). Branches, tags and short ids still refresh.
        if ref and not _checked_out_at(cache, ref):
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
        # "HEAD" is the remote's default branch, which a plain clone checks
        # out; git has no branch of that name to pass to --branch.
        if ref and ref != "HEAD" and not _looks_like_commit_sha(ref):
            cmd.extend(["--branch", ref])
        cmd.extend([url, str(clone_tmp)])
        _run_clone(cmd, org, repo)
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
    the requested ref; ``"HEAD"`` fetches the remote's default branch again. With ``path``, only that path is fetched (see the
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


_INVALID_REPO = (
    "Invalid repo format: {repo!r}. Expected 'org/repo' (e.g. "
    "'benchflow-ai/benchmarks') or a BenchFlow dataset ID (e.g. 'terminal-bench-2')."
)


def _dataset_id_source(
    repo: str, path: str | None, ref: str | None
) -> tuple[str, str | None, str | None]:
    """A BenchFlow dataset ID as the ``org/repo``, path and ref it stands for.

    ``path`` is read inside the ID's folder and ``ref`` overrides its pin.
    Anything else is returned unchanged.
    """
    if "/" in repo or repo not in TASK_ALIASES:
        return repo, path, ref
    org_repo, pinned, folder = TASK_ALIASES[repo]
    if folder and path:
        path = posixpath.join(folder, path)
    return org_repo, path or folder, ref or pinned


def resolve_source(repo: str, path: str | None = None, ref: str | None = None) -> Path:
    """Resolve a dataset source to a local filesystem path.

    Args:
        repo: GitHub repository as ``org/repo`` (e.g. ``benchflow-ai/benchmarks``),
            or a BenchFlow dataset ID (``terminal-bench-2``, ``skillsbench``;
            :data:`TASK_ALIASES`), which names its repository, pin and folder.
        path: Optional subpath within the repo, or within a dataset ID's
            folder (e.g. ``log-summary-date-ranges``).
        ref: Optional branch or tag to clone (e.g. ``main``, ``v2.0``).

    Returns:
        Path to the resolved directory on the local filesystem.
    """
    repo, path, ref = _dataset_id_source(repo, path, ref)
    parts = repo.split("/", 1)
    if len(parts) != 2:
        raise ValueError(_INVALID_REPO.format(repo=repo))
    org, repo_name = parts
    root = _clone_repo(org, repo_name, ref, path)

    if path:
        return _resolve_repo_path(root, path, f"{org}/{repo_name}")
    return root


def _git_stdout(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        pass_fds=_CACHE_LOCK_FDS.get(),
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
    repo, path, ref = _dataset_id_source(repo, path, ref)
    parts = repo.split("/", 1)
    if len(parts) != 2:
        raise ValueError(_INVALID_REPO.format(repo=repo))
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

    # Upgrades retain the old namespace. Recognize it for read-only audit
    # metadata, but leave resolution, locks and all mutation in datasets-v2.
    cache_rel = None
    for cache_root in (_cache_dir(), _repo_root() / ".cache" / "datasets"):
        try:
            cache_rel = task_resolved.relative_to(cache_root.resolve(strict=False))
        except ValueError:
            continue
        break

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
    # .cache/{datasets,datasets-v2}/<org>/<repo>/<subpath...>. Attribute provenance to the
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
    # Terminal-Bench 2.0 at the commit BenchFlow's registry pins (89 tasks at
    # the repository root).
    "terminal-bench-2": (
        "laude-institute/terminal-bench-2",
        "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c",
        None,
    ),
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
