"""Process-wide, single-flight provisioning of AgentCore images and runtimes.

A rollout on AgentCore is a **session**, not a runtime. One agent runtime can
host many concurrent sessions, each an isolated microVM with its own
filesystem — measured at 8 concurrent sessions on one runtime, and the account
quota for *Active Session Workloads* is 5000 against only 100 *Total Agents*.
So the expensive, rate-limited artifacts (an ECR image and a registered
runtime) must be created **once per distinct image and execution contract** and
shared by every compatible rollout, while sessions are what scale out.

Getting that wrong is not merely slow. Keying a runtime on the task name meant
that three trials of one task raced to create the same runtime and the first to
finish deleted it out from under the other two. Keying on the *content* of the
build context makes the mapping deterministic: same image ⇒ same runtime ⇒ one
build, one push, one registration, N sessions.

The control plane is also far tighter than the data plane
(``CreateAgentRuntime`` and ``ListAgentRuntimes`` are 5/s, while
``InvokeAgentRuntimeCommand`` is 200/s), which is why nothing here may run per
rollout. Results are memoized for the life of the process and every miss is
funnelled through a per-key lock.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import posixpath
import re
import stat
import threading
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger("benchflow").getChild("agentcore")

# Names generated into the build context; excluded from its digest so the
# digest describes the *task*, not our own scaffolding.
GENERATED_DOCKERFILE = "Dockerfile.benchflow-agentcore"
GENERATED_SHIM = ".benchflow_agentcore_shim.py"
_GENERATED_NAMES = frozenset({GENERATED_DOCKERFILE, GENERATED_SHIM})

# Tag convention shared with the Daytona reaper so cleanup can tell BenchFlow's
# resources apart from anything else in the account.
MANAGED_TAG = "benchflow-managed"
MANAGED_VALUE = "1"
#: Tag holding an ISO-8601 instant until which a runtime must not be reaped.
#: There is no API to enumerate a runtime's active sessions — ``ListSessions``
#: is Memory-scoped — and session traffic does not move the runtime's
#: ``lastUpdatedAt``. A lease written at provisioning time is therefore the
#: only signal cleanup has that a runtime may still be serving a matrix.
LEASE_TAG = "benchflow-lease-until"

# Not adjustable per AWS service quotas: "Maximum size (in MB) for a Docker
# image in an AgentCore Runtime" = 2048.
MAX_IMAGE_MB = 2048

_GLOBAL_LOCK = asyncio.Lock()
_KEY_LOCKS: dict[str, asyncio.Lock] = {}
_RESULTS: dict[str, Any] = {}
#: runtime ARN -> monotonic time of the last lease refresh by this process.
_LEASE_RENEWED: dict[str, float] = {}
_LEASE_LOCKS: dict[str, asyncio.Lock] = {}
# Contexts are trusted task inputs. Cache only an observed immutable file
# version, including ctime/inode, never a path or mtime alone. This bounds warm
# trials' reads without hiding edits between runs or allocating a whole file.
_FILE_DIGESTS: OrderedDict[tuple[int, ...], bytes] = OrderedDict()
_FILE_DIGEST_LOCK = threading.Lock()
_FILE_DIGEST_CACHE_SIZE = 4096


def _file_version(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mode,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _context_file_digest(path: Path) -> bytes:
    # Serialize misses as well as cache access: a fan-out of publishers must
    # not each read the same multi-GB file while the first hash is in flight.
    with _FILE_DIGEST_LOCK:
        fd = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(fd, "rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("Build context entry must remain a regular file")
            key = _file_version(before)
            if key in _FILE_DIGESTS:
                _FILE_DIGESTS.move_to_end(key)
                return _FILE_DIGESTS[key]
            digest = hashlib.sha256()
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
            if key != _file_version(os.fstat(source.fileno())):
                raise ValueError("Build context changed while hashing")
            value = digest.digest()
            _FILE_DIGESTS[key] = value
            if len(_FILE_DIGESTS) > _FILE_DIGEST_CACHE_SIZE:
                _FILE_DIGESTS.popitem(last=False)
            return value


async def once[T](key: str, factory: Callable[[], Awaitable[T]]) -> T:
    """Run *factory* at most once per *key* for the life of the process.

    Concurrent callers with the same key block on one lock and then read the
    memoized result, so a fan-out of N rollouts over the same task image
    performs exactly one build, one push, and one runtime registration.

    Failures are deliberately not memoized: a transient throttle or a network
    blip should not poison every later rollout in a long matrix run.
    """
    if key in _RESULTS:
        return _RESULTS[key]
    async with _GLOBAL_LOCK:
        lock = _KEY_LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        if key in _RESULTS:
            return _RESULTS[key]
        value = await factory()
        _RESULTS[key] = value
        return value


def reset_cache() -> None:
    """Drop memoized provisioning state (tests only)."""
    _RESULTS.clear()
    _KEY_LOCKS.clear()
    _LEASE_RENEWED.clear()
    _LEASE_LOCKS.clear()
    with _FILE_DIGEST_LOCK:
        _FILE_DIGESTS.clear()


def lease_renewal_interval(window_seconds: float) -> float:
    """How often a process may refresh one runtime's lease."""
    return max(window_seconds / 4, 60.0)


def lease_needs_renewal(runtime_arn: str, window_seconds: float, now: float) -> bool:
    """Whether this process should refresh *runtime_arn*'s lease.

    Provisioning is memoized, so only the first rollout of an image reaches the
    creation path — every later rollout would inherit a lease that keeps aging
    while it runs. A long or staggered matrix therefore ends up with active
    sessions on an expired lease, which is the deletion hazard the lease exists
    to prevent.

    Renewal is throttled to a quarter of the lease window so the refresh costs
    a handful of control-plane calls per runtime per run rather than one per
    rollout (``TagResource`` shares the tight control-plane budget).

    Pure predicate: it records nothing. The throttle is only advanced by
    :func:`mark_lease_renewed` after a write actually lands, so a failed
    renewal cannot be remembered as a successful one and suppress the retry
    that would have fixed it.
    """
    last = _LEASE_RENEWED.get(runtime_arn)
    if last is None:
        return True
    return (now - last) >= lease_renewal_interval(window_seconds)


def mark_lease_renewed(runtime_arn: str, now: float) -> None:
    """Record a lease write that succeeded."""
    _LEASE_RENEWED[runtime_arn] = now


async def renew_lease(
    runtime_arn: str,
    window_seconds: float,
    write: Callable[[], Awaitable[None]],
    *,
    monotonic: Callable[[], float],
) -> bool:
    """Single-flight a due lease refresh for one runtime.

    The due check is repeated under the per-runtime lock. Without that second
    check, a fan-out of rollouts all observes the same stale timestamp before
    the first AWS call completes and sends one ``TagResource`` per rollout.
    The timestamp advances only after *write* succeeds, so a failed first call
    leaves the immediate retry due.
    """
    now = monotonic()
    if not lease_needs_renewal(runtime_arn, window_seconds, now):
        return False
    async with _GLOBAL_LOCK:
        lock = _LEASE_LOCKS.setdefault(runtime_arn, asyncio.Lock())
    async with lock:
        now = monotonic()
        if not lease_needs_renewal(runtime_arn, window_seconds, now):
            return False
        await write()
        mark_lease_renewed(runtime_arn, monotonic())
        return True


def read_regular_text(path: Path) -> str:
    """Read a regular file without following a final symlink.

    Task-controlled Dockerfiles and ignore files are build inputs. Following a
    symlink here would let the context escape even though the canonical walker
    correctly skips symlinks.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{path} must be a regular, non-symlink file") from exc
    try:
        mode = os.fstat(fd).st_mode
        if not stat.S_ISREG(mode):
            raise ValueError(f"{path} must be a regular, non-symlink file")
        with os.fdopen(fd, encoding="utf-8") as handle:
            fd = -1
            return handle.read()
    finally:
        if fd >= 0:
            os.close(fd)


def _active_ignore_file(context_dir: Path) -> Path | None:
    """Return Docker's active ignore file for the generated Dockerfile."""
    specific = context_dir / f"{GENERATED_DOCKERFILE}.dockerignore"
    default = context_dir / ".dockerignore"
    for candidate in (specific, default):
        if os.path.lexists(candidate):
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"{candidate} must be a regular, non-symlink file")
            return candidate
    return None


def _clean_dockerignore_lines(text: str) -> list[str]:
    """Docker ignores comments in column one, strips a BOM, then cleans paths."""
    cleaned: list[str] = []
    for raw in text.removeprefix("\ufeff").splitlines():
        if raw.startswith("#"):
            continue
        line = raw.strip()
        if not line:
            continue
        negated = line.startswith("!")
        body = line[1:].strip() if negated else line
        if not body:
            raise ValueError("Invalid empty Docker ignore exclusion")
        normalized = posixpath.normpath(body).strip("/")
        if normalized in {"", "."}:
            continue
        cleaned.append(("!" if negated else "") + normalized)
    return cleaned


def _docker_pattern(pattern: str) -> re.Pattern[str]:
    """Compile Linux Docker globs, anchored at the context root.

    Docker's *, ?, classes and escapes follow filepath.Match; ** crosses
    separators. The caller carries each rule's parent-match state down the
    tree. Git's implicit **/ and directory priority do not
    apply. Invalid patterns fail before either builder stages any files.
    """
    # Moby's leading-** literal form is a suffix match, even when ** is
    # adjacent to a filename rather than a separator (e.g. **.log).
    if pattern.startswith("**") and not any(c in pattern[2:] for c in "*?[]\\"):
        suffix = pattern[2:]
        if suffix.startswith("/"):
            return re.compile(
                "^(?:.*" + re.escape(suffix) + "|" + re.escape(suffix[1:]) + ")$"
            )
        return re.compile("^.*" + re.escape(suffix) + "$")
    parts = ["^"]
    i = 0
    while i < len(pattern):
        char = pattern[i]
        i += 1
        if char == "*":
            if i < len(pattern) and pattern[i] == "*":
                i += 1
                if i < len(pattern) and pattern[i] == "/":
                    i += 1
                parts.append(".*" if i == len(pattern) else "(?:.*/)?")
            else:
                parts.append("[^/]*")
        elif char == "?":
            parts.append("[^/]")
        elif char == "\\":
            if i == len(pattern):
                raise ValueError("Invalid trailing escape in Docker ignore pattern")
            parts.append(re.escape(pattern[i]))
            i += 1
        elif char == "[":
            end = i
            if end < len(pattern) and pattern[end] == "^":
                end += 1
            start = end
            while end < len(pattern) and pattern[end] != "]":
                if pattern[end] == "\\":
                    end += 1
                end += 1
            if end >= len(pattern) or end == start:
                raise ValueError("Invalid character class in Docker ignore pattern")
            # Go uses ^ for negation; ! is an ordinary class member.
            body = pattern[i:end]
            body = (
                body.replace("[", "\\[")
                .replace("&&", "\\&\\&")
                .replace("~~", "\\~\\~")
                .replace("||", "\\|\\|")
            )
            parts.append("[" + body + "]")
            i = end + 1
        else:
            parts.append(re.escape(char))
    parts.append("$")
    try:
        return re.compile("".join(parts))
    except re.error as exc:
        raise ValueError("Invalid Docker ignore pattern") from exc


def _dockerignore_matcher(context_dir: Path) -> Callable[[str], bool]:
    """One Docker matcher for hashing, local staging and CodeBuild uploads."""
    ignore_file = _active_ignore_file(context_dir)
    if ignore_file is None:
        return lambda _relative: False
    rules = []
    for line in _clean_dockerignore_lines(read_regular_text(ignore_file)):
        excluded = line.startswith("!")
        rules.append((_docker_pattern(line[1:] if excluded else line), excluded))

    def ignored(relative: str) -> bool:
        parts = relative.rstrip("/").split("/")
        parent_matches = [False] * len(rules)
        matched = False
        # Match directories in traversal order, retaining the result of EACH
        # rule, as Docker's MatchesUsingParentResults does. A rule skipped at
        # a parent must not be retroactively applied to a child: in particular,
        # `config/private; !config` must keep the private subtree excluded.
        for depth in range(1, len(parts) + 1):
            path = "/".join(parts[:depth])
            matched = False
            current_matches = [False] * len(rules)
            for index, (pattern, excluded) in enumerate(rules):
                match = parent_matches[index]
                if not match:
                    if excluded != matched:
                        continue
                    match = pattern.fullmatch(path) is not None
                current_matches[index] = match
                if match:
                    matched = not excluded
            parent_matches = current_matches
        return matched

    return ignored


def iter_context_entries(context_dir: Path) -> Iterator[tuple[Path, str]]:
    """Canonical Docker build context entries, including empty directories.

    Used by **both** the image digest and the CodeBuild upload so the two views
    cannot drift. They previously did: the local Docker daemon honored
    ``.dockerignore`` while the remote path zipped and uploaded every regular
    file, which shipped ignored files — including secrets — into S3 and also
    let an ignored file change the image identity.

    Symlinks and non-regular special files are skipped so a task-controlled
    link cannot pull host files into the image or the upload (#411). Directory
    entries are retained: an empty directory changes a real Docker context and
    therefore must change both the digest and the CodeBuild ZIP.
    """
    ignored = _dockerignore_matcher(context_dir)
    for path in sorted(context_dir.rglob("*")):
        if path.is_symlink():
            continue
        relative = path.relative_to(context_dir).as_posix()
        if relative in _GENERATED_NAMES:
            continue
        if path.is_dir():
            if not ignored(relative + "/"):
                yield path, relative
            continue
        if not path.is_file() or ignored(relative):
            continue
        yield path, relative


def iter_context_files(context_dir: Path) -> Iterator[tuple[Path, str]]:
    """Compatibility iterator over regular files in the canonical context."""
    for path, relative in iter_context_entries(context_dir):
        if path.is_file():
            yield path, relative


def build_context_digest(
    context_dir: Path, dockerfile_text: str, shim_text: str = ""
) -> str:
    """Content digest of everything that determines the built image.

    Hashes file *contents* rather than paths and mtimes on purpose: BenchFlow
    copies tasks into temporary directories before a run, so any identity based
    on location or timestamp would change every run and defeat image reuse
    entirely.

    ``shim_text`` and the executable mode bit are part of the identity because
    both change the built image: the shim is copied in as the entrypoint, and
    an ``entrypoint.sh`` flipped from 0644 to 0755 produces a different
    container even though every byte of content is unchanged.
    """
    digest = hashlib.sha256()

    def field(label: bytes, payload: bytes) -> None:
        # Length-prefixed fields: without this a file literally named "shim",
        # or a path containing the separator byte, could be framed to produce
        # the same digest as a different context.
        digest.update(label)
        digest.update(str(len(payload)).encode())
        digest.update(b":")
        digest.update(payload)

    field(b"dockerfile", dockerfile_text.encode())
    field(b"shim", shim_text.encode())
    for path, relative in iter_context_entries(context_dir):
        field(b"path", relative.encode())
        field(b"kind", b"dir" if path.is_dir() else b"file")
        # Full permission bits, not just the executable flag.
        field(b"mode", format(path.stat().st_mode & 0o7777, "04o").encode())
        if path.is_file():
            field(b"blob", _context_file_digest(path))
    return digest.hexdigest()


def image_tag(task_name: str, digest: str) -> str:
    """ECR tag for a task image: readable prefix plus content digest."""
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", task_name).strip("-.")[:40].lower()
    return f"bf-{safe}-{digest[:16]}"


def runtime_name(task_name: str, digest: str) -> str:
    """Agent-runtime name derived from an image-plus-contract identity.

    Two compatible rollouts — different trials, or the with-skill and no-skill
    arms when their images and runtime contracts match — resolve to the same
    name and therefore share one runtime. AgentCore accepts
    ``[A-Za-z][A-Za-z0-9_]*``; the ``bf_`` prefix guarantees the leading letter.
    """
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", task_name)[:28].strip("_")
    return f"bf_{safe}_{digest[:12]}"[:48]


def image_size_error(size_bytes: int, image_uri: str) -> str | None:
    """Return an error message if *size_bytes* exceeds AgentCore's hard cap.

    The 2 GB limit is **not adjustable**. Without this check an oversized image
    fails later as an opaque runtime error, which reads as a task failure
    rather than as an environment that AgentCore cannot host at all.
    """
    size_mb = size_bytes / (1024 * 1024)
    if size_mb <= MAX_IMAGE_MB:
        return None
    return (
        f"Image {image_uri} is {size_mb:.0f} MB, over AgentCore's "
        f"{MAX_IMAGE_MB} MB per-image limit (a hard service quota, not "
        "adjustable). Slim the task image, or run this task on the docker or "
        "daytona sandbox instead."
    )


def find_runtime_by_name(control: Any, name: str) -> tuple[str, str, str | None] | None:
    """Look up an existing runtime by name → ``(arn, id, image_uri)``.

    ``ListAgentRuntimes`` is a 5/s quota, so this is only ever the slow path
    behind :func:`once` and the create-conflict fallback — never per rollout.
    """
    paginator = control.get_paginator("list_agent_runtimes")
    for page in paginator.paginate():
        for runtime in page.get("agentRuntimes", []):
            if runtime.get("agentRuntimeName") != name:
                continue
            runtime_id = runtime["agentRuntimeId"]
            detail = control.get_agent_runtime(agentRuntimeId=runtime_id)
            artifact = detail.get("agentRuntimeArtifact") or {}
            image = (artifact.get("containerConfiguration") or {}).get("containerUri")
            return detail["agentRuntimeArn"], runtime_id, image
    return None
