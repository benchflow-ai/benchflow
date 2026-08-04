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
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

from pathspec import GitIgnoreSpec

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
    """Apply Docker's path-cleaning pass before wildcard matching."""
    cleaned: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        body = line[1:] if negated else line
        # Backslash-escaped leading ``!``/``#`` are literals, not control
        # markers; PathSpec understands those escapes after cleaning.
        normalized = posixpath.normpath(body).strip("/")
        if normalized in {"", "."}:
            continue
        cleaned.append(("!" if negated else "") + normalized)
    return cleaned


def _dockerignore_matcher(context_dir: Path) -> Callable[[str], bool]:
    """Compile Docker-compatible ignore rules for the generated Dockerfile.

    ``GitIgnoreSpec`` supplies the mature ``**``, character-class, escaped
    leading ``!``/``#``, directory, and last-match-wins semantics. Docker's
    own preprocessing difference is handled above by cleaning ``.``/``..`` and
    leading/trailing separators first. A Dockerfile-specific ignore file takes
    precedence over the root ``.dockerignore``, just like ``docker build -f``.
    """
    ignore_file = _active_ignore_file(context_dir)
    if ignore_file is None:
        return lambda _relative: False
    spec = GitIgnoreSpec.from_lines(
        _clean_dockerignore_lines(read_regular_text(ignore_file))
    )
    return spec.match_file


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
            field(b"blob", hashlib.sha256(path.read_bytes()).digest())
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
