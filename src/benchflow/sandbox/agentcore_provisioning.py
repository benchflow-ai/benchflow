"""Process-wide, single-flight provisioning of AgentCore images and runtimes.

A rollout on AgentCore is a **session**, not a runtime. One agent runtime can
host many concurrent sessions, each an isolated microVM with its own
filesystem — measured at 8 concurrent sessions on one runtime, and the account
quota for *Active Session Workloads* is 5000 against only 100 *Total Agents*.
So the expensive, rate-limited artifacts (an ECR image and a registered
runtime) must be created **once per distinct task image** and shared by every
rollout that uses it, while sessions are what scale out.

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
import re
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
    """
    interval = max(window_seconds / 4, 60.0)
    last = _LEASE_RENEWED.get(runtime_arn)
    if last is not None and now - last < interval:
        return False
    _LEASE_RENEWED[runtime_arn] = now
    return True


def _translate_ignore_pattern(pattern: str) -> re.Pattern[str]:
    """Compile one ``.dockerignore`` pattern to a regex over relative paths.

    Follows Docker's matcher rather than approximating it: leading and trailing
    separators are stripped (so ``/secret.env`` is the root file, not an
    absolute path that never matches), ``*`` and ``?`` stop at a separator,
    ``**`` spans zero or more path segments, and ``[...]`` character classes
    are honored. Approximating any of these leaks files Docker excludes into
    the CodeBuild upload, which is a credential-exposure bug because that
    archive goes to S3.
    """
    cleaned = pattern.strip().strip("/")
    segments = cleaned.split("/")
    regex = ""
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if segment == "**":
            regex += "(?:.*)?" if last else "(?:[^/]+/)*"
            continue
        regex += _translate_segment(segment)
        if not last:
            regex += "/"
    return re.compile(f"^{regex}$")


def _translate_segment(segment: str) -> str:
    """Translate one path segment's wildcards, including character classes."""
    out = ""
    index = 0
    while index < len(segment):
        char = segment[index]
        if char == "*":
            out += "[^/]*"
        elif char == "?":
            out += "[^/]"
        elif char == "[":
            close = segment.find("]", index + 2)
            if close == -1:
                # An unterminated class is a literal bracket, as in shell
                # globbing — not a syntax error.
                out += re.escape(char)
            else:
                body = segment[index + 1 : close]
                negate = body[:1] in ("!", "^")
                if negate:
                    body = body[1:]
                # Keep ranges intact; escape only what would change meaning.
                body = body.replace("\\", "\\\\").replace("]", "\\]")
                out += f"[{'^' if negate else ''}{body}]"
                index = close + 1
                continue
        else:
            out += re.escape(char)
        index += 1
    return out


def _dockerignore_matcher(context_dir: Path) -> Callable[[str], bool]:
    """Compile ``.dockerignore`` into a predicate over context-relative paths.

    Docker semantics: ``#`` comments, ``!`` re-includes, last matching rule
    wins, and a rule matching a directory excludes everything beneath it.
    """
    ignore_file = context_dir / ".dockerignore"
    if not ignore_file.is_file():
        return lambda _relative: False

    rules: list[tuple[re.Pattern[str], bool]] = []
    for raw in ignore_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        body = line.lstrip("!").strip()
        if body.strip("/"):
            rules.append((_translate_ignore_pattern(body), negated))

    def matches(relative: str) -> bool:
        # A rule that matches an ancestor directory excludes its contents.
        segments = relative.split("/")
        candidates = ["/".join(segments[: i + 1]) for i in range(len(segments))]
        ignored = False
        for compiled, negated in rules:
            if any(compiled.match(candidate) for candidate in candidates):
                ignored = not negated
        return ignored

    return matches


def iter_context_files(context_dir: Path) -> Iterator[tuple[Path, str]]:
    """The canonical Docker build context: ``(absolute path, relative path)``.

    Used by **both** the image digest and the CodeBuild upload so the two views
    cannot drift. They previously did: the local Docker daemon honored
    ``.dockerignore`` while the remote path zipped and uploaded every regular
    file, which shipped ignored files — including secrets — into S3 and also
    let an ignored file change the image identity.

    Symlinks are skipped so a task-controlled link cannot pull host files into
    the image or the upload (#411).
    """
    ignored = _dockerignore_matcher(context_dir)
    for path in sorted(context_dir.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(context_dir).as_posix()
        if relative in _GENERATED_NAMES or ignored(relative):
            continue
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
    for path, relative in iter_context_files(context_dir):
        field(b"path", relative.encode())
        # Full permission bits, not just the executable flag: 0600 and 0644
        # produce different containers, and a secret dropped to 0644 during an
        # upgrade must not reuse the old image.
        field(b"mode", format(path.stat().st_mode & 0o7777, "04o").encode())
        field(b"blob", hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def image_tag(task_name: str, digest: str) -> str:
    """ECR tag for a task image: readable prefix plus content digest."""
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", task_name).strip("-.")[:40].lower()
    return f"bf-{safe}-{digest[:16]}"


def runtime_name(task_name: str, digest: str) -> str:
    """Agent-runtime name derived from image identity, not the task name.

    Two rollouts of the same task image — different trials, or the with-skill
    and no-skill arms when their images happen to match — resolve to the same
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
