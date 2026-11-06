"""Daytona snapshots named by what they hold (opt-in, owner-scoped).

Without this, a Daytona trial sends its image recipe (the task image plus
U3b's install layers, ``daytona_layers``) and Daytona builds it, reusing a
build only for 24 hours on the runner that made it. With
``BENCHFLOW_DAYTONA_SNAPSHOTS=content`` and ``BENCHFLOW_DAYTONA_OWNER`` set,
the direct strategy registers that recipe once as a snapshot named by a hash
of everything the build reads, and creates every later sandbox of the same
inputs from it, on any runner and from any process.

The name is ``bf-img2-<owner hash32>-<content hash16>``: the fixed-width owner
field is the one branch snapshots use
(``daytona_reaper.owner_snapshot_prefix``), with ``bf-img2-`` instead of
``bf-snap2-``, so the branch-snapshot reaper (and ``bench sandbox cleanup``)
never selects one, and ``bench cache clean -s daytona`` removes only this
owner's. The content hash covers (:func:`content_identity`):

- the identity format;
- the exact recipe the SDK sends (the task Dockerfile or ``FROM <image>``,
  plus the layer lines), and every context file the SDK uploads for it
  (path, full mode and bytes, walked the way the SDK archives them);
- U3b's layer lines and keys, in order;
- every ``FROM``, which must be pinned by digest (or ``scratch``, or an
  earlier stage);
- cpu, memory and disk, because a sandbox made from a snapshot takes the
  snapshot's resources;
- the Daytona API URL and target.

Inputs the hash cannot see make a trial ineligible (it builds as before, and
``sandbox.json`` says why): a moving tag in any ``FROM``, ``ADD``, a ``COPY``
glob or line continuation, a ``RUN --mount`` from the context, a symlink or
special file in the context, a ``FROM`` with a variable, or ``ONBUILD``. A recipe hash pins
the recipe, not the network: package resolution inside a ``RUN`` is what it
was the day the snapshot was built.

Creation is coordinated by the name, not by a lock across processes: Daytona
keeps one snapshot per name, so of two processes that race, one ``create``
succeeds and the other gets a conflict and waits for the snapshot to be
active. In one process, an ``asyncio.Lock`` per name keeps trials from even
trying twice. States are handled explicitly (:func:`ensure_snapshot`).
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import tempfile
import time
import weakref
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

#: Turns content snapshots on (``content``); anything else leaves them off.
SNAPSHOTS_ENV = "BENCHFLOW_DAYTONA_SNAPSHOTS"
CONTENT = "content"
NAME_PREFIX = "bf-img2-"
#: Bump when what :func:`content_identity` hashes changes.
IDENTITY_FORMAT = 4
_HASH_LEN = 16
_POLL_START_SEC = 1.0
_POLL_MAX_SEC = 5.0

_DIGEST_REF = re.compile(r"\S+@sha256:[0-9a-f]{64}")
_FROM = re.compile(r"^\s*FROM\s+(.*)$", re.IGNORECASE)
_INSTRUCTION = re.compile(r"^\s*([A-Za-z]+)(?:\s+(.*))?$")
_MODELED_INSTRUCTIONS = frozenset(
    [
        "FROM",
        "RUN",
        "CMD",
        "LABEL",
        "MAINTAINER",
        "EXPOSE",
        "ENV",
        "ADD",
        "COPY",
        "ENTRYPOINT",
        "VOLUME",
        "USER",
        "WORKDIR",
        "ARG",
        "ONBUILD",
        "STOPSIGNAL",
        "HEALTHCHECK",
        "SHELL",
    ]
)


class SnapshotFallback(Exception):
    """No content snapshot for this trial; the message is the record's reason."""


def enabled() -> bool:
    return os.environ.get(SNAPSHOTS_ENV, "").strip().lower() == CONTENT


def owner_prefix() -> str | None:
    """The same complete owner field as branch snapshots, in our namespace."""
    from benchflow.sandbox.daytona_reaper import owner_snapshot_prefix

    prefix = owner_snapshot_prefix()
    if prefix is None:
        return None
    return NAME_PREFIX + prefix.removeprefix("bf-snap2-")


def snapshot_name(prefix: str, content_hash: str) -> str:
    return f"{prefix}{content_hash}"


def is_own_name(name: Any, prefix: str | None) -> bool:
    """Whether ``name`` is one of this owner's content snapshots, exactly.

    The whole owner prefix (part and hash) and a 16-hex content hash, nothing
    more: a near-prefix, another owner's or a malformed name never matches.
    """
    if prefix is None or not isinstance(name, str) or not name.startswith(prefix):
        return False
    return re.fullmatch(f"[0-9a-f]{{{_HASH_LEN}}}", name[len(prefix) :]) is not None


# -- what the build reads ----------------------------------------------


def _instructions(dockerfile: str) -> list[tuple[str, bool]]:
    """The recipe's instructions, continuation lines joined, comments left out.

    Each comes with whether it spanned lines (the SDK reads COPY sources one
    physical line at a time, so a continued COPY uploads the wrong context).
    """
    out: list[tuple[str, bool]] = []
    current: list[str] = []
    for raw in dockerfile.splitlines():
        line = raw.strip()
        if not current and (not line or line.startswith("#")):
            continue
        if current and line.startswith("#"):
            continue
        if line.endswith("\\"):
            current.append(line[:-1])
            continue
        current.append(line)
        out.append((" ".join(current), len(current) > 1))
        current = []
    if current:
        out.append((" ".join(current), len(current) > 1))
    return out


def _copy_parts(rest: str) -> tuple[str, list[str]]:
    """COPY flags and operands, only where JSON rewriting preserves meaning."""
    flags = re.match(r"(?:\s*--[^\s]+\s+)*", rest)
    assert flags is not None
    operands = rest[flags.end() :].strip()
    if operands.startswith("["):
        paths = json.loads(operands)
    else:
        # shlex removes quotes and escapes before Docker's environment
        # replacement, turning e.g. /out/\$DEST into /out/$DEST. Keep the
        # original recipe for shell operands whose syntax cannot survive
        # translation to JSON; JSON operands retain their decoded strings.
        if any(c in operands for c in "\\'\"`"):
            raise ValueError("COPY shell quoting or escaping cannot be preserved")
        paths = shlex.split(operands)
    if (
        not isinstance(paths, list)
        or len(paths) < 2
        or not all(isinstance(path, str) for path in paths)
    ):
        raise ValueError("COPY needs source paths and a destination")
    return flags.group().strip(), paths


def check_recipe(dockerfile: str) -> tuple[list[str], str | None]:
    """The recipe's external ``FROM`` refs, or why it cannot be hashed.

    Returns (refs, None) when every input is visible to the hash; (refs so
    far, reason) otherwise.
    """
    # Docker joins continuations before recognizing instruction names and
    # flags. Rather than maintain a second version of its grammar, admit only
    # physical single-line instructions under the default escape grammar.
    # Refuse directives even when Docker would ignore their position/case.
    for raw in dockerfile.splitlines():
        if re.match(r"^\s*#\s*escape\s*=", raw, re.IGNORECASE):
            return [], "Docker escape directives are not modeled"
        if raw.rstrip().endswith("\\"):
            return [], "Docker line continuations are not modeled"
    stages: set[str] = set()
    refs: list[str] = []
    for line, continued in _instructions(dockerfile):
        match = _INSTRUCTION.match(line)
        if match is None:
            return refs, f"an instruction cannot be read: {line[:60]}"
        word = match.group(1).upper()
        rest = match.group(2) or ""
        if word not in _MODELED_INSTRUCTIONS:
            return refs, f"instruction {word} is not modeled"
        if word == "ONBUILD":
            # Deferred instructions can fetch images, mount inputs or copy
            # contexts when a later stage consumes this one. We do not model
            # that execution graph; even a harmless-looking trigger falls back.
            return refs, "ONBUILD inputs are not modeled"
        if "<<" in rest and word in {"RUN", "COPY"}:
            return refs, f"a {word} heredoc"
        if word == "FROM":
            try:
                parts = shlex.split(rest)
            except ValueError:
                return refs, "a FROM line cannot be read"
            parts = [p for p in parts if not p.startswith("--platform")]
            if not parts:
                return refs, "a FROM line names no image"
            ref = parts[0]
            if any("$" in p for p in parts):
                return refs, f"FROM {ref} uses a variable"
            # FROM resolves against earlier stages, before this stage's
            # alias exists (FROM ubuntu AS ubuntu still reads the registry).
            earlier = ref.lower() == "scratch" or ref.lower() in stages
            if not earlier and not _DIGEST_REF.fullmatch(ref):
                return refs, f"FROM {ref} is not pinned by digest"
            if not earlier:
                refs.append(ref)
            if len(parts) >= 3 and parts[1].lower() == "as":
                stages.add(parts[2].lower())
        elif word == "ADD":
            return refs, "ADD can fetch or unpack what the hash cannot see"
        elif word == "COPY":
            if continued:
                return refs, "a COPY line continues on the next line"
            source = re.search(r"--from=(\S+)", rest)
            if source is not None:
                origin = source.group(1)
                if origin.lower() in stages or origin.isdigit():
                    continue
                if not _DIGEST_REF.fullmatch(origin):
                    return refs, f"COPY --from={origin} is not pinned by digest"
                refs.append(origin)
                continue
            try:
                flags, paths = _copy_parts(rest)
            except ValueError:
                return refs, "a COPY line cannot be read"
            if any(
                flag.split("=", 1)[0] not in {"--chown", "--chmod", "--link"}
                for flag in flags.split()
            ):
                return refs, "a COPY option cannot be preserved in the upload namespace"
            if any("$" in source for source in paths[:-1]):
                return refs, "a COPY source uses a variable"
            if any(c in source for source in paths[:-1] for c in "*?["):
                return refs, "a COPY source is a glob"
        elif word == "RUN" and "--mount" in rest:
            try:
                tokens = shlex.split(rest)
                if "--mount" in tokens:
                    return refs, "a RUN --mount option cannot be read"
                mounts = [p[8:] for p in tokens if p.startswith("--mount=")]
            except ValueError:
                return refs, "a RUN --mount cannot be read"
            for mount in mounts:
                options: dict[str, str] = {}
                for option in mount.split(","):
                    key, _, value = option.partition("=")
                    if key in options:
                        return refs, "a RUN --mount repeats an option"
                    options[key] = value
                # Only self-contained scratch mounts are modeled. In
                # particular cache's `from` can read an external image too.
                # Default type is bind, independently for every mount.
                allowed = {
                    "cache": {
                        "type",
                        "target",
                        "dst",
                        "destination",
                        "id",
                        "sharing",
                        "mode",
                        "uid",
                        "gid",
                        "ro",
                        "readonly",
                    },
                    "tmpfs": {"type", "target", "dst", "destination", "size"},
                }.get(options.get("type", "bind"))
                if allowed is None or options.keys() - allowed:
                    return refs, "a RUN --mount reads from outside the recipe"
    return refs, None


def _walk(path: Path, archive: str, out: list[list[Any]]) -> str | None:
    """Add ``path``'s entries (as the SDK archives them) to ``out``; or why not."""
    try:
        st = os.lstat(path)
    except OSError as e:
        return f"context entry {archive} cannot be read ({e.strerror})"
    mode = st.st_mode
    if stat.S_ISLNK(mode):
        return f"context entry {archive} is a symlink"
    if stat.S_ISREG(mode):
        digest = hashlib.sha256()
        try:
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
        except OSError as e:
            return f"context file {archive} cannot be read ({e.strerror})"
        out.append(["f", archive, stat.S_IMODE(mode), digest.hexdigest()])
        return None
    if stat.S_ISDIR(mode):
        out.append(["d", archive, stat.S_IMODE(mode)])
        try:
            names = sorted(os.listdir(path))
        except OSError as e:
            return f"context folder {archive} cannot be listed ({e.strerror})"
        for name in names:
            problem = _walk(path / name, f"{archive}/{name}", out)
            if problem is not None:
                return problem
        return None
    return f"context entry {archive} is not a file or folder"


def context_entries(
    context: Iterable[tuple[str, str]],
) -> tuple[list[list[Any]], str | None]:
    """Every file and folder the SDK uploads for ``context``, or why not.

    ``context`` is the SDK image's ``(source_path, archive_path)`` pairs
    (``Image._context_list``): exactly what ``snapshot.create`` uploads.
    Regular files are hashed with their full mode, folders by mode and
    contents; a symlink, device, socket or missing path is a reason.
    """
    entries: list[list[Any]] = []
    for source, archive in context:
        problem = _walk(Path(source), archive.strip("/") or ".", entries)
        if problem is not None:
            return entries, problem
    return entries, None


@dataclass
class ContentIdentity:
    """What the content hash covers, and the hash (or why there is none)."""

    material: dict[str, Any] = field(default_factory=dict)
    content_hash: str | None = None
    reason: str | None = None


def content_identity(
    *,
    recipe: str,
    context: Sequence[tuple[str, str]],
    layer_lines: Sequence[str],
    layer_keys: dict[str, str],
    cpu: Any,
    memory: Any,
    disk: Any,
    api_target: str,
) -> ContentIdentity:
    """The content hash of one snapshot's inputs (see the module docstring)."""
    identity = ContentIdentity()
    refs, reason = check_recipe(recipe)
    if reason is None:
        entries, reason = context_entries(context)
    else:
        entries = []
    if reason is not None:
        identity.reason = reason
        return identity
    identity.material = {
        "format": IDENTITY_FORMAT,
        "recipe": recipe,
        "context": entries,
        "from": refs,
        "layers": list(layer_lines),
        "layer_keys": sorted(layer_keys.items()),
        "resources": {"cpu": cpu, "memory": memory, "disk": disk},
        "api": api_target,
    }
    blob = json.dumps(identity.material, sort_keys=True, separators=(",", ":"))
    identity.content_hash = hashlib.sha256(blob.encode()).hexdigest()[:_HASH_LEN]
    return identity


def api_target() -> str:
    """The Daytona API this process creates on: URL and target."""
    url = os.environ.get("DAYTONA_API_URL", "").strip() or "default"
    target = os.environ.get("DAYTONA_TARGET", "").strip() or "default"
    return f"{url.rstrip('/')}|{target}"


# -- ensure ------------------------------------------------------------


def _capture(source: str, destination: Path, *, parent_fd: int | None = None) -> None:
    """Copy a regular tree through no-follow descriptors into private staging.

    Opening nonblocking also makes a file swapped for a FIFO fail without
    stalling setup. Directory entries are opened relative to the held fd;
    changing the original pathname cannot redirect an in-progress walk.
    """
    st = os.stat(source, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(st.st_mode):
        raise SnapshotFallback("a context entry is a symlink")
    if not (stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode)):
        raise SnapshotFallback("a context entry is not a file or folder")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(source, flags, dir_fd=parent_fd)
    try:
        st = os.fstat(fd)
        if stat.S_ISREG(st.st_mode):
            with (
                os.fdopen(fd, "rb", closefd=False) as src,
                destination.open("xb") as dst,
            ):
                shutil.copyfileobj(src, dst)
        elif stat.S_ISDIR(st.st_mode):
            destination.mkdir(mode=0o700)
            for name in sorted(os.listdir(fd)):
                _capture(name, destination / name, parent_fd=fd)
        else:
            raise SnapshotFallback("a context entry is not a file or folder")
        destination.chmod(stat.S_IMODE(st.st_mode))
    finally:
        os.close(fd)


@contextmanager
def captured_image(image: Any) -> Iterator[Any]:
    """An SDK Image whose context stays fixed until the upload has finished.

    Hash and upload this copy, never the source paths. Keep the caller's
    Image intact for recipe fallback; the private directory is released on
    success, failure and cancellation. Each upload's archive namespace
    includes a full manifest digest: SDK 0.184 hashes names and bytes without
    boundaries or modes. Source-path staging alone cannot fix its cache.
    Format 3 excludes snapshots that may have used those polluted uploads.
    """
    # Unsupported recipes never allocate staging or touch their context.
    reason = check_recipe(image.dockerfile())[1]
    if reason is not None:
        raise SnapshotFallback(reason)
    staged = copy.deepcopy(image)
    if not getattr(staged, "_context_list", None):
        yield staged
        return
    try:
        with tempfile.TemporaryDirectory(prefix="benchflow-snapshot-") as root:
            for index, entry in enumerate(getattr(staged, "_context_list", None) or []):
                destination = Path(root) / str(index)
                _capture(str(entry.source_path), destination)
                entry.source_path = str(destination)
            # Do not rewrite unsupported recipes: their original reason is
            # recorded by content_identity and the caller builds as before.
            if check_recipe(staged.dockerfile())[1] is None:
                _namespace_context(staged)
            yield staged
    except OSError as e:
        raise SnapshotFallback(f"context could not be captured: {e.strerror}") from e


async def blocking_work(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Finish disk work before cancellation releases its staging directory."""
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        await asyncio.gather(worker, return_exceptions=True)
        raise


@asynccontextmanager
async def captured_image_async(image: Any) -> AsyncIterator[Any]:
    """Capture and release off-loop, including cancellation during capture."""
    manager = captured_image(image)
    entered = False

    def enter() -> Any:
        nonlocal entered
        staged = manager.__enter__()
        entered = True
        return staged

    try:
        staged = await blocking_work(enter)
        yield staged
    finally:
        if entered:
            await blocking_work(manager.__exit__, None, None, None)


def _context_path(path: str) -> str:
    normalized = PurePosixPath(path.lstrip("/"))
    if ".." in normalized.parts:
        raise SnapshotFallback("a context archive path contains '..'")
    return str(normalized)


def _namespace_context(image: Any) -> None:
    """Salt each SDK upload key without changing what any COPY selects.

    Use the manifest in the *archive* path, which the SDK actually hashes,
    and rewrite local COPY sources to those paths. No SDK-global patch,
    extra build inputs or change to the caller's fallback recipe is needed.
    """
    sources: dict[str, str] = {}
    for entry in image._context_list:
        if not isinstance(entry.archive_path, str):
            raise SnapshotFallback("a context archive path is not explicit")
        original = _context_path(str(entry.archive_path))
        entries, reason = context_entries([(str(entry.source_path), original)])
        if reason is not None:
            raise SnapshotFallback(reason)
        manifest = json.dumps(
            {"format": IDENTITY_FORMAT, "context": entries},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(manifest.encode()).hexdigest()
        archive = f"bf-context-{digest}/{original}"
        if original in sources and sources[original] != archive:
            raise SnapshotFallback("a context source has conflicting captures")
        sources[original] = archive
        entry.archive_path = archive

    lines = []
    for line in image.dockerfile().splitlines(keepends=True):
        match = _INSTRUCTION.match(line)
        if (
            match is None
            or match.group(1).upper() != "COPY"
            or re.search(r"--from=", match.group(2))
        ):
            lines.append(line)
            continue
        flags, paths = _copy_parts(match.group(2))
        try:
            copied = [sources[_context_path(path)] for path in paths[:-1]]
        except KeyError as e:
            raise SnapshotFallback("a COPY source has no captured context") from e
        # Parsing accepts only JSON strings or unquoted, unescaped shell
        # operands. Retain those strings and flags, including the destination.
        lines.append(
            f"COPY {flags + ' ' if flags else ''}{json.dumps([*copied, paths[-1]])}\n"
        )
    image._dockerfile = "".join(lines)


_name_locks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
] = weakref.WeakKeyDictionary()


def _name_lock(name: str) -> asyncio.Lock:
    locks = _name_locks.setdefault(asyncio.get_running_loop(), {})
    return locks.setdefault(name, asyncio.Lock())


def _state(snapshot: Any) -> str:
    state = getattr(snapshot, "state", None)
    value = getattr(state, "value", state)
    return str(value or "").lower()


def _is_not_found(error: BaseException) -> bool:
    if type(error).__name__ == "DaytonaNotFoundError":
        return True
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    return status == 404


def _is_conflict(error: BaseException) -> bool:
    if type(error).__name__ == "DaytonaConflictError":
        return True
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    text = str(error).lower()
    return status == 409 or "already exists" in text or "conflict" in text


_BUSY = {"pending", "building", "pulling"}
_FAILED = {"error", "build_failed"}


async def _remove_failed(client: Any, name: str) -> None:
    """After a failed create, remove the snapshot it left if it is failed.

    Read again first: only a snapshot still in ERROR or BUILD_FAILED goes,
    never an ACTIVE or PENDING one another process made meanwhile. Best
    effort; what stays is replaced once by the next trial.
    """
    try:
        left = await client.snapshot.get(name)
        if _state(left) in _FAILED:
            await client.snapshot.delete(left)
    except Exception:
        return


@dataclass
class EnsureResult:
    outcome: str  # "created", "reused" or "waited"
    snapshot: Any
    create_attempts: int = 0
    conflicts: int = 0
    replaced: int = 0
    events: list[str] = field(default_factory=list)


async def ensure_snapshot(
    client: Any,
    name: str,
    make_params: Any,
    *,
    timeout_sec: float,
) -> EnsureResult:
    """Make sure snapshot ``name`` is ACTIVE: reuse it, wait for it, or create it.

    ``make_params()`` builds the SDK's ``CreateSnapshotParams`` (only called
    when a create is tried). Bounded by ``timeout_sec`` in all:

    - ACTIVE: reused (``waited`` when this call waited for it first);
    - PENDING, BUILDING, PULLING: waited for;
    - REMOVING: waited for until it is gone, then created;
    - missing: created; a conflict (another process created it first) is
      waited for;
    - ERROR or BUILD_FAILED: read again, and only when the same snapshot (by
      id) is still failed, deleted and created again, at most once; a newly
      ACTIVE or PENDING replacement by someone else is never deleted;
    - INACTIVE: activated once (the pinned SDK has ``snapshot.activate``);
    - anything else, a failed create, or the time limit: SnapshotFallback.

    Cancellation propagates (``asyncio.CancelledError`` is not caught).
    """
    deadline = time.monotonic() + max(timeout_sec, 0.0)
    try:
        # Includes waiting for our process lock, every SDK call, polling
        # and failed-create cleanup. SDK per-request timeouts alone do not
        # bound those operations. External cancellation is not translated.
        async with asyncio.timeout(max(timeout_sec, 0.0)):
            return await _ensure_snapshot(
                client,
                name,
                make_params,
                timeout_sec=timeout_sec,
                deadline=deadline,
            )
    except TimeoutError as e:
        raise SnapshotFallback(
            f"snapshot {name} was not active within {timeout_sec:.0f}s"
        ) from e


async def _ensure_snapshot(
    client: Any,
    name: str,
    make_params: Any,
    *,
    timeout_sec: float,
    deadline: float,
) -> EnsureResult:
    result = EnsureResult(outcome="reused", snapshot=None)
    waited = False
    activated = False
    delay = _POLL_START_SEC

    async def pause() -> None:
        nonlocal delay
        left = deadline - time.monotonic()
        if left <= 0:
            raise SnapshotFallback(
                f"snapshot {name} was not active within {timeout_sec:.0f}s"
            )
        await asyncio.sleep(min(delay, left))
        delay = min(delay * 2, _POLL_MAX_SEC)

    async with _name_lock(name):
        while True:
            if time.monotonic() > deadline:
                raise SnapshotFallback(
                    f"snapshot {name} was not active within {timeout_sec:.0f}s"
                )
            try:
                snapshot = await client.snapshot.get(name)
            except Exception as e:
                if not _is_not_found(e):
                    raise SnapshotFallback(
                        f"snapshot {name} could not be read: {e}"
                    ) from e
                snapshot = None
            if snapshot is None:
                result.create_attempts += 1
                result.events.append("create")
                try:
                    created = await client.snapshot.create(
                        make_params(),
                        timeout=max(deadline - time.monotonic(), 1.0),
                    )
                except Exception as e:
                    if _is_conflict(e):
                        result.conflicts += 1
                        result.events.append("conflict")
                        waited = True
                        # Back off before reading the name again: a name the
                        # service holds while `get` answers 404 must not
                        # spin get and create until the deadline.
                        await pause()
                        continue
                    await _remove_failed(client, name)
                    raise SnapshotFallback(
                        f"snapshot {name} could not be created: {e}"
                    ) from e
                if _state(created) == "active":
                    result.outcome = "created"
                    result.snapshot = created
                    return result
                # The SDK waits for a terminal state; read what it is now.
                waited = True
                continue
            state = _state(snapshot)
            result.events.append(state or "?")
            if state == "active":
                result.outcome = "waited" if waited else "reused"
                result.snapshot = snapshot
                return result
            if state in _BUSY or state == "removing":
                waited = True
                await pause()
                continue
            if state in _FAILED:
                if result.replaced:
                    raise SnapshotFallback(
                        f"snapshot {name} is {state} again after one replacement"
                    )
                try:
                    again = await client.snapshot.get(name)
                except Exception as e:
                    if _is_not_found(e):
                        continue  # someone removed it: create
                    raise SnapshotFallback(
                        f"snapshot {name} could not be read: {e}"
                    ) from e
                if _state(again) not in _FAILED or getattr(
                    again, "id", None
                ) != getattr(snapshot, "id", None):
                    continue  # replaced by someone else meanwhile: use theirs
                try:
                    await client.snapshot.delete(again)
                except Exception as e:
                    if not _is_not_found(e):
                        raise SnapshotFallback(
                            f"snapshot {name} is {state} and could not be removed: {e}"
                        ) from e
                result.replaced += 1
                result.events.append("deleted-failed")
                waited = True
                continue
            if state == "inactive" and not activated:
                activate = getattr(client.snapshot, "activate", None)
                if activate is None:
                    raise SnapshotFallback(
                        f"snapshot {name} is inactive and this SDK cannot activate it"
                    )
                activated = True
                waited = True
                try:
                    await activate(snapshot)
                except Exception as e:
                    raise SnapshotFallback(
                        f"snapshot {name} is inactive and could not be activated: {e}"
                    ) from e
                continue
            raise SnapshotFallback(f"snapshot {name} is {state or 'in no state'}")


# -- list and delete (``bench cache -s daytona``) ----------------------


def list_own(client: Any) -> tuple[list[Any], str | None]:
    """This owner's content snapshots (sync client), or why none are listed."""
    prefix = owner_prefix()
    if prefix is None:
        return [], (
            "BENCHFLOW_DAYTONA_OWNER is not set, so no snapshot can be proved "
            "to be yours; none are listed"
        )
    found: list[Any] = []
    page = 1
    while True:
        result = client.snapshot.list(page=page, limit=100)
        for snap in getattr(result, "items", None) or []:
            if is_own_name(getattr(snap, "name", None), prefix):
                found.append(snap)
        total_pages = getattr(result, "total_pages", 1) or 1
        if page >= total_pages:
            break
        page += 1
    return found, None


def delete_own(client: Any, snapshot: Any) -> str | None:
    """Delete one of this owner's content snapshots; None, or why it stayed.

    The name is checked again, and the snapshot read again by name: one
    being built or removed stays.
    """
    prefix = owner_prefix()
    name = getattr(snapshot, "name", None)
    if not is_own_name(name, prefix):
        return "not one of this owner's content snapshots"
    try:
        current = client.snapshot.get(name)
    except Exception as e:
        if _is_not_found(e):
            return "already gone"
        raise
    if getattr(current, "id", None) != getattr(snapshot, "id", None):
        return "replaced since it was listed"
    state = _state(current)
    if state in _BUSY or state == "removing":
        return f"it is {state}"
    client.snapshot.delete(current)
    return None
