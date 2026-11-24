"""What a held container's mounts land on, read before each of its starts.

Docker resolves a mount's destination inside the container when it starts
it, following the links it finds there (the image's, and anything the
container's own processes left in its filesystem on an earlier run). So a
destination that names nothing of BenchFlow's can still land on ``/.bfx``,
and cover the waiter every held service starts in. Before every start of a
held container, its first included, BenchFlow reads where each destination
of the created or stopped container's ``docker inspect`` lands, component by
component, and starts it only when:

- no destination lands on ``/.bfx``, beneath it, or on ``/`` (BenchFlow's own
  read-only mount at ``/.bfx`` excepted);
- no destination lands beneath another, unless both are BenchFlow's own
  (bfx, the log folders, the operator's mounts: known by source and
  destination), or the outer one may hold others: read-only as Docker made
  it, and its source written by nothing of the trial's (the caller decides,
  across every service);
- ``/.bfx`` is a folder, not a link;
- ``/.bfx/bfx`` holds the pinned bfx.

Each path is read with ``docker cp <container>:<path> -`` up to the first
member of its archive (a link is one small member; a folder's tree is cut off
after its own), which runs nothing in the container and works wherever the
docker CLI reaches the daemon.
"""

from __future__ import annotations

import asyncio
import io
import posixpath
import tarfile
from collections.abc import Awaitable, Callable
from typing import Any

from benchflow._bfx.locate import SANDBOX_DIR
from benchflow.sandbox.service_cover import ServiceCoverError

#: How many links one destination may pass through.
MAX_LINKS = 40
#: How much of an archive is read for its first member.
FIRST_MEMBER_LIMIT = 1 << 20
#: How many paths are read at once.
CONCURRENCY = 8
#: How long one path's read may take.
READ_SEC = 30.0

#: (kind, link target): kind is "dir", "file", "link", "other" or "missing".
Entry = tuple[str, str]
Reader = Callable[[str], Awaitable[Entry]]


async def gather_all(*awaitables: Awaitable[Any]) -> list[Any]:
    """All of *awaitables* at once; the first failure cancels the rest
    (awaited) before it is raised."""
    tasks = [asyncio.ensure_future(a) for a in awaitables]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class HeldStartRefused(ServiceCoverError):
    """A held container's mounts would cover BenchFlow's waiter, or bfx is
    not BenchFlow's: the container is not started."""


class HeldStartUnread(RuntimeError):
    """A path of a held container could not be read in time: the container
    is not started, and the start fails as BenchFlow's own (the daemon's),
    never as the task's."""


def destinations(inspected: dict[str, Any]) -> list[str]:
    """Every place the created container's inspect says something is mounted
    or made: its mounts (binds, volumes, the image's VOLUMEs, file secrets
    and configs, volumes_from), its tmpfs and its devices."""
    found = [
        str(m.get("Destination"))
        for m in inspected.get("Mounts") or ()
        if isinstance(m, dict) and m.get("Destination")
    ]
    host = inspected.get("HostConfig") or {}
    found += [str(path) for path in host.get("Tmpfs") or {}]
    found += [
        str(d.get("PathInContainer"))
        for d in host.get("Devices") or ()
        if isinstance(d, dict) and d.get("PathInContainer")
    ]
    return sorted(set(found))


def first_member(data: bytes) -> Entry | None:
    """The first member of the (possibly cut off) archive *data*, as an
    :data:`Entry`; None until *data* holds all of its header."""
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
            member = tar.next()
    except (tarfile.TarError, EOFError):
        return None
    if member is None:
        return None
    if member.issym():
        return "link", member.linkname
    if member.isdir():
        return "dir", ""
    if member.isfile():
        return "file", ""
    return "other", ""


async def resolve(path: str, read: Reader, cache: dict[str, Entry]) -> str:
    """Where *path* lands inside the container: each component read with
    *read*, links followed within the container's root as Docker follows
    them (``..`` at the root stays there). A component that does not exist
    is made as a folder, so nothing after it is a link."""
    parts = [p for p in path.split("/") if p not in ("", ".")]
    done: list[str] = []
    links = 0
    while parts:
        part = parts.pop(0)
        if part == "..":
            if done:
                done.pop()
            continue
        here = "/" + "/".join([*done, part])
        if here not in cache:
            cache[here] = await read(here)
        kind, target = cache[here]
        if kind == "link":
            links += 1
            if links > MAX_LINKS:
                raise HeldStartRefused(f"{path} passes through too many links")
            if target.startswith("/"):
                done = []
            parts = [p for p in target.split("/") if p not in ("", ".")] + parts
            continue
        done.append(part)
        if kind == "missing":
            rest = posixpath.normpath("/" + "/".join([*done, *parts]))
            return rest
    return "/" + "/".join(done)


def beneath(path: str, root: str) -> bool:
    """Whether *path* is strictly beneath *root*."""
    return path != root and path.startswith(root.rstrip("/") + "/")


async def check(
    inspected: dict[str, Any],
    read: Reader,
    *,
    own: Callable[[dict[str, Any]], bool],
    holds_none: Callable[[dict[str, Any] | None], str | None] = lambda _: "",
) -> None:
    """Raise :class:`HeldStartRefused`, naming it, when a destination of the
    created container *inspected* lands on, beneath or over ``/.bfx`` or
    beneath another, or ``/.bfx`` is a link. *own* says which of its mounts
    BenchFlow put there; *holds_none* why a mount (its inspect entry, None
    for a tmpfs or a device) may not hold another, None when it may; *read*
    reads one path (:data:`Reader`)."""
    service = ((inspected.get("Config") or {}).get("Labels") or {}).get(
        "com.docker.compose.service", "main"
    )
    wanted = destinations(inspected)
    ours = {
        str(m.get("Destination"))
        for m in inspected.get("Mounts") or ()
        if isinstance(m, dict) and own(m)
    }
    # By their names first, before anything is read (Docker may not even
    # mount the container for a read then).
    named = {d: posixpath.normpath(d) for d in wanted}
    mounts = {
        str(m.get("Destination")): m
        for m in inspected.get("Mounts") or ()
        if isinstance(m, dict)
    }
    why = {d: holds_none(mounts.get(d)) for d in wanted}
    _refuse_over_bfx(service, named, ours)
    _refuse_nested(service, named, ours, why)
    cache: dict[str, Entry] = {}
    gate = asyncio.Semaphore(CONCURRENCY)

    async def bounded(path: str) -> Entry:
        async with gate:
            return await read(path)

    kind, _ = await bounded(SANDBOX_DIR)
    cache[SANDBOX_DIR] = (kind, "")
    if kind != "dir":
        raise HeldStartRefused(
            f"in service {service!r}, {SANDBOX_DIR} is not a folder ({kind}); "
            "BenchFlow's bfx is mounted there"
        )
    landed = await gather_all(*(resolve(d, bounded, cache) for d in wanted))
    places = dict(zip(wanted, landed, strict=True))
    _refuse_over_bfx(service, places, ours)
    _refuse_nested(service, places, ours, why)


def _refuse_over_bfx(service: str, places: dict[str, str], ours: set[str]) -> None:
    """*places*: each destination and where it lands; only BenchFlow's own
    mount at /.bfx may land there."""
    for name, place in places.items():
        if name == SANDBOX_DIR and name in ours and place == SANDBOX_DIR:
            continue
        if place == "/" or place == SANDBOX_DIR or beneath(place, SANDBOX_DIR):
            raise HeldStartRefused(
                f"in service {service!r}, the mount at {name} lands on {place}: "
                f"{SANDBOX_DIR} is BenchFlow's own read-only folder, so nothing "
                "may be mounted at, beneath or over it"
            )


def _refuse_nested(
    service: str,
    places: dict[str, str],
    ours: set[str],
    why: dict[str, str | None],
) -> None:
    """*places*: each destination and where it lands; *ours*: BenchFlow's
    own, which may sit inside one another; *why*: why each may not hold
    another (None: it may)."""
    for name, place in places.items():
        for other, under in places.items():
            if other == name or (name in ours and other in ours):
                continue
            if beneath(place, under) and why.get(other) is not None:
                reason = f" ({why[other]})" if why[other] else ""
                raise HeldStartRefused(
                    f"in service {service!r}, the mount at {name} lands on "
                    f"{place}, beneath the mount at {other} ({under}); a held "
                    f"service's mounts may not sit inside one another{reason}"
                )
