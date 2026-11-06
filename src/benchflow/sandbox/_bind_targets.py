"""Trusted container-side bind identities, retained across container lifetimes.

The host's path proof authorizes the initial mount graph. Only a stat inside
the mounted container can establish what the daemon actually exposed. Never
relearn that baseline after agent code has run (S1-CONF-6/7/8, 6dfad9e5).
"""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import stat
from collections.abc import Collection, Mapping
from typing import Any

from benchflow.sandbox._base import system_shell_route
from benchflow.sandbox._exec_output import exec_raw
from benchflow.task.paths import SandboxPaths

LABEL = "io.benchflow.bind-targets.v1"
BindTargets = dict[str, dict[str, str]]
# The mount targets of BenchFlow's own per-trial log folders. Each sandbox
# binds its own fresh host folders there (docker-compose-base.yaml). Matched
# by mount target: an operator --mount at one of these targets is treated the
# same way (its first start's host-path proof authorized it; a later swap
# still refuses).
LOG_TARGETS = frozenset(
    str(path)
    for path in (
        SandboxPaths.agent_dir,
        SandboxPaths.verifier_dir,
        SandboxPaths.artifacts_dir,
    )
)


def decode(value: str) -> BindTargets:
    remedy = "Invalid bind identity baseline; create a fresh checkpoint"
    if not isinstance(value, str):
        raise ValueError(remedy)
    try:
        if value.startswith("base64:"):
            value = base64.b64decode(value[7:], validate=True).decode("utf-8")
        data = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(remedy) from exc
    if (
        not isinstance(data, dict)
        or "main" not in data
        or not all(
            isinstance(service, str)
            and isinstance(targets, dict)
            and all(
                isinstance(path, str)
                and path.startswith("/")
                and isinstance(identity, str)
                and re.fullmatch(r"[0-9]+:[0-9]+", identity)
                for path, identity in targets.items()
            )
            for service, targets in data.items()
        )
    ):
        raise ValueError(remedy)
    return data


def encode(value: BindTargets) -> str:
    """Dockerfile LABEL expansion cannot express '$' in this alphabet (CONF-10)."""
    return "base64:" + base64.b64encode(json.dumps(value).encode("utf-8")).decode(
        "ascii"
    )


async def host_bind_paths(env: Any, inspected: dict) -> list[str]:
    """Classify daemon mounts without executing anything in the container."""
    mounts = inspected.get("Mounts")
    if not isinstance(mounts, list):
        raise RuntimeError("Cannot inspect bind targets; recreate the sandbox")
    paths = []
    for mount in mounts:
        if not isinstance(mount, dict):
            raise RuntimeError("Cannot inspect bind target; recreate the sandbox")
        kind = mount.get("Type")
        if kind not in {"bind", "volume", "tmpfs", "image"}:
            raise RuntimeError("Cannot identify mount backing; recreate the sandbox")
        backed = kind == "bind"
        if kind == "volume":
            name = mount.get("Name")
            if not isinstance(name, str) or not name:
                raise RuntimeError(
                    "Cannot inspect volume backing; recreate the sandbox"
                )
            result = await exec_raw(env._docker_cli, ["volume", "inspect", name])
            try:
                volumes = json.loads(result.stdout)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Cannot inspect volume backing; recreate the sandbox"
                ) from exc
            if (
                not isinstance(volumes, list)
                or len(volumes) != 1
                or not isinstance(volumes[0], dict)
                or "Options" not in volumes[0]
                or not isinstance(volumes[0]["Options"], (dict, type(None)))
            ):
                raise RuntimeError(
                    "Cannot inspect volume backing; recreate the sandbox"
                )
            # Local driver bind volumes are reported as volumes, not binds.
            backed = bool(volumes[0].get("Options"))
        if backed:
            path = mount.get("Destination")
            if not isinstance(path, str) or not path.startswith("/"):
                raise RuntimeError("Cannot inspect bind target; recreate the sandbox")
            paths.append(path)
    return sorted(set(paths))


async def inspect_targets(env: Any, container: str, inspected: dict) -> dict[str, str]:
    paths = await host_bind_paths(env, inspected)
    if not paths:
        return {}
    from benchflow.sandbox.docker import route_exec_argv

    service = ((inspected.get("Config") or {}).get("Labels") or {}).get(
        "com.docker.compose.service", "main"
    )
    route = await system_shell_route(getattr(env, "_env", env), service)
    # Absolute tools and an empty environment: task PATH/HOME/startup files
    # never participate. The image, system tools and daemon are trusted.
    result = await exec_raw(
        env._docker_cli,
        route_exec_argv(
            route,
            shlex.join(["stat", "-L", "-c", "%d:%i", "--", *paths]),
            container=container,
            user="0",
        ),
        check=False,
    )
    rows = (result.stdout or "").splitlines()
    if (
        result.return_code
        or len(rows) != len(paths)
        or any(re.fullmatch(r"[0-9]+:[0-9]+", row) is None for row in rows)
    ):
        raise RuntimeError(
            "Cannot establish container bind identities; provide trusted "
            "/usr/bin/env and /usr/bin/stat in the image, or remove host mounts"
        )
    return dict(zip(paths, rows, strict=True))


#: Symbolic-link hops one held walk follows before it gives up.
_MAX_LINK_HOPS = 40


def held_chain(path: str) -> list[tuple[int, int]] | None:
    """The (device, inode) of every folder from ``/`` to the object ``path``
    names, resolved as the kernel would (links followed, ``..`` the real
    parent), the object itself last; None when it cannot be walked.

    One component at a time on held descriptors: each is opened
    ``O_PATH | O_NOFOLLOW`` relative to the folder before it, judged with
    ``fstat`` on that descriptor, and a link is read from that same
    descriptor (``readlinkat`` on the held link). Nothing is looked up by
    name twice, so what this returns is one consistent chain of objects even
    if a host-side writer swaps names meanwhile (DET-R2b round 4, L-R3-A1).
    Linux only (``O_PATH``); elsewhere None.
    """
    o_path = getattr(os, "O_PATH", None)
    if o_path is None:
        return None
    flags = o_path | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    stack: list[int] = []
    try:
        stack.append(os.open("/", flags | os.O_DIRECTORY))
        pending = [c for c in path.split("/") if c][::-1]
        hops = 0
        while pending:
            name = pending.pop()
            if name == ".":
                continue
            if name == "..":
                if len(stack) > 1:
                    os.close(stack.pop())
                continue
            fd = os.open(name, flags, dir_fd=stack[-1])
            mode = os.fstat(fd).st_mode
            if stat.S_ISLNK(mode):
                try:
                    link = os.readlink("", dir_fd=fd)
                finally:
                    os.close(fd)
                hops += 1
                if hops > _MAX_LINK_HOPS:
                    return None
                if link.startswith("/"):
                    while len(stack) > 1:
                        os.close(stack.pop())
                pending.extend([c for c in link.split("/") if c][::-1])
                continue
            if pending and not stat.S_ISDIR(mode):
                os.close(fd)
                return None
            stack.append(fd)
        chain = []
        for fd in stack:
            st = os.fstat(fd)
            chain.append((st.st_dev, st.st_ino))
        return chain
    except (OSError, ValueError):
        return None
    finally:
        for fd in stack:
            os.close(fd)


def staged_bind_targets(
    inspected: dict, task_dir: str, pinned: Mapping[str, str]
) -> frozenset[str]:
    """The targets of ``inspected``'s bind mounts whose source, with every
    link resolved, lies inside ``task_dir`` (resolved too), this sandbox's
    private copy of the task (``StagedEnvironment.task_dir``, a ``0700``
    folder BenchFlow made), and is the object the container mounted.

    The daemon reports a bind's source as Compose gave it: with host access
    allowed, staging keeps a link that leaves the task as a link, and a path
    inside the copy can expose a shared host folder. That bind is not this
    sandbox's own and keeps the checkpoint's identity (review A, M-A1).
    Decided at the first start, when the identities are recorded, and only
    where the daemon sees this machine's files (the caller's check).

    Containment and identity come from one held walk (:func:`held_chain`):
    the task copy's folder must be on the source's resolved chain of
    folders, and the object at its end must have the identity that start
    pinned for the target (``pinned``, device and inode). No name is
    resolved twice, so a link swapped between two lookups cannot pair one
    object's containment with another's identity (rounds 3 and 4,
    L-R2-A1, L-R3-A1).
    """
    root = held_chain(task_dir)
    if root is None:
        return frozenset()
    task = root[-1]
    found = set()
    for mount in inspected.get("Mounts") or ():
        if not isinstance(mount, dict) or mount.get("Type") != "bind":
            continue
        source, target = mount.get("Source"), mount.get("Destination")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        chain = held_chain(source)
        if chain is None or task not in chain:
            continue
        dev, ino = chain[-1]
        if pinned.get(target) == f"{dev}:{ino}":
            found.add(target)
    return frozenset(found)


async def staged_project_targets(
    env: Any, ids: list[str], task_dir: str, receipt: BindTargets
) -> dict[str, frozenset[str]]:
    """Per service of the project's containers ``ids``, the bind targets
    whose resolved source is in ``task_dir`` and is the folder ``receipt``
    (the first start's pinned identities) records (DET-R2b)."""
    found: dict[str, frozenset[str]] = {}
    for container in ids:
        inspected = await env._inspect_container(container)
        labels = (inspected.get("Config") or {}).get("Labels") or {}
        service = labels.get("com.docker.compose.service")
        if isinstance(service, str):
            found[service] = staged_bind_targets(
                inspected, task_dir, receipt.get(service, {})
            )
    return found


def with_own_log_binds(
    carried: BindTargets,
    own: BindTargets | None,
    staged: Mapping[str, Collection[str]] | None = None,
) -> BindTargets:
    """What a container restored from a checkpoint is proved against.

    ``carried`` is the checkpoint's receipt (its label: the identities the
    first start of the sandbox that took it recorded); ``own`` is the
    receipt this sandbox recorded at its own first start, before any agent
    code ran in it. A container this sandbox makes from the checkpoint gets
    this sandbox's own folders where a sandbox has its own: BenchFlow's log
    folders (``LOG_TARGETS``, matched by target) and, per service, the
    targets in ``staged`` (binds whose source is this sandbox's private copy
    of the task, recorded at that same first start; DET-R2b). Those are
    expected at ``own``'s identities. Every other bind is carried over from
    the checkpoint and keeps its identity. A target ``own`` does not hold
    keeps the carried identity and so refuses.

    A sandbox restoring its own checkpoint in place has an own receipt equal
    to that checkpoint's, so nothing changes for it. A checkpoint another
    sandbox took, restored in place, is proved like a child's restore: its
    log and staged targets take this sandbox's identities (DET-R2).
    """
    if own is None:
        return carried
    result: BindTargets = {}
    for service, targets in carried.items():
        mine = own.get(service, {})
        theirs = LOG_TARGETS | frozenset((staged or {}).get(service, ()))
        result[service] = {
            path: mine[path] if path in theirs and path in mine else identity
            for path, identity in targets.items()
        }
    return result


def verify(actual: BindTargets, expected: BindTargets) -> None:
    if actual != expected:
        raise RuntimeError(
            "Container bind identity changed since the first start; restore "
            "the original bind sources and mount targets, or start a fresh trial "
            "without checkpoint reuse"
        )


async def pin_project(
    env: Any, output: str, baseline: BindTargets | None
) -> BindTargets:
    containers = output.splitlines()
    if not containers or any(
        not re.fullmatch(r"[0-9a-f]{12,64}", c) for c in containers
    ):
        raise RuntimeError("Cannot identify containers for bind proof; retry startup")
    actual = {}
    inherited = None
    for container in containers:
        inspected = await env._inspect_container(container)
        labels = (inspected.get("Config") or {}).get("Labels") or {}
        service = labels.get("com.docker.compose.service")
        if not isinstance(service, str) or service in actual:
            raise RuntimeError(
                "Cannot identify bind service; use one container per service"
            )
        actual[service] = await inspect_targets(env, container, inspected)
        # Only BenchFlow's read-only bfx at or beneath /.bfx (TRUSTED-RUNTIME
        # §6), checked on the same inspect, before any task process runs.
        check_bfx = getattr(type(env), "_check_bfx_mounts", None)
        if service == "main" and check_bfx is not None:
            check_bfx(env, inspected)
        if service == "main" and LABEL in labels:
            inherited = decode(labels[LABEL])
    if "main" not in actual:
        raise RuntimeError("Cannot identify main for bind proof; retry startup")
    expected = baseline if baseline is not None else inherited
    if expected is not None:
        verify(actual, expected)
    return actual
