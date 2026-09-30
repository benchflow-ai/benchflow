"""Remove leftover BenchFlow Docker containers and networks, never live ones.

Every ``main`` container and default network that BenchFlow's compose files
create carries ``benchflow.owned=process`` and ``benchflow.process``: the
host, pid, boot and start time of the BenchFlow process that created it.
:func:`sweep_leftovers`, which an ``Evaluation`` on Docker runs when it
starts, before a retry and when it ends, removes such a resource only when it
is certainly garbage:

- the process that created it is gone: same host, and another boot, no
  process with that pid, or a pid that now belongs to a later process; or
- it is this process's, and its sandbox has been torn down
  (:func:`release_project`).

A resource of a live process, of another host, or without the label (made by
an older BenchFlow) is left alone. Containers are removed with ``docker rm``
without ``--force``, and networks with ``docker network rm``; Docker refuses
both for a running container and for a network a running container uses.

This replaces ``docker container prune`` and ``docker network prune``
filtered by ``benchflow.owned=true`` alone, which removed every stopped
BenchFlow container and every unused BenchFlow network on the daemon. A live
rollout's resources pass through exactly those states: Compose creates the
network and the container before it starts the container, and branch restore
replaces the ``main`` container. Run while another rollout started (another
task of the same job retrying, another job in the same process, any other
BenchFlow process on the daemon), the prune deleted them mid-start:
"container is marked for removal and cannot be started", "failed to set up
container networking: network <project>_default not found", "removal of
container ... is already in progress".
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger("benchflow")

OWNED_LABEL = "benchflow.owned"
# Before this sweep the value was "true", and every BenchFlow process pruned
# all stopped ``benchflow.owned=true`` containers on the daemon. A new value
# keeps such an older BenchFlow, still running on a shared daemon, off this
# version's containers; and this sweep never lists the older ones.
OWNED_VALUE = "process"
PROCESS_LABEL = "benchflow.process"
PROJECT_LABEL = "com.docker.compose.project"
# The compose variable the ``benchflow.process`` label is read from
# (``_compose_files/docker-compose-base.yaml``).
PROCESS_ENV = "BENCHFLOW_PROCESS"

# ``docker container prune`` removed exactly these states.
_STOPPED_STATES = ("created", "exited", "dead")
_CLI_TIMEOUT_SEC = 30
_LISTING_FORMAT = (
    f'{{{{.ID}}}}\t{{{{.Label "{PROCESS_LABEL}"}}}}\t{{{{.Label "{PROJECT_LABEL}"}}}}'
)

_live_projects: set[str] = set()
_live_lock = threading.Lock()


def claim_project(project: str) -> None:
    """Mark this process's compose *project* as in use: the sweep keeps it.

    The project is also recorded in this process's sandbox lease
    (:mod:`benchflow.sandbox.leases`), so it is deleted even if the process
    is killed before its teardown runs.
    """
    with _live_lock:
        _live_projects.add(project)
    from benchflow.sandbox import leases

    leases.record("docker", project)


def release_project(project: str) -> None:
    """*project*'s sandbox is torn down; the sweep may remove what it left."""
    with _live_lock:
        _live_projects.discard(project)
    from benchflow.sandbox import leases

    leases.release("docker", project)


def live_projects() -> frozenset[str]:
    with _live_lock:
        return frozenset(_live_projects)


# ---------------------------------------------------------------------------
# Which process made a resource, and whether it is gone
# ---------------------------------------------------------------------------


def _boot_id() -> str:
    """The kernel's boot id (Linux), or ""."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


def _start_ticks(pid: int) -> str:
    """When *pid* started, in clock ticks since boot (Linux), or ""."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return ""
    # Field 22. The command name (field 2) may hold spaces and parentheses,
    # so count from the last ")".
    fields = stat.rsplit(")", 1)[-1].split()
    return fields[19] if len(fields) > 19 else ""


def _pid_exists(pid: int) -> bool:
    if sys.platform == "win32":
        # os.kill(pid, 0) sends CTRL_C_EVENT on Windows; never probe there.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # EPERM: it exists; anything else: cannot tell
    return True


@dataclass(frozen=True)
class ProcessId:
    """The ``benchflow.process`` label: ``<host>:<pid>:<boot id>:<start ticks>``.

    Boot id and start time are Linux-only and empty elsewhere; without them
    a pid that exists counts as alive.
    """

    host: str
    pid: int
    boot: str = ""
    start: str = ""

    def token(self) -> str:
        return f"{self.host}:{self.pid}:{self.boot}:{self.start}"

    @classmethod
    def parse(cls, token: str) -> ProcessId | None:
        parts = token.split(":")
        if len(parts) != 4 or not parts[0]:
            return None
        try:
            pid = int(parts[1])
        except ValueError:
            return None
        if pid <= 0:
            return None
        return cls(parts[0], pid, parts[2], parts[3])


_current: ProcessId | None = None


def current_process() -> ProcessId:
    """This process's id; recomputed in a forked child, whose pid differs."""
    global _current
    pid = os.getpid()
    if _current is None or _current.pid != pid:
        _current = ProcessId(socket.gethostname(), pid, _boot_id(), _start_ticks(pid))
    return _current


def process_token() -> str:
    """The ``benchflow.process`` label value for resources this process makes."""
    return current_process().token()


OwnerState = Literal["this", "alive", "gone", "unknown"]


def owner_state(token: str | None) -> OwnerState:
    """Whether the process a ``benchflow.process`` label names is this one,
    alive, provably gone, or unknown (no label, or another host)."""
    owner = ProcessId.parse(token or "")
    if owner is None:
        return "unknown"
    me = current_process()
    if owner.host != me.host:
        return "unknown"
    if owner.boot and me.boot and owner.boot != me.boot:
        return "gone"
    if owner.pid == me.pid and owner.start == me.start:
        return "this"
    if not _pid_exists(owner.pid):
        return "gone"
    if owner.start:
        started = _start_ticks(owner.pid)
        if started and started != owner.start:
            return "gone"  # the pid now belongs to a later process
    return "alive"


def is_leftover(token: str | None, project: str, live: frozenset[str]) -> bool:
    """Whether a stopped container or unused network may be removed."""
    state = owner_state(token)
    if state == "gone":
        return True
    return state == "this" and bool(project) and project not in live


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """Ids the sweep asked Docker to remove."""

    containers: list[str] = field(default_factory=list)
    networks: list[str] = field(default_factory=list)


def _docker(args: list[str]) -> str | None:
    """Run ``docker *args``; its stdout, or None when it failed."""
    try:
        done = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            timeout=_CLI_TIMEOUT_SEC,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Docker leftover sweep: `docker %s` failed: %s", args[0], exc)
        return None
    if done.returncode != 0:
        logger.debug(
            "Docker leftover sweep: `docker %s` exited %s: %s",
            " ".join(args[:2]),
            done.returncode,
            (done.stderr or "").strip()[:300],
        )
        return None
    return done.stdout or ""


def _leftovers(listing: str | None, live: frozenset[str]) -> list[str]:
    ids: list[str] = []
    for line in (listing or "").splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or not parts[0].strip():
            continue
        resource_id, token, project = (part.strip() for part in parts)
        if is_leftover(token, project, live):
            ids.append(resource_id)
    return ids


def sweep_leftovers() -> SweepResult:
    """Remove BenchFlow containers and networks whose rollout is over.

    Only resources labelled ``benchflow.owned=process`` are listed, and of
    those only the ones :func:`is_leftover` accepts are removed. Best
    effort: a failing ``docker`` call is logged and skipped.
    """
    owned = f"label={OWNED_LABEL}={OWNED_VALUE}"
    statuses = [
        arg for state in _STOPPED_STATES for arg in ("--filter", f"status={state}")
    ]
    containers = _docker(
        [
            "ps",
            "-a",
            "--no-trunc",
            "--filter",
            owned,
            *statuses,
            "--format",
            _LISTING_FORMAT,
        ]
    )
    networks = _docker(
        ["network", "ls", "--no-trunc", "--filter", owned, "--format", _LISTING_FORMAT]
    )
    # Read after listing: a sandbox claims its project before Compose creates
    # anything, so a resource in the listing is never newer than its claim.
    live = live_projects()
    result = SweepResult(_leftovers(containers, live), _leftovers(networks, live))
    if result.containers:
        _docker(["rm", "--volumes", *result.containers])
    if result.networks:
        _docker(["network", "rm", *result.networks])
    if result.containers or result.networks:
        logger.info(
            "Removed leftover BenchFlow Docker resources: %d container(s), %d network(s)",
            len(result.containers),
            len(result.networks),
        )
    return result
