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

A resource of a live process, of another host, of another pid namespace, or
without the label (made by an older BenchFlow) is left alone. Stopped
containers are removed with ``docker rm`` without ``--force``, and networks
with ``docker network rm``; Docker refuses both for a running container and
for a network a running container uses. No container goes with its
volumes (``rm --volumes``); a swept project's volumes go by the label
BenchFlow gave them at creation (:mod:`benchflow.sandbox._trial_volumes`),
without force, so one a container still uses stays.

A *running* container is removed (``docker rm --force``) only when its label
proves its process gone in this host's pid namespace (:func:`is_dead_running`).
After a ``kill -9`` (an OOM kill, a preempted VM) the job's sandboxes keep
running, and ``docker compose up`` can outlive its killed parent and start
more; nothing would ever stop them (DF-p3-eval-runner-9). ``bench sandbox
cleanup`` runs the same sweep, so they can be listed and removed by hand too.

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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from benchflow.sandbox._trial_volumes import LABEL as TRIAL_VOLUME_LABEL

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
# Containers are listed in every state, with their state and name: stopped
# ones are sweep candidates, running ones of a gone process are reported.
_CONTAINER_FORMAT = _LISTING_FORMAT + "\t{{.State}}\t{{.Names}}"
# Running orphans already reported by this process (once each, not per job).
_reported_orphans: set[str] = set()

_RUNNING_STATES = ("running", "paused", "restarting")

_live_projects: set[str] = set()
_live_lock = threading.Lock()


def claim_project(project: str) -> None:
    """Mark this process's compose *project* as in use: the sweep keeps it."""
    with _live_lock:
        _live_projects.add(project)


def release_project(project: str) -> None:
    """*project*'s sandbox is torn down; the sweep may remove what it left."""
    with _live_lock:
        _live_projects.discard(project)


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


def _pid_namespace() -> str:
    """This process's pid namespace inode (Linux), or "".

    Two processes in different pid namespaces (BenchFlow in a container that
    shares the host's name, ``--network host``) can carry the same pid; a
    pid is only checkable in the namespace that issued it.
    """
    try:
        link = os.readlink("/proc/self/ns/pid")
    except OSError:
        return ""
    digits = "".join(c for c in link if c.isdigit())
    return digits


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
    """The ``benchflow.process`` label:
    ``<host>:<pid>:<boot id>:<start ticks>[:<pid namespace>]``.

    Boot id, start time and pid namespace are Linux-only and empty elsewhere;
    without them a pid that exists counts as alive. Labels written before
    0.8.3 have no namespace part.
    """

    host: str
    pid: int
    boot: str = ""
    start: str = ""
    ns: str = ""

    def token(self) -> str:
        base = f"{self.host}:{self.pid}:{self.boot}:{self.start}"
        return f"{base}:{self.ns}" if self.ns else base

    @classmethod
    def parse(cls, token: str) -> ProcessId | None:
        parts = token.split(":")
        if len(parts) not in (4, 5) or not parts[0]:
            return None
        try:
            pid = int(parts[1])
        except ValueError:
            return None
        if pid <= 0:
            return None
        return cls(parts[0], pid, parts[2], parts[3], *parts[4:])


_current: ProcessId | None = None


def current_process() -> ProcessId:
    """This process's id; recomputed in a forked child, whose pid differs."""
    global _current
    pid = os.getpid()
    if _current is None or _current.pid != pid:
        _current = ProcessId(
            socket.gethostname(), pid, _boot_id(), _start_ticks(pid), _pid_namespace()
        )
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
    if owner.ns and me.ns and owner.ns != me.ns:
        return "unknown"  # another pid namespace: its pids are not ours
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


def is_dead_running(token: str | None) -> bool:
    """Whether a *running* container may be force-removed: its process is
    provably gone, and the label names this pid namespace (or another boot),
    so the pid that was checked is the one that made it. A label without the
    namespace part (before 0.8.3) never authorizes a forced removal, not even
    from another boot: its stopped containers are swept as before."""
    if owner_state(token) != "gone":
        return False
    owner = ProcessId.parse(token or "")
    me = current_process()
    if owner is None or not owner.ns:
        return False
    if owner.boot and me.boot and owner.boot != me.boot:
        return True
    return bool(owner.ns) and owner.ns == me.ns


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """What the sweep asked Docker to remove, what Docker removed, and what
    is still held.

    ``containers`` and ``networks`` are the ids asked for (or, in a dry run,
    that would be); Docker refuses a network a running container still uses,
    so ``removed_*`` (the ids Docker printed back) is what the log reports
    (DF-p1-first-run-5: it used to report the request). ``running`` are the
    running containers of dead processes that :func:`is_dead_running`
    allows to force-remove, also in ``containers``; ``projects`` their and
    the other removed resources' compose projects, for a person reading
    ``bench sandbox cleanup``. ``orphans`` names the running containers of a
    BenchFlow process that has exited that are not removed (a label from
    before 0.8.3, or a forced removal Docker refused), so the log says what
    holds the networks and how to clear it.
    """

    containers: list[str] = field(default_factory=list)
    networks: list[str] = field(default_factory=list)
    removed_containers: list[str] = field(default_factory=list)
    removed_networks: list[str] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    # Leftover networks an orphan still uses, so not asked for.
    held_networks: list[str] = field(default_factory=list)
    running: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    # The swept projects' trial volumes Docker removed.
    removed_volumes: list[str] = field(default_factory=list)


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


def _docker_remove(args: list[str], ids: list[str]) -> list[str]:
    """Run ``docker <args> <ids>``; the ids Docker reports it removed.

    ``docker rm`` and ``docker network rm`` print each removed id and exit
    non-zero when any one was refused, so the exit code alone cannot say
    what went.
    """
    try:
        done = subprocess.run(
            ["docker", *args, *ids],
            capture_output=True,
            text=True,
            timeout=_CLI_TIMEOUT_SEC,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Docker leftover sweep: `docker %s` failed: %s", args[0], exc)
        return []
    if done.returncode != 0:
        logger.debug(
            "Docker leftover sweep: `docker %s` exited %s: %s",
            " ".join(args[:2]),
            done.returncode,
            (done.stderr or "").strip()[:300],
        )
    printed = {line.strip() for line in (done.stdout or "").splitlines()}
    return [resource_id for resource_id in ids if resource_id in printed]


def _trial_volumes(projects: set[str]) -> list[str]:
    """The volumes BenchFlow named for a trial of *projects*: those whose
    :data:`TRIAL_VOLUME_LABEL` is one of them (one listing)."""
    if not projects:
        return []
    listing = _docker(
        [
            "volume",
            "ls",
            "--filter",
            f"label={TRIAL_VOLUME_LABEL}",
            "--format",
            '{{.Name}}\t{{.Label "' + TRIAL_VOLUME_LABEL + '"}}',
        ]
    )
    return sorted(name for name, project in _rows(listing, 2) if project in projects)


def _rows(listing: str | None, columns: int = 3) -> list[tuple[str, ...]]:
    rows = []
    for line in (listing or "").splitlines():
        parts = line.split("\t")
        if len(parts) != columns or not parts[0].strip():
            continue
        rows.append(tuple(part.strip() for part in parts))
    return rows


def _container_listing(
    owned: str, states: tuple[str, ...], fmt: str = _LISTING_FORMAT
) -> str | None:
    statuses = [arg for state in states for arg in ("--filter", f"status={state}")]
    return _docker(
        ["ps", "-a", "--no-trunc", "--filter", owned, *statuses, "--format", fmt]
    )


def _report(result: SweepResult) -> None:
    if result.removed_containers or result.removed_networks or result.removed_volumes:
        volumes = len(result.removed_volumes)
        logger.info(
            "Removed leftover BenchFlow Docker resources: %d container(s), "
            "%d network(s)%s",
            len(result.removed_containers),
            len(result.removed_networks),
            f", {volumes} volume(s)" if volumes else "",
        )
    new = [name for name in result.orphans if name not in _reported_orphans]
    if not new:
        return
    _reported_orphans.update(new)
    held = len(result.held_networks)
    held += len(result.networks) - len(result.removed_networks)
    holding = f"; they hold {held} BenchFlow network(s)" if held > 0 else ""
    logger.warning(
        "%d BenchFlow container(s) are still running although the BenchFlow "
        "process that started them has exited: %s%s. BenchFlow force-removes "
        "such a container only when its label proves which process made it; "
        "if you did not keep these on purpose, `docker rm -f %s` removes them "
        "(their networks go at the next run).",
        len(result.orphans),
        ", ".join(result.orphans),
        holding,
        " ".join(result.orphans),
    )


def sweep_leftovers(*, dry_run: bool = False) -> SweepResult:
    """Remove BenchFlow containers and networks whose rollout is over.

    Only resources labelled ``benchflow.owned=process`` are listed, and of
    those only the ones :func:`is_leftover` (stopped containers, networks)
    or :func:`is_dead_running` (running containers) accepts are removed.
    *dry_run* lists them and removes nothing. Best effort: a failing
    ``docker`` call is logged and skipped.
    """
    owned = f"label={OWNED_LABEL}={OWNED_VALUE}"
    # The three listings at once (each is a docker CLI start plus a daemon
    # call, about 15 ms); every job sweeps before and after its trials.
    with ThreadPoolExecutor(max_workers=2) as pool:
        listed_networks = pool.submit(
            _docker,
            [
                "network",
                "ls",
                "--no-trunc",
                "--filter",
                owned,
                "--format",
                _LISTING_FORMAT,
            ],
        )
        listed_running = pool.submit(
            _container_listing, owned, _RUNNING_STATES, _CONTAINER_FORMAT
        )
        stopped = _container_listing(owned, _STOPPED_STATES)
        networks = listed_networks.result()
        running = listed_running.result()
    # Read after listing: a sandbox claims its project before Compose creates
    # anything, so a resource in the listing is never newer than its claim.
    live = live_projects()
    result = SweepResult()
    projects: set[str] = set()
    for resource_id, token, project in _rows(stopped):
        if is_leftover(token, project, live):
            result.containers.append(resource_id)
            projects.add(project)
    # A network a kept running container uses cannot go (Docker refuses);
    # leaving it out keeps a dry run's list to what would really be removed.
    still_running: set[str] = set()
    orphan_projects: set[str] = set()
    names: dict[str, str] = {}
    for resource_id, token, project, _state, name in _rows(running, 5):
        if is_dead_running(token):
            result.running.append(resource_id)
            names[resource_id] = name or resource_id
            projects.add(project)
            continue
        still_running.add(project)
        if owner_state(token) == "gone":
            # Gone, but its label cannot prove which process made it.
            result.orphans.append(name or resource_id)
            orphan_projects.add(project)
    for resource_id, token, project in _rows(networks):
        if project in still_running:
            if project in orphan_projects and is_leftover(token, project, live):
                result.held_networks.append(resource_id)
            continue
        if is_leftover(token, project, live):
            result.networks.append(resource_id)
            projects.add(project)
    result.containers += result.running
    result.projects = sorted(p for p in projects if p)
    if dry_run:
        return result
    stopped_ids = [c for c in result.containers if c not in result.running]
    if stopped_ids:
        result.removed_containers = _docker_remove(["rm"], stopped_ids)
    if result.running:
        # Their process is gone: nothing will ever stop them (kill -9, OOM).
        forced = _docker_remove(["rm", "--force"], result.running)
        result.removed_containers += forced
        result.orphans += [names[c] for c in result.running if c not in forced]
    volumes = _trial_volumes(set(result.projects))
    if volumes:
        # Without force: Docker keeps a volume a container still uses.
        result.removed_volumes = _docker_remove(["volume", "rm"], volumes)
    if result.networks:
        result.removed_networks = _docker_remove(["network", "rm"], result.networks)
    _report(result)
    return result
