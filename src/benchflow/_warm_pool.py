"""The warm pool: the next trials' sandboxes, made while the current ones run.

A Docker job with more trials than ``--concurrency`` spends each trial's
first seconds creating and hardening its sandbox (compose up, the sandbox
user, lockdown, the kept-service freezer proof, the verifier baseline, the
agent install) while its concurrency slot is already held. The pool moves
that work ahead of the slot: when a running trial's sandbox is ready, the
pool begins the setup of the next trial still waiting for a slot
(:meth:`Rollout.begin_prestart`), so when a slot frees that trial finds its
sandbox ready or nearly so.

What an entry is, and what it never is:

- An entry is one queued trial's own rollout, made from the same
  ``RolloutConfig`` its cold start would use, and prepared by the same
  ``setup()``, ``start()`` and ``install_agent()`` calls in the same order.
  Hardening and isolation are therefore the cold start's, step for step.
- An entry is used exactly once, by the trial it was made for, and never
  given to another trial; a retried attempt always starts cold.
- An entry carries the image it was made from (the task's digest and its
  image reference). When the trial takes it and the task now resolves to a
  different image, the entry is removed and the trial starts cold.
- ``--concurrency`` counts running trials; the pool adds at most N prepared
  sandboxes beside them (``--warm-pool N``, 0 to ``--concurrency``; by
  default 1 when two or more trials run at once, 0 for one),
  started, starting or being removed. It begins one per sandbox that becomes
  ready, so it never adds a burst of starts beside the job's own.
- An entry whose trial does not run (a budget or usage stop, an open
  circuit breaker, Ctrl-C, the job's end) is removed: its prestart cancelled,
  its sandbox stopped and deleted, and the trial folder it made (which holds
  no result) removed. :meth:`WarmPool.drain` runs in the job's ``finally``.
- Tasks whose sandbox reserves or shares something two copies cannot (a
  fixed container name, published ports, a fixed or external network, a
  pinned subnet, an external volume, a host path mounted into a service,
  devices or reserved GPUs, a host network, pid, ipc, cgroup or user
  namespace, a privileged service) are never prepared early:
  two trials of such a task must not overlap, and with the pool they could.

Prestarts run in a copy of the job's context, as a cold trial's task does,
never in a sibling trial's (its budget, timing and kill-record context
variables stay its own). ``BENCHFLOW_WARM_POOL=off`` is ``--warm-pool 0``.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
from collections.abc import Callable, Coroutine, Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WARM_POOL_ENV = "BENCHFLOW_WARM_POOL"
_OFF_VALUES = frozenset({"0", "off", "false", "no", "none"})


def warm_pool_requested() -> bool:
    """Whether ``BENCHFLOW_WARM_POOL`` leaves the pool on (the default)."""
    return os.environ.get(WARM_POOL_ENV, "").strip().lower() not in _OFF_VALUES


def warm_pool_size(concurrency: int, requested: int | None = None) -> int:
    """How many sandboxes a job may prepare ahead.

    ``--concurrency`` counts running trials; the pool adds at most this many
    prepared ones. ``--warm-pool N`` (0 to ``concurrency``) sets it; the
    default is 1 when two or more trials run at once and 0 for one, so a
    one-sandbox job stays one sandbox. ``BENCHFLOW_WARM_POOL=off`` is 0.
    """
    if not warm_pool_requested():
        return 0
    if requested is not None:
        return max(0, min(int(requested), concurrency))
    return 1 if concurrency >= 2 else 0


def warm_pool_note(size: int) -> str:
    """What the job's start line says about the pool."""
    return f"warm pool: up to {size} prepared sandbox{'' if size == 1 else 'es'}"


_COMPOSE_NAMES = ("docker-compose.yaml", "docker-compose.yml")
# Service keys that reserve something on the daemon or the host by name, or
# share the host's devices, GPUs or privileges.
_FIXED_SERVICE_KEYS = (
    "container_name",
    "ports",
    "devices",
    "privileged",
    "gpus",
    "runtime",
    # Resolving inheritance is the provider's job. Until it has done so,
    # another service may supply any of the exclusive resources above.
    "extends",
)
# Service keys whose value "host" shares a host namespace.
_HOST_NAMESPACE_KEYS = ("network_mode", "pid", "ipc", "cgroup", "userns_mode")
# The short volume syntax's source names a host path when it starts so;
# a bare word is a named volume of the project.
_HOST_SOURCE_PREFIXES = ("/", ".", "~", "$")


def _host_bind(volume: Any) -> bool:
    """Whether a service's ``volumes`` entry mounts a host path (or cannot be read)."""
    if isinstance(volume, str):
        source = volume.split(":", 1)[0] if ":" in volume else ""
        return source.startswith(_HOST_SOURCE_PREFIXES)
    if isinstance(volume, dict):
        if volume.get("type") == "bind":
            return True
        source = volume.get("source")
        return isinstance(source, str) and source.startswith(_HOST_SOURCE_PREFIXES)
    return True


def _service_shares_host(service: dict[str, Any]) -> bool:
    if any(service.get(key) for key in _FIXED_SERVICE_KEYS):
        return True
    for key in _HOST_NAMESPACE_KEYS:
        value = str(service.get(key) or "")
        if value.startswith("host") or (
            key == "network_mode" and value.startswith("container:")
        ):
            return True
    deploy = service.get("deploy")
    if deploy is not None:
        if not isinstance(deploy, dict):
            return True
        resources = deploy.get("resources") or {}
        if not isinstance(resources, dict) or resources.get("reservations"):
            return True
    # Another container's volumes (outside the project) are shared state.
    shared = service.get("volumes_from") or []
    if not isinstance(shared, list) or any(
        not isinstance(source, str) or source.startswith("container:")
        for source in shared
    ):
        return True
    volumes = service.get("volumes") or []
    if not isinstance(volumes, list):
        return True
    return any(_host_bind(volume) for volume in volumes)


def task_shares_nothing(task_dir: Path) -> bool:
    """Whether two sandboxes of this task can exist at once.

    True for a task with no compose file of its own, or one whose compose
    file names nothing two copies would contend for. Anything unreadable
    counts as shared: the pool then leaves the task to its cold start.
    """
    env_dir = task_dir / "environment"
    files = [env_dir / name for name in _COMPOSE_NAMES if (env_dir / name).exists()]
    if not files:
        return True
    try:
        import yaml

        documents = [yaml.safe_load(path.read_text()) for path in files]
    except Exception:
        return False
    for document in documents:
        if document is None:
            continue
        if (
            not isinstance(document, dict)
            or document.get("name")
            or document.get("include")
        ):
            return False
        services = document.get("services") or {}
        if not isinstance(services, dict):
            return False
        for service in services.values():
            if not isinstance(service, dict):
                return False
            if _service_shares_host(service):
                return False
        for section in ("networks", "volumes"):
            declared = document.get(section) or {}
            if not isinstance(declared, dict):
                return False
            for item in declared.values():
                if item is None:
                    continue
                if not isinstance(item, dict):
                    return False
                if item.get("external") or item.get("name") or item.get("ipam"):
                    return False
                # A volume through driver options (the local driver's bind
                # of a host device or path) or another driver is host state.
                if section == "volumes" and (
                    item.get("driver_opts") or item.get("driver") not in (None, "local")
                ):
                    return False
    return True


def _spawn(work: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
    return asyncio.ensure_future(work)


@dataclass
class _Entry:
    rollout: Any
    key: Hashable
    task: asyncio.Task[None]


class WarmPool:
    """Prepared rollouts for the job's queued trials, by queue position.

    ``make(index)`` returns ``(rollout, key)`` for the trial at that position
    of the job's queue, or None when that trial is not to be prepared;
    ``key(index)`` gives the trial's current image key at the moment it takes
    its entry; ``stopped()`` is true once the job starts no more trials.
    Trials take their slots in queue order, so the next positions without a
    slot are the next to run.
    """

    def __init__(
        self,
        *,
        size: int,
        capacity: int,
        make: Callable[[int], tuple[Any, Hashable] | None],
        key: Callable[[int], Hashable],
        stopped: Callable[[], bool],
    ) -> None:
        self._size = size
        self._capacity = max(0, capacity)
        self._make = make
        self._key = key
        self._stopped = stopped
        self._entries: dict[int, _Entry] = {}
        self._taken: set[int] = set()
        self._next = 0
        self._closed = False
        self._removals: set[asyncio.Task[None]] = set()
        # Prestarts run in the job's context, never in the trial's that
        # happened to report its sandbox ready.
        self._context = contextvars.copy_context()
        self.made = 0
        self.used = 0
        self.removed = 0

    def __len__(self) -> int:
        return len(self._entries)

    def ready(self) -> None:
        """A trial's sandbox is ready: begin preparing the next trial, if any."""
        if self._closed or self._stopped():
            return
        # Entries still being removed hold a sandbox too.
        while (
            self._next < self._size
            and len(self._entries) + len(self._removals) < self._capacity
        ):
            index = self._next
            self._next += 1
            if index in self._taken or index in self._entries:
                continue
            try:
                made = self._context.copy().run(self._begin, index)
            except Exception:
                logger.warning("Warm pool could not prepare a trial", exc_info=True)
                continue
            if made:
                return

    def _begin(self, index: int) -> bool:
        made = self._make(index)
        if made is None:
            return False
        rollout, key = made
        task = rollout.begin_prestart()
        self._entries[index] = _Entry(rollout, key, task)
        self.made += 1
        return True

    def take(self, index: int) -> Any | None:
        """The trial at ``index`` starts: its prepared rollout, or None (cold).

        Called once per trial, when it holds its slot. An entry whose image no
        longer matches the trial's is removed, and the trial starts cold.
        """
        self._taken.add(index)
        self._next = max(self._next, index + 1)
        entry = self._entries.pop(index, None)
        if entry is None:
            return None
        try:
            current = self._key(index)
        except Exception:
            current = None
        if current is None or current != entry.key:
            logger.info(
                "Warm pool: the prepared sandbox for trial %d no longer matches "
                "its image; it starts cold",
                index,
            )
            self._remove(entry)
            return None
        self.used += 1
        return entry.rollout

    def drop(self, index: int) -> None:
        """The trial at ``index`` will not run: remove its entry, if any."""
        self._taken.add(index)
        entry = self._entries.pop(index, None)
        if entry is not None:
            self._remove(entry)

    def _remove(self, entry: _Entry) -> None:
        self.removed += 1
        task = self._context.copy().run(_spawn, entry.rollout.discard_prestart())
        self._removals.add(task)
        task.add_done_callback(self._removals.discard)

    async def drain(self) -> None:
        """Remove every entry no trial took, and wait for every removal."""
        self._closed = True
        for index in list(self._entries):
            self._remove(self._entries.pop(index))
        # Only the removals not done yet: gather completes at once (without
        # yielding, so no done callback runs) when every one already is.
        while pending := [task for task in self._removals if not task.done()]:
            await asyncio.gather(*pending, return_exceptions=True)
        if self.made:
            logger.info(
                "Warm pool: %d sandbox(es) prepared ahead, %d used, %d removed",
                self.made,
                self.used,
                self.removed,
            )
