"""Kill-safe cleanup of the sandboxes this process starts.

A rollout deletes its own sandbox when it ends, including when it is
cancelled (``bench eval run`` turns SIGTERM into a cancellation). What that
cannot cover is a process that dies without running its ``finally`` blocks:
SIGKILL, an out-of-memory kill, a machine that is shut down, a trainer's
thread pool that exits under the event loop. PostTrain Arena once left 84
Daytona sandboxes behind that way, and used up the quota.

So every Docker project and Daytona sandbox BenchFlow starts is also
recorded here, in memory and in a *lease file* for this process
(``~/.cache/benchflow/leases/<host>-<pid>-<start>.json``, or
``$BENCHFLOW_LEASE_DIR``), and removed again when the sandbox is torn down:

- :func:`install_signal_cleanup` makes SIGTERM, SIGINT and SIGHUP delete
  every live sandbox of this process before the previous handler runs (for
  SIGINT that is usually asyncio's cancellation; for SIGTERM the default,
  termination). ``bf.rollout_group`` installs it while groups run.
- :func:`reap_dead_leases` deletes what the lease files of dead processes
  on this machine still list (the process id is gone, or belongs to a later
  process, or the machine rebooted). ``bf.rollout_group`` runs it before a
  group starts, and ``bench sandbox cleanup`` runs it too.
- A lease file cannot help when the machine never comes back. Daytona
  sandboxes also carry the process in their ``benchflow.lease`` label and,
  when a lease time is set, an expiry in ``benchflow.expires``; the Daytona
  reaper deletes a sandbox whose lease names a dead process on this machine
  or whose expiry has passed, and each sandbox stops and deletes itself
  after its idle auto-stop interval (:func:`daytona_lifetime`).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("benchflow")

LEASE_DIR_ENV = "BENCHFLOW_LEASE_DIR"
LEASE_LABEL = "benchflow.lease"
EXPIRES_LABEL = "benchflow.expires"
DAYTONA_AUTO_STOP_ENV = "BENCHFLOW_DAYTONA_AUTO_STOP_MIN"
DAYTONA_AUTO_DELETE_ENV = "BENCHFLOW_DAYTONA_AUTO_DELETE_MIN"
DEFAULT_DAYTONA_AUTO_STOP_MIN = 1440
DEFAULT_DAYTONA_AUTO_DELETE_MIN = 1440
_HANDLED_SIGNALS = tuple(
    sig
    for sig in (
        getattr(signal, "SIGTERM", None),
        getattr(signal, "SIGINT", None),
        getattr(signal, "SIGHUP", None),
    )
    if sig is not None
)
_CLI_TIMEOUT_SEC = 60

_lock = threading.RLock()  # re-entrant: the signal handler runs in the main thread
_live: dict[tuple[str, str], dict[str, Any]] = {}


# --- lifetimes (per rollout, set by the RL rollout API) ---------------------


@dataclass(frozen=True)
class SandboxLifetime:
    """How long a sandbox may outlive the process that started it.

    ``lease_sec`` stamps Daytona sandboxes with an expiry
    (``benchflow.expires``) after which the reaper deletes them, whatever
    process is still alive. ``auto_stop_min`` is Daytona's idle auto-stop
    interval (a sandbox with no activity for that long stops), and
    ``auto_delete_min`` the time a stopped sandbox is kept (0: deleted as
    soon as it stops).
    """

    lease_sec: float | None = None
    auto_stop_min: int | None = None
    auto_delete_min: int | None = None


_LIFETIME: ContextVar[SandboxLifetime | None] = ContextVar(
    "benchflow_sandbox_lifetime", default=None
)


@contextlib.contextmanager
def sandbox_lifetime(lifetime: SandboxLifetime | None) -> Iterator[None]:
    """Use ``lifetime`` for sandboxes started inside the block."""
    token = _LIFETIME.set(lifetime)
    try:
        yield
    finally:
        _LIFETIME.reset(token)


def _env_minutes(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a whole number of minutes; using %d", name, raw, default
        )
        return default
    return max(value, 0)


def daytona_lifetime() -> tuple[int, int]:
    """``(auto_stop_min, auto_delete_min)`` for a new Daytona sandbox.

    The rollout's :class:`SandboxLifetime` wins, then
    ``BENCHFLOW_DAYTONA_AUTO_STOP_MIN`` / ``BENCHFLOW_DAYTONA_AUTO_DELETE_MIN``,
    then one day each.
    """
    lifetime = _LIFETIME.get()
    stop = _env_minutes(DAYTONA_AUTO_STOP_ENV, DEFAULT_DAYTONA_AUTO_STOP_MIN)
    delete = _env_minutes(DAYTONA_AUTO_DELETE_ENV, DEFAULT_DAYTONA_AUTO_DELETE_MIN)
    if lifetime is not None and lifetime.auto_stop_min is not None:
        stop = lifetime.auto_stop_min
    if lifetime is not None and lifetime.auto_delete_min is not None:
        delete = lifetime.auto_delete_min
    return stop, delete


def lease_labels() -> dict[str, str]:
    """Labels that tie a new Daytona sandbox to this process (and its expiry)."""
    labels = {LEASE_LABEL: lease_token()}
    lifetime = _LIFETIME.get()
    if lifetime is not None and lifetime.lease_sec is not None:
        labels[EXPIRES_LABEL] = str(int(time.time() + lifetime.lease_sec))
    return labels


# --- who owns a resource ----------------------------------------------------


def _process() -> Any:
    from benchflow.sandbox._docker_sweep import current_process

    return current_process()


def lease_token() -> str:
    """``<host>:<pid>:<boot id prefix>:<start ticks>``, short enough for a label."""
    me = _process()
    host = re.sub(r"[^A-Za-z0-9_.-]+", "-", me.host)[:40]
    return f"{host}:{me.pid}:{me.boot[:8]}:{me.start}"


def lease_state(token: str | None) -> str:
    """Whether the process a lease names is ``this`` one, ``alive``, ``gone``, or ``unknown``.

    ``unknown`` covers another machine and an unreadable token: only a
    process provably gone on this machine counts as dead.
    """
    from benchflow.sandbox._docker_sweep import _pid_exists, _start_ticks

    parts = (token or "").split(":")
    if len(parts) != 4 or not parts[0]:
        return "unknown"
    host, pid_text, boot, start = parts
    try:
        pid = int(pid_text)
    except ValueError:
        return "unknown"
    me = _process()
    my_host = re.sub(r"[^A-Za-z0-9_.-]+", "-", me.host)[:40]
    if pid <= 0 or host != my_host:
        return "unknown"
    if boot and me.boot and not me.boot.startswith(boot):
        return "gone"
    if pid == me.pid and start == me.start:
        return "this"
    if not _pid_exists(pid):
        return "gone"
    if start:
        now_start = _start_ticks(pid)
        if now_start and now_start != start:
            return "gone"
    return "alive"


# --- the registry and this process's lease file ------------------------------


def lease_dir() -> Path:
    raw = os.environ.get(LEASE_DIR_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".cache" / "benchflow" / "leases"


def _lease_path() -> Path:
    me = _process()
    host = re.sub(r"[^A-Za-z0-9_.-]+", "-", me.host)[:40]
    return lease_dir() / f"{host}-{me.pid}-{me.start or '0'}.json"


def _write_lease_file() -> None:
    path = _lease_path()
    resources = list(_live.values())
    try:
        if not resources:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "process": lease_token(),
                    "written_at": time.time(),
                    "resources": resources,
                },
                indent=1,
            )
        )
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("Could not write sandbox lease file %s: %s", path, exc)


def record(provider: str, resource_id: str, **extra: Any) -> None:
    """A sandbox of ``provider`` (``docker`` project or ``daytona`` id) is live."""
    if not resource_id:
        return
    entry = {
        "provider": provider,
        "id": resource_id,
        "created_at": time.time(),
        **extra,
    }
    with _lock:
        _live[(provider, resource_id)] = entry
        _write_lease_file()


def release(provider: str, resource_id: str) -> None:
    """The sandbox is torn down: nothing to clean up for it any more."""
    with _lock:
        if _live.pop((provider, resource_id), None) is not None:
            _write_lease_file()


def live() -> list[dict[str, Any]]:
    """The sandboxes this process has started and not yet torn down."""
    with _lock:
        return [dict(entry) for entry in _live.values()]


# --- deletion ----------------------------------------------------------------


def _docker(args: list[str]) -> str | None:
    try:
        done = subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=_CLI_TIMEOUT_SEC
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def _delete_docker_project(project: str) -> bool:
    label = f"label=com.docker.compose.project={project}"
    containers = (_docker(["ps", "-aq", "--filter", label]) or "").split()
    if containers:
        _docker(["rm", "-f", "-v", *containers])
    networks = (_docker(["network", "ls", "-q", "--filter", label]) or "").split()
    if networks:
        _docker(["network", "rm", *networks])
    left = (_docker(["ps", "-aq", "--filter", label]) or "").split()
    return not left


def _delete_daytona(sandbox_ids: list[str]) -> dict[str, bool]:
    try:
        from benchflow.sandbox.daytona import build_sync_client

        client = build_sync_client()
    except Exception as exc:
        logger.warning("Daytona cleanup: no client (%s)", type(exc).__name__)
        return dict.fromkeys(sandbox_ids, False)
    outcome: dict[str, bool] = {}
    for sandbox_id in sandbox_ids:
        try:
            sandbox = client.get(sandbox_id)
        except Exception as exc:
            # Already gone is success; anything else we cannot tell.
            gone = "404" in str(exc) or "not found" in str(exc).lower()
            outcome[sandbox_id] = gone
            continue
        try:
            client.delete(sandbox)
            outcome[sandbox_id] = True
        except Exception as exc:
            gone = "404" in str(exc) or "not found" in str(exc).lower()
            outcome[sandbox_id] = gone
            if not gone:
                logger.warning(
                    "Daytona cleanup: could not delete %s (%s)",
                    sandbox_id,
                    type(exc).__name__,
                )
    return outcome


def delete_resources(
    resources: Iterable[dict[str, Any]], *, timeout_sec: float = 120.0
) -> dict[str, list[str]]:
    """Delete sandboxes by provider and id; ``{"deleted": [...], "failed": [...]}``.

    Docker projects are removed with ``docker rm -f`` by their Compose
    project label (containers and networks); Daytona sandboxes by id. Each
    provider runs in its own thread; the whole call is bounded by
    ``timeout_sec``, and what did not finish in time counts as failed.
    """
    items = list(resources)
    docker = [r["id"] for r in items if r.get("provider") == "docker"]
    daytona = [r["id"] for r in items if r.get("provider") == "daytona"]
    results: dict[str, bool] = {}

    def _docker_all() -> None:
        for project in docker:
            results[f"docker:{project}"] = _delete_docker_project(project)

    def _daytona_all() -> None:
        for sandbox_id, ok in _delete_daytona(daytona).items():
            results[f"daytona:{sandbox_id}"] = ok

    jobs = [
        job
        for job, needed in ((_docker_all, docker), (_daytona_all, daytona))
        if needed
    ]
    if jobs:
        pool = ThreadPoolExecutor(max_workers=len(jobs), thread_name_prefix="bf-lease")
        futures = [pool.submit(job) for job in jobs]
        wait(futures, timeout=timeout_sec)
        pool.shutdown(wait=False, cancel_futures=True)
    wanted = [f"docker:{p}" for p in docker] + [f"daytona:{s}" for s in daytona]
    return {
        "deleted": [key for key in wanted if results.get(key) is True],
        "failed": [key for key in wanted if results.get(key) is not True],
    }


def emergency_cleanup(*, timeout_sec: float = 90.0) -> dict[str, list[str]]:
    """Delete every sandbox this process still has live; forget the deleted ones."""
    entries = live()
    if not entries:
        return {"deleted": [], "failed": []}
    outcome = delete_resources(entries, timeout_sec=timeout_sec)
    deleted = set(outcome["deleted"])
    for entry in entries:
        if f"{entry['provider']}:{entry['id']}" in deleted:
            release(entry["provider"], entry["id"])
    return outcome


# --- signals -----------------------------------------------------------------

_installed = 0
_previous: dict[int, Any] = {}
_cleaning = threading.Event()


def _on_signal(signum: int, frame: Any) -> None:
    if not _cleaning.is_set():
        _cleaning.set()
        try:
            entries = live()
            if entries:
                with contextlib.suppress(Exception):
                    os.write(
                        2,
                        (
                            f"benchflow: signal {signum}: deleting {len(entries)} "
                            "live sandbox(es) before exiting\n"
                        ).encode(),
                    )
                emergency_cleanup()
        except Exception:  # never let cleanup hide the signal
            logger.debug("emergency sandbox cleanup failed", exc_info=True)
        finally:
            _cleaning.clear()
    previous = _previous.get(signum, signal.SIG_DFL)
    if callable(previous):
        previous(signum, frame)
    elif previous == signal.SIG_DFL:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


def install_signal_cleanup() -> Callable[[], None]:
    """Delete this process's live sandboxes on SIGTERM, SIGINT and SIGHUP.

    The previous handlers run afterwards, so asyncio's Ctrl-C cancellation
    and a caller's own handlers still work. Nested installs are counted;
    the returned function uninstalls one. Outside the main thread signals
    cannot be handled, so it does nothing there.
    """
    global _installed
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    with _lock:
        if _installed == 0:
            for sig in _HANDLED_SIGNALS:
                try:
                    _previous[sig] = signal.signal(sig, _on_signal)
                except (OSError, ValueError):
                    continue
        _installed += 1

    done = False

    def _uninstall() -> None:
        global _installed
        nonlocal done
        if done:
            return
        done = True
        with _lock:
            _installed -= 1
            if _installed == 0:
                for sig, previous in list(_previous.items()):
                    with contextlib.suppress(OSError, ValueError, TypeError):
                        if signal.getsignal(sig) is _on_signal:
                            signal.signal(sig, previous)
                _previous.clear()

    return _uninstall


# --- reaping what dead processes left ------------------------------------------


def reap_dead_leases(*, timeout_sec: float = 180.0) -> dict[str, Any]:
    """Delete the sandboxes that lease files of dead processes still list.

    Only lease files of processes provably gone on this machine are acted
    on (:func:`lease_state`); a file whose resources are all deleted is
    removed, one with failures is kept for the next try.
    """
    directory = lease_dir()
    report: dict[str, Any] = {"leases": 0, "deleted": [], "failed": []}
    if not directory.is_dir():
        return report
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or lease_state(data.get("process")) != "gone":
            continue
        resources = [
            r
            for r in data.get("resources") or []
            if isinstance(r, dict)
            and r.get("provider") in {"docker", "daytona"}
            and r.get("id")
        ]
        report["leases"] += 1
        outcome = delete_resources(resources, timeout_sec=timeout_sec)
        report["deleted"] += outcome["deleted"]
        report["failed"] += outcome["failed"]
        if outcome["failed"]:
            kept = [
                r
                for r in resources
                if f"{r['provider']}:{r['id']}" in outcome["failed"]
            ]
            with contextlib.suppress(OSError):
                path.write_text(json.dumps({**data, "resources": kept}, indent=1))
        else:
            path.unlink(missing_ok=True)
    if report["deleted"] or report["failed"]:
        logger.info(
            "Reaped sandboxes left by dead processes: %d deleted, %d failed",
            len(report["deleted"]),
            len(report["failed"]),
        )
    return report


__all__ = [
    "EXPIRES_LABEL",
    "LEASE_LABEL",
    "SandboxLifetime",
    "daytona_lifetime",
    "delete_resources",
    "emergency_cleanup",
    "install_signal_cleanup",
    "lease_dir",
    "lease_labels",
    "lease_state",
    "lease_token",
    "live",
    "reap_dead_leases",
    "record",
    "release",
    "sandbox_lifetime",
]
