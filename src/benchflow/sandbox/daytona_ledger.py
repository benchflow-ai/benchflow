"""The process's ledger of Daytona resources whose removal is not confirmed.

DAYTONA-START round 3, rule D. A create attempt with an unknown outcome (a
timeout, a 5xx, a transport error, a cancellation, a failed lookup), an
attempt's sandbox whose delete failed, and a trial sandbox whose delete at
``stop()`` failed are entries here until their removal is confirmed: a lookup
by the attempt's label found sandboxes and every one was deleted, or the
sandbox's delete went through (or Daytona answered 404). Elapsed time never
closes an entry while the job lives.

- While the job's event loop runs, one sweeper per loop looks every
  :data:`SWEEP_INTERVAL_SEC` (``ensure_sweeper``); ``stop()`` sweeps its own
  entries until they are resolved or :data:`SETTLE_SEC` after their failure
  (``drain``), and leaves the rest here.
- At process exit (``atexit``, sync client) the exit sweep retries lookups and
  failed deletes for up to :data:`EXIT_SWEEP_SEC`.
- Every entry is written to a durable file when it opens and when it is
  resolved, so one left by an exit sweep that ran out of budget, or by a
  killed process, is read by ``bench sandbox cleanup`` (``cleanup_leftovers``).
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import importlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchflow.sandbox.daytona_delete import (
    delete_sandbox,
    delete_sandbox_sync,
    list_sandboxes_sync,
)


def _load_fcntl() -> Any:
    """``fcntl``, or None where the platform has none (Windows)."""
    try:
        return importlib.import_module("fcntl")
    except ImportError:  # pragma: no cover - Windows
        return None


fcntl: Any = _load_fcntl()

logger = logging.getLogger("benchflow")

LEFTOVERS_ENV = "BENCHFLOW_DAYTONA_LEFTOVERS"
CREATE_ATTEMPT_LABEL = "benchflow.create-attempt"
# How often a living job looks for its open entries.
SWEEP_INTERVAL_SEC = 10.0
# How long stop() and the exit sweep keep looking for an attempt whose
# sandbox has not landed: the Rust bench's landing window, inside the
# separate verifier's 120 s stop bound. The entry stays open afterwards.
SETTLE_SEC = 90.0
# The exit sweep's whole budget: lookups and failed deletes are retried with
# backoff until it is spent.
EXIT_SWEEP_SEC = 60.0
# `bench sandbox cleanup` closes an attempt that a lookup finds empty only
# once it is this old: a sandbox that lands later carries Daytona's
# inactivity stop and a lapsing lease, so the auto-reap removes it.
FORGET_EMPTY_ATTEMPT_SEC = 3600.0


def leftovers_path() -> Path:
    configured = os.environ.get(LEFTOVERS_ENV, "").strip()
    if configured:
        return Path(configured)
    state = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(state) if state else Path.home() / ".local" / "state"
    return base / "benchflow" / "daytona-leftovers.jsonl"


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(
        path.with_suffix(".lock"),
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _append(record: dict[str, Any]) -> None:
    """Best effort: a ledger that cannot be written still sweeps in memory."""
    path = leftovers_path()
    try:
        with _locked(path), path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        logger.debug("Could not write the Daytona leftovers file %s: %s", path, exc)


def _fold(lines: list[str]) -> dict[str, dict[str, Any]]:
    """The open entries of the file: every ``open`` without a later ``done``."""
    entries: dict[str, dict[str, Any]] = {}
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or not isinstance(record.get("key"), str):
            continue
        if record.get("op") == "open":
            entries[record["key"]] = record
        elif record.get("op") == "done":
            entries.pop(record["key"], None)
    return entries


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def read_leftovers(path: Path | None = None) -> list[dict[str, Any]]:
    return list(_fold(_read_lines(path or leftovers_path())).values())


@dataclass
class Entry:
    key: str
    kind: str  # "attempt" (an attempt label) or "sandbox" (an id)
    labels: dict[str, str]
    sandbox_id: str | None
    managed: str
    opened_at: float  # wall clock, for the file and `bench sandbox cleanup`
    settle_at: float  # monotonic: when sweeps stop waiting for it to land
    reason: str
    client: Any = None
    handle: Any = None
    holder: Any = None
    loop: Any = None
    on_resolved: Callable[[], None] | None = None
    busy: bool = False
    found: bool = False
    errors: list[str] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return {
            "op": "open",
            "key": self.key,
            "kind": self.kind,
            "labels": self.labels,
            "sandbox_id": self.sandbox_id,
            "managed": self.managed,
            "opened_at": round(self.opened_at, 3),
            "reason": self.reason,
            "pid": os.getpid(),
        }


def _is_not_found(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    return status == 404 or type(exc).__name__ == "DaytonaNotFoundError"


def _labels_query(labels: dict[str, str]) -> Any:
    try:
        from daytona import ListSandboxesQuery
    except ImportError:  # a faked client (tests without the Daytona extra)
        import types

        return types.SimpleNamespace(labels=labels)
    return ListSandboxesQuery(labels=labels)


def _one_line(exc: BaseException) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:300]


class Ledger:
    def __init__(self) -> None:
        self.entries: dict[str, Entry] = {}
        self._sweepers: dict[Any, asyncio.Task[None]] = {}
        self._exit_registered = False

    # Opening and closing entries.

    def _open(self, entry: Entry) -> Entry:
        known = self.entries.get(entry.key)
        if known is not None:
            return known
        self.entries[entry.key] = entry
        _append(entry.record())
        if not self._exit_registered:
            atexit.register(self.exit_sweep)
            self._exit_registered = True
        return entry

    def open_attempt(
        self,
        client: Any,
        labels: dict[str, str],
        reason: str,
        *,
        managed: str,
        holder: Any = None,
        on_resolved: Callable[[], None] | None = None,
    ) -> Entry:
        attempt = labels[CREATE_ATTEMPT_LABEL]
        return self._open(
            Entry(
                key=f"attempt:{attempt}",
                kind="attempt",
                labels={CREATE_ATTEMPT_LABEL: attempt},
                sandbox_id=None,
                managed=managed,
                opened_at=time.time(),
                settle_at=time.monotonic() + SETTLE_SEC,
                reason=reason,
                client=client,
                holder=holder,
                loop=_running_loop(),
                on_resolved=on_resolved,
            )
        )

    def open_sandbox(
        self,
        client: Any,
        sandbox_id: str,
        reason: str,
        *,
        managed: str,
        handle: Any = None,
        holder: Any = None,
        on_resolved: Callable[[], None] | None = None,
    ) -> Entry:
        # A known sandbox is not waited for: it exists, only its delete failed.
        return self._open(
            Entry(
                key=f"sandbox:{sandbox_id}",
                kind="sandbox",
                labels={},
                sandbox_id=sandbox_id,
                managed=managed,
                opened_at=time.time(),
                settle_at=time.monotonic(),
                reason=reason,
                client=client,
                handle=handle,
                holder=holder,
                loop=_running_loop(),
                on_resolved=on_resolved,
            )
        )

    def _resolve(self, entry: Entry) -> None:
        if self.entries.pop(entry.key, None) is None:
            return
        _append({"op": "done", "key": entry.key})
        if entry.on_resolved is not None:
            with contextlib.suppress(Exception):
                entry.on_resolved()

    def held_by(self, holder: Any) -> list[Entry]:
        return [e for e in self.entries.values() if e.holder is holder]

    def close_sandbox(self, sandbox_id: str) -> None:
        """A later delete of this sandbox went through (or it is gone)."""
        entry = self.entries.get(f"sandbox:{sandbox_id}")
        if entry is not None:
            self._resolve(entry)

    def close_attempt(self, labels: dict[str, str]) -> None:
        """The attempt was refused or all of its allocations are confirmed gone."""
        entry = self.entries.get(f"attempt:{labels[CREATE_ATTEMPT_LABEL]}")
        if entry is not None:
            self._resolve(entry)

    # Asynchronous sweeps (the job's own event loop).

    async def _sweep_one(self, entry: Entry) -> bool:
        """One look at *entry*; True once its removal is confirmed."""
        if entry.busy:
            return False
        entry.busy = True
        try:
            if entry.kind == "sandbox":
                try:
                    handle = entry.handle or await entry.client.get(entry.sandbox_id)
                    await delete_sandbox(handle)
                except Exception as exc:
                    if not _is_not_found(exc):
                        entry.errors.append(_one_line(exc))
                        return False
                logger.info(f"Deleted Daytona sandbox {entry.sandbox_id} (ledger)")
                self._resolve(entry)
                return True
            try:
                found = [
                    s async for s in entry.client.list(_labels_query(entry.labels))
                ]
            except Exception as exc:
                entry.errors.append(_one_line(exc))
                return False
            if not found:
                return False
            entry.found = True
            deleted = True
            for sandbox in found:
                try:
                    await delete_sandbox(sandbox)
                except Exception as exc:
                    if not _is_not_found(exc):
                        deleted = False
                        entry.errors.append(_one_line(exc))
                        logger.warning(
                            f"Could not delete sandbox {getattr(sandbox, 'id', '?')} "
                            f"of create attempt {entry.key}: {_one_line(exc)}; "
                            "it stays in the ledger and is retried"
                        )
                        continue
                logger.info(
                    f"Deleted sandbox {getattr(sandbox, 'id', '?')} of create "
                    f"attempt {entry.key}"
                )
            if deleted:
                self._resolve(entry)
            return deleted
        finally:
            entry.busy = False

    async def sweep(self, entries: list[Entry] | None = None) -> None:
        loop = _running_loop()
        for entry in list(self.entries.values() if entries is None else entries):
            if entry.key in self.entries and entry.loop is loop:
                await self._sweep_one(entry)

    def ensure_sweeper(self) -> None:
        """One sweeper per event loop while any of its entries is open."""
        loop = _running_loop()
        if loop is None:
            return
        task = self._sweepers.get(loop)
        if task is None or task.done():
            self._sweepers[loop] = loop.create_task(self._sweeper(loop))

    async def _sweeper(self, loop: Any) -> None:
        try:
            while any(e.loop is loop for e in self.entries.values()):
                await asyncio.sleep(SWEEP_INTERVAL_SEC)
                await self.sweep()
        finally:
            self._sweepers.pop(loop, None)

    async def drain(self, holder: Any) -> None:
        """stop(): sweep *holder*'s create attempts until each is resolved or
        past its settle time; what is still open stays in the ledger. A
        sandbox whose delete failed is not retried here (stop() already
        retried it, and the caller may still hold its handle): the job's
        sweeper and the exit sweep retry it."""

        def attempts() -> list[Entry]:
            return [e for e in self.held_by(holder) if e.kind == "attempt"]

        while True:
            await self.sweep(attempts())
            now = time.monotonic()
            waiting = [e for e in attempts() if e.settle_at > now and not e.busy]
            if not waiting:
                break
            next_settle = min(e.settle_at for e in waiting)
            await asyncio.sleep(max(0.0, min(SWEEP_INTERVAL_SEC, next_settle - now)))
        if self.held_by(holder):
            self.ensure_sweeper()

    # The exit sweep (synchronous: the job's event loop is gone).

    def exit_sweep(
        self,
        client_factory: Callable[[], Any] | None = None,
        *,
        budget_sec: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> list[Entry]:
        """Retry every open entry with the sync client for up to *budget_sec*;
        return (and log) what is still open. Each is already in the file."""
        if not self.entries:
            return []
        try:
            if client_factory is None:
                from benchflow.sandbox.daytona import build_sync_client

                client_factory = build_sync_client
            client = client_factory()
        except Exception as exc:
            logger.debug(f"Daytona exit sweep could not open a client: {exc!r}")
            client = None
        deadline = time.monotonic() + (
            EXIT_SWEEP_SEC if budget_sec is None else budget_sec
        )
        wait = 1.0
        while client is not None and self.entries:
            for entry in list(self.entries.values()):
                if time.monotonic() >= deadline:
                    break
                self._sweep_one_sync(client, entry, deadline=deadline)
            now = time.monotonic()
            # Keep going while a delete or lookup failed, or an attempt may
            # still land; never past the budget.
            pending = [
                e
                for e in self.entries.values()
                if e.kind == "sandbox" or e.found or e.errors or e.settle_at > now
            ]
            if not pending or now + wait > deadline:
                break
            sleep(wait)
            wait = min(wait * 2, 10.0)
        left = list(self.entries.values())
        if left:
            logger.warning(
                "Daytona: %d sandbox(es) or create attempt(s) could not be "
                "confirmed deleted before exit: %s. They are recorded in %s; "
                "`bench sandbox cleanup` deletes them.",
                len(left),
                ", ".join(e.sandbox_id or e.key for e in left),
                leftovers_path(),
            )
        return left

    def _sweep_one_sync(
        self, client: Any, entry: Entry, *, deadline: float | None = None
    ) -> None:
        def remaining() -> float:
            from benchflow.sandbox.daytona_delete import DELETE_TIMEOUT_SEC

            budget = DELETE_TIMEOUT_SEC
            if deadline is not None:
                budget = min(budget, deadline - time.monotonic())
            if budget <= 0:
                raise TimeoutError("Daytona exit cleanup deadline reached")
            return budget

        entry.errors.clear()
        if entry.kind == "sandbox":
            try:
                sandbox = client.get(entry.sandbox_id, request_timeout=remaining())
                delete_sandbox_sync(client, sandbox, timeout=remaining())
            except Exception as exc:
                if not _is_not_found(exc):
                    entry.errors.append(_one_line(exc))
                    return
            self._resolve(entry)
            return
        try:
            found = list_sandboxes_sync(
                client, _labels_query(entry.labels), timeout=remaining()
            )
        except Exception as exc:
            entry.errors.append(_one_line(exc))
            return
        if not found:
            return
        entry.found = True
        for sandbox in found:
            try:
                delete_sandbox_sync(client, sandbox, timeout=remaining())
            except Exception as exc:
                if not _is_not_found(exc):
                    entry.errors.append(_one_line(exc))
        if not entry.errors:
            self._resolve(entry)


def _running_loop() -> Any:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


LEDGER = Ledger()


def cleanup_leftovers(
    client: Any,
    managed: str,
    *,
    dry_run: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    """``bench sandbox cleanup``: delete the open entries of the leftovers file
    that belong to *managed* (this owner scope), then compact the file.

    Each entry was given up by its own run (its create never returned the
    sandbox, or its delete failed), so it is deleted whatever its lease. An
    attempt whose lookup is empty is closed once it is older than
    :data:`FORGET_EMPTY_ATTEMPT_SEC`. Returns counts and the ids deleted.
    """
    now = time.time() if now is None else now
    path = leftovers_path()
    counts: dict[str, Any] = {
        "open": 0,
        "deleted": 0,
        "closed": 0,
        "kept": 0,
        "failed": 0,
        "other_owner": 0,
        "ids": [],
    }
    done: list[str] = []
    # The lock covers only the file reads and the compaction: a job on this
    # host appends to the file from its event loop, so the lookups and
    # deletes run outside it.
    with _locked(path):
        entries = _fold(_read_lines(path))
    for key, record in entries.items():
        if record.get("managed") != managed:
            counts["other_owner"] += 1
            continue
        counts["open"] += 1
        if record.get("kind") == "sandbox":
            sandbox_id = str(record.get("sandbox_id") or "")
            if dry_run:
                counts["deleted"] += 1
                counts["ids"].append(sandbox_id)
                continue
            try:
                delete_sandbox_sync(client, client.get(sandbox_id))
            except Exception as exc:
                if not _is_not_found(exc):
                    counts["failed"] += 1
                    continue
                counts["closed"] += 1
            else:
                counts["deleted"] += 1
                counts["ids"].append(sandbox_id)
            done.append(key)
            continue
        labels = record.get("labels")
        if not isinstance(labels, dict) or CREATE_ATTEMPT_LABEL not in labels:
            continue
        try:
            found = list(client.list(_labels_query(labels)))
        except Exception:
            counts["failed"] += 1
            continue
        if not found:
            if now - float(record.get("opened_at") or now) >= (
                FORGET_EMPTY_ATTEMPT_SEC
            ):
                counts["closed"] += 1
                if not dry_run:
                    done.append(key)
            else:
                counts["kept"] += 1
            continue
        failed = False
        for sandbox in found:
            counts["ids"].append(str(getattr(sandbox, "id", "?")))
            if dry_run:
                counts["deleted"] += 1
                continue
            try:
                delete_sandbox_sync(client, sandbox)
            except Exception as exc:
                if not _is_not_found(exc):
                    failed = True
                    continue
            counts["deleted"] += 1
        if failed:
            counts["failed"] += 1
        elif not dry_run:
            done.append(key)
    if done:
        with _locked(path):
            # Read again: entries opened or closed meanwhile are kept as
            # they are now.
            keep = [r for k, r in _fold(_read_lines(path)).items() if k not in done]
            tmp = path.with_suffix(".tmp")
            try:
                tmp.write_text(
                    "".join(json.dumps(r, sort_keys=True) + "\n" for r in keep),
                    encoding="utf-8",
                )
                os.replace(tmp, path)
            except OSError as exc:
                logger.warning(f"Could not compact {path}: {exc}")
    return counts
