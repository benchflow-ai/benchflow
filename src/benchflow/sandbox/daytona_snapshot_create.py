"""Create a Daytona snapshot, and account for one that outlives its call.

``AsyncSandbox._experimental_create_snapshot`` asks Daytona for a snapshot and
then waits for the sandbox to leave SNAPSHOTTING, within the SDK's default of
60 s. A sandbox holding a real agent's install (claude-agent-acp, about 3 GiB)
takes about 4 minutes, so the call failed with ``DaytonaTimeoutError`` while
Daytona went on to finish a 2.5 GB snapshot that nothing recorded.

The create now gets :func:`snapshot_timeout_sec` (default 600 s). When it fails
anyway, the snapshot is looked up by its name (never by listing) until Daytona
shows a final state, for up to ``look_sec``:

* ``active``: the snapshot is the result after all, a late success
  (:data:`LATE_META` in its handle says so), once the sandbox has left
  SNAPSHOTTING too;
* ``error`` or ``build_failed``: it is deleted until it is gone;
* never seen: nothing is deleted (``none_found``);
* still being made when the look ends: ``delete_deferred``. A delete then would
  not hold (on Daytona it shows ``removing``, then the snapshot turns active),
  so it is left to :func:`settle_leftovers` at the sandbox's stop, which waits
  for it again and deletes it.

A delete Daytona refuses, or one that doesn't hold, is tried again until the
snapshot is gone or the look ends; ``removed`` means Daytona no longer lists it.
Lookups that fail are counted in the record with the last error, so a look in
an API outage says why it found nothing. The create's error is re-raised with
the record at :data:`LEFTOVER_ATTR`, for the caller's own record
(``checkpoints.json``, a fork's record). A cancelled create or look leaves its
name to the stop too, and the stop writes every leftover it settles to
:data:`LEFTOVERS_FILE` in the trial's folder. Only the snapshot of this exact
name is looked up and deleted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger("benchflow")

SNAPSHOT_TIMEOUT_ENV = "BENCHFLOW_DAYTONA_SNAPSHOT_TIMEOUT_SEC"
SNAPSHOT_TIMEOUT_DEFAULT_SEC = 600.0
#: On the error a failed create raises: what became of its snapshot.
LEFTOVER_ATTR = "benchflow_snapshot_leftover"
#: In a late snapshot's handle: the create's error and how much later it was
#: active (JSON).
LATE_META = "benchflow_snapshot_late"
#: In the trial's folder: every snapshot the sandbox's stop settled.
LEFTOVERS_FILE = "daytona-snapshot-leftovers.json"

_FINAL_FAILED = frozenset({"error", "build_failed"})
_ABSENT = "absent"


def snapshot_timeout_sec() -> float:
    """Seconds a snapshot create may take: ``BENCHFLOW_DAYTONA_SNAPSHOT_TIMEOUT_SEC``
    when it is a finite number above 0, else 600 (an unusable value is warned)."""
    raw = os.environ.get(SNAPSHOT_TIMEOUT_ENV, "").strip()
    if not raw:
        return SNAPSHOT_TIMEOUT_DEFAULT_SEC
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if math.isfinite(value) and value > 0:
        return value
    logger.warning(
        "Invalid %s=%r; a Daytona snapshot gets %.0fs",
        SNAPSHOT_TIMEOUT_ENV,
        raw,
        SNAPSHOT_TIMEOUT_DEFAULT_SEC,
    )
    return SNAPSHOT_TIMEOUT_DEFAULT_SEC


async def create_snapshot(
    sandbox: Any,
    name: str,
    leftovers: dict[str, dict[str, Any]],
    *,
    look_sec: float,
    poll_sec: float,
) -> dict[str, str]:
    """Create snapshot ``name`` of ``sandbox`` (the SDK's ``AsyncSandbox``).

    Returns the meta to add to its handle: empty when the create returned,
    :data:`LATE_META` when it failed but the snapshot turned active. Otherwise
    the create's error is raised with the record at :data:`LEFTOVER_ATTR`; a
    snapshot left to the stop is in ``leftovers`` under its name.
    """
    timeout = snapshot_timeout_sec()
    loop = asyncio.get_running_loop()
    started = loop.time()
    record: dict[str, Any] = {"name": name, "timeout_sec": timeout}
    try:
        try:
            await sandbox._experimental_create_snapshot(name, timeout=timeout)
        except Exception as exc:
            failure = exc
        else:
            return {}
        record.update(
            cause=type(failure).__name__,
            create_sec=round(loop.time() - started, 1),
        )
        deadline = loop.time() + look_sec
        await _look(
            name,
            record,
            deadline=deadline,
            poll_sec=poll_sec,
            made_nothing=_made_nothing(failure),
        )
        if record["outcome"] == "late":
            await _sandbox_back(sandbox, name, deadline=deadline, poll_sec=poll_sec)
    except asyncio.CancelledError:
        record.setdefault("cause", "cancelled")
        record["outcome"] = "delete_deferred"
        leftovers[name] = record
        raise
    if record["outcome"] == "late":
        logger.warning(
            "Daytona snapshot %s: the create failed (%s) but the snapshot was "
            "active %.0fs later; it is kept",
            name,
            record["cause"],
            record["look_sec"],
        )
        return {
            LATE_META: json.dumps(
                {
                    "cause": record["cause"],
                    "timeout_sec": timeout,
                    "late_sec": record["look_sec"],
                }
            )
        }
    if record["outcome"] == "delete_deferred":
        leftovers[name] = record
    logger.warning(
        "Daytona snapshot %s: the create failed (%s); %s",
        name,
        record["cause"],
        _OUTCOME_TEXT[record["outcome"]],
    )
    setattr(failure, LEFTOVER_ATTR, record)
    raise failure


_OUTCOME_TEXT = {
    "removed": "the snapshot it made is deleted",
    "none_found": "Daytona shows no snapshot of that name",
    "delete_deferred": (
        "the snapshot was still being made, or its delete didn't go; it is "
        "deleted when the sandbox stops"
    ),
}


async def settle_leftovers(
    leftovers: dict[str, dict[str, Any]], *, look_sec: float, poll_sec: float
) -> list[dict[str, Any]]:
    """At the sandbox's stop, before the sandbox is deleted (Daytona refuses that
    while it is snapshotting): wait again, for up to ``look_sec`` in all, for
    each leftover to be final, then delete it until it is gone. Returns the
    records, each with its outcome (``removed``, ``none_found``, or
    ``delete_deferred`` still, named for ``bench sandbox cleanup``); a settled
    one leaves ``leftovers``. A cancelled stop leaves the rest to the next."""
    from benchflow.sandbox.daytona_reaper import owner_snapshot_prefix

    deadline = asyncio.get_running_loop().time() + look_sec
    settled = []
    for name, record in list(leftovers.items()):
        await _look(name, record, deadline=deadline, poll_sec=poll_sec)
        if record["outcome"] == "late":
            # Active, and no checkpoint holds it: delete it now.
            gone = await _delete_until_gone(
                name, record, deadline=deadline, poll_sec=poll_sec
            )
            record["outcome"] = "removed" if gone else "delete_deferred"
        record["at"] = "stop"
        if record["outcome"] != "delete_deferred":
            leftovers.pop(name, None)
        elif owner_snapshot_prefix() is None:
            logger.warning(
                "Daytona snapshot %s could not be deleted (last state %s). With no "
                "BENCHFLOW_DAYTONA_OWNER, `bench sandbox cleanup` can't tell it is "
                "yours: delete it in Daytona",
                name,
                record.get("state"),
            )
        else:
            logger.warning(
                "Daytona snapshot %s could not be deleted (last state %s); "
                "`bench sandbox cleanup` removes it once it is stale",
                name,
                record.get("state"),
            )
        settled.append(dict(record))
    return settled


def write_leftovers(trial_dir: Path, settled: list[dict[str, Any]]) -> None:
    """Write the stop's records to :data:`LEFTOVERS_FILE` in ``trial_dir``:
    one per snapshot name, a later stop's record replacing an earlier one's,
    the rest kept. An earlier file that can't be read is moved aside, and the
    new file names where. Atomic, and best effort like ``daytona-execs.json``:
    a failed write only logs."""
    from benchflow.review.persistence import write_json_atomic

    path = trial_dir / LEFTOVERS_FILE
    document: dict[str, Any] = {
        "schema_version": 1,
        "kind": "benchflow-daytona-snapshot-leftovers",
    }
    by_name: dict[Any, dict[str, Any]] = {}
    try:
        earlier = json.loads(path.read_text())
        items = earlier["leftovers"]
        if not isinstance(items, list):
            raise TypeError("leftovers is not a list")
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError, KeyError) as exc:
        aside = path.with_name(f"{LEFTOVERS_FILE}.unreadable-{time.time_ns()}")
        try:
            path.replace(aside)
        except OSError:
            aside = path
        logger.warning(
            "%s could not be read (%s); it is now %s", path, _short(exc), aside.name
        )
        document["unreadable_earlier_file"] = aside.name
    else:
        by_name = {item.get("name"): item for item in items if isinstance(item, dict)}
        if earlier.get("unreadable_earlier_file"):
            document["unreadable_earlier_file"] = earlier["unreadable_earlier_file"]
    by_name.update((item.get("name"), item) for item in settled)
    document["leftovers"] = list(by_name.values())
    try:
        write_json_atomic(path, document)
    except OSError as exc:
        logger.warning("Could not write %s: %s", path, exc)


async def _look(
    name: str,
    record: dict[str, Any],
    *,
    deadline: float,
    poll_sec: float,
    made_nothing: bool = False,
) -> None:
    """Look up ``name`` until Daytona shows a final state or ``deadline``; set
    ``record``'s ``state``, ``outcome`` and ``look_sec``. A failed snapshot is
    deleted until it is gone.

    A snapshot never seen in three polls is ``none_found`` (Daytona lists one
    within a second of the request), at once when the create can't have made
    one. Lookups that fail never count as "not found"."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    absent_since: float | None = None
    outcome = "delete_deferred"
    seen = record.get("state")
    while True:
        state = await _state(name, record)
        now = loop.time()
        if state == _ABSENT:
            absent_since = now if absent_since is None else absent_since
            if seen is not None:
                # Seen before and now gone (deleted at an earlier stop, or by
                # someone else).
                outcome = "removed"
                break
            if made_nothing or now - absent_since >= 3 * poll_sec:
                outcome = "none_found"
                break
        elif state is not None:
            absent_since = None
            seen = state
            if state == "active":
                outcome = "late"
                break
            if state in _FINAL_FAILED:
                gone = await _delete_until_gone(
                    name, record, deadline=deadline, poll_sec=poll_sec
                )
                outcome = "removed" if gone else "delete_deferred"
                break
        if now >= deadline:
            break
        await asyncio.sleep(min(poll_sec, max(deadline - now, 0.0)))
    record.update(state=seen, outcome=outcome, look_sec=round(loop.time() - started, 1))


async def _delete_until_gone(
    name: str, record: dict[str, Any], *, deadline: float, poll_sec: float
) -> bool:
    """Delete snapshot ``name`` until Daytona no longer lists it, or ``deadline``.

    A refused delete is tried again at the next poll, and so is one that didn't
    hold (the snapshot shows a state other than ``removing`` after it). The
    tries are counted in ``record``."""
    from benchflow.sandbox.daytona_strategies import _delete_provider_snapshot

    loop = asyncio.get_running_loop()
    while True:
        record["delete_tries"] = record.get("delete_tries", 0) + 1
        try:
            await _delete_provider_snapshot(name)
        except Exception as exc:
            record["delete_error"] = _short(exc)
        while True:
            state = await _state(name, record)
            if state == _ABSENT:
                return True
            if state is not None:
                record["state"] = state
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(poll_sec)
            if state not in (None, "removing"):
                break


async def _sandbox_back(
    sandbox: Any, name: str, *, deadline: float, poll_sec: float
) -> None:
    """Wait, until ``deadline``, for ``sandbox`` to leave SNAPSHOTTING: the SDK's
    create waits for that, and the run's next command needs it."""
    loop = asyncio.get_running_loop()
    while True:
        try:
            await sandbox.refresh_data()
            state = getattr(sandbox, "state", None)
            if str(getattr(state, "value", state)).lower() != "snapshotting":
                return
        except Exception as exc:
            logger.debug("Daytona sandbox state for %s: %s", name, exc)
        if loop.time() >= deadline:
            logger.warning(
                "Daytona snapshot %s is active, but its sandbox is still "
                "snapshotting; the run goes on",
                name,
            )
            return
        await asyncio.sleep(poll_sec)


async def _state(name: str, record: dict[str, Any]) -> str | None:
    """Daytona's state for snapshot ``name`` (``active``, ``snapshotting``,
    ...), ``absent`` when Daytona answers 404, None when the lookup itself
    failed (counted in ``record`` with its error; looked up again at the next
    poll)."""
    from benchflow.sandbox import daytona as _sdk

    try:
        manager = await _sdk.DaytonaClientManager.get_instance()
        client = cast(Any, await manager.get_client())
        snapshot = await client.snapshot.get(name)
    except Exception as exc:
        if (
            isinstance(exc, _sdk.DaytonaNotFoundError)
            or getattr(exc, "status_code", None) == 404
        ):
            return _ABSENT
        record["lookups_failed"] = record.get("lookups_failed", 0) + 1
        record["lookup_error"] = _short(exc)
        return None
    state = getattr(snapshot, "state", None)
    return str(getattr(state, "value", state)).lower()


def _short(exc: BaseException) -> str:
    """``Type: first line``, at most 200 characters, for a record."""
    lines = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {lines[0] if lines else ''}"[:200]


def _made_nothing(exc: BaseException) -> bool:
    """The create can't have made a snapshot: Daytona refused it (a 4xx other
    than 408 or 429), or the error is neither Daytona's nor a timeout or
    transport error, so the request never reached Daytona."""
    if not type(exc).__module__.startswith(("daytona", "aiohttp")) and not isinstance(
        exc, (TimeoutError, OSError)
    ):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status not in (408, 429)
