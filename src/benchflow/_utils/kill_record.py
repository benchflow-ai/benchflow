"""The trial's record of the kill before grading (K1).

:func:`benchflow.sandbox.lockdown.harden_before_verify` stops the sandbox
user's processes before the verifier runs (freeze rounds, then a kill). The
scoring path wraps it in :func:`recording_into`, and the kill writes who was
stopped and how many processes it killed into the dict it was handed; the
trial writes that dict to ``kill-before-grading.json`` in its folder.

The turn-end quiesce runs before artifact collection, even without optional
workspace capture (P78). That kill is recorded into the same dict, and
counts add up in ``killed_total``, so a later empty kill cannot hide earlier
work. ``killed`` is the latest kill's own count (K2), and ``quiesce_killed``
is the turn-end share (K1). Outside such a block (a soft verify
between rounds) nothing is recorded.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from contextvars import ContextVar
from pathlib import Path
from typing import Any

#: The file in the trial folder.
KILL_BEFORE_GRADING_RECORD = "kill-before-grading.json"

_current: ContextVar[dict[str, Any] | None] = ContextVar(
    "benchflow_kill_before_grading", default=None
)


@contextlib.contextmanager
def recording_into(record: dict[str, Any]) -> Iterator[None]:
    """Let the kill before grading in code below write into ``record``."""
    token = _current.set(record)
    try:
        yield
    finally:
        _current.reset(token)


def record_kill_before_grading(sandbox_user: str, killed: int) -> None:
    """Add this kill's count to the record of the block it runs in.

    Counts add up, so a later kill that finds nothing never hides an earlier
    one that killed something.
    """
    record = _current.get()
    if record is not None:
        record["user"] = sandbox_user
        record["killed_total"] = int(record.get("killed_total", 0)) + killed
        record["killed"] = killed


def write_kill_record(rollout_dir: str | Path, record: dict[str, Any]) -> None:
    """Write ``record`` to ``kill-before-grading.json`` in the trial folder."""
    path = Path(rollout_dir) / KILL_BEFORE_GRADING_RECORD
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def record_kept_services(
    rows: list[dict[str, Any]],
    *,
    ports: list[int],
    source: str,
    no_listener: list[int],
    waited_sec: float,
    not_nnp: list[dict[str, Any]] | None = None,
) -> None:
    """Note the services the kill kept (``x-keep-services``, K2).

    ``kept_services`` has one row per process kept: its port, pid, command
    line (at most 200 characters) and where the names came from;
    ``kept_services_ports`` the ports the trial keeps, ``kept_services_source``
    where they came from, ``kept_services_no_listener`` the named ports no
    process of the agent user listened on, and ``kept_services_waited_sec``
    how long the kill waited for listeners first. ``kept_services_not_nnp``
    lists the listeners on kept ports it killed instead because they ran
    without no_new_privs: not the agent's launch, so a set-ID file
    or a file capability put back before grading would work for them.
    """
    record = _current.get()
    if record is not None:
        record.update(
            {
                "kept_services": rows,
                "kept_services_ports": ports,
                "kept_services_source": source,
                "kept_services_no_listener": no_listener,
                "kept_services_waited_sec": waited_sec,
                **({"kept_services_not_nnp": not_nnp} if not_nnp else {}),
            }
        )


def record_kept_services_turn_end(turn_end: dict[str, Any], *, lost: list[int]) -> None:
    """Note what the trial's turn-end kill did before this one.

    ``kept_services_turn_end`` holds how many it killed, how long it waited
    for listeners and the pids it kept (port and pid); ``killed`` and
    ``kept_services_waited_sec`` stay this kill's own. A pid the turn-end
    kill kept that this one does not is listed under
    ``kept_services_lost_before_grading``.
    """
    record = _current.get()
    if record is not None:
        record["kept_services_turn_end"] = turn_end
        if lost:
            record["kept_services_lost_before_grading"] = lost
