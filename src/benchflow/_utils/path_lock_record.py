"""The trial's record of the folders on PATH locked to root before the agent ran.

:func:`benchflow.sandbox.lockdown.setup_sandbox_user` locks every system
folder on the sandbox's ``PATH`` (the image's folders, then the standard
system folders appended after them), and the programs directly in them, that
someone other than root could write or that another user owns, before the
agent starts. The trial wraps that call in :func:`recording_into`
and writes the dict it handed in to ``locked-system-path.json`` in its folder:
``state`` is ``enforced`` when every such folder and program is root's alone (one
the image left open and BenchFlow locked is listed under ``locked``, with a
note starting ``repaired:``), or ``failed`` when a ``chown`` or ``chmod`` failed
or a program on PATH resolves into the workspace, a home or a temp folder
(listed under ``unlockable``). Outside such a block nothing is recorded.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar
from typing import Any

#: The file in the trial folder.
LOCKED_SYSTEM_PATH_RECORD = "locked-system-path.json"

_current: ContextVar[dict[str, Any] | None] = ContextVar(
    "benchflow_locked_system_path", default=None
)


@contextlib.contextmanager
def recording_into(record: dict[str, Any]) -> Iterator[None]:
    """Let the PATH lock in code below write into ``record``."""
    token = _current.set(record)
    try:
        yield
    finally:
        _current.reset(token)


def path_lock_record(locked: list[str], unlockable: list[str]) -> dict[str, Any]:
    """The record for one lock: its state, the folders, and a one-line note."""
    if unlockable:
        state = "failed"
        note = (
            f"the image lets others write {', '.join(unlockable)}, and it could "
            "not be locked to root"
        )
    elif locked:
        state = "enforced"
        note = (
            f"repaired: the image lets others write {', '.join(locked)}; locked "
            "to root before the agent ran"
        )
    else:
        state = "enforced"
        note = "the folders on PATH and the programs in them are root's alone"
    return {"state": state, "locked": locked, "unlockable": unlockable, "note": note}


def record_path_lock(locked: list[str], unlockable: list[str]) -> None:
    """Note what the lock did, when the trial asked for a record."""
    record = _current.get()
    if record is not None:
        record.clear()
        record.update(path_lock_record(locked, unlockable))
