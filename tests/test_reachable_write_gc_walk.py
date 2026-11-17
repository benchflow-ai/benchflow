"""destination() walks the pinned roots while a dropped stage may be closed.

A coroutine suspended inside ``with private_folder(...)`` and then dropped
(a task left pending when its loop closed) lives on in a reference cycle.
The cyclic GC frees it by calling its ``close()``, on whichever thread the
GC ran and between any two bytecodes; private_folder's exit then pops that
stage's root from ``_DESTINATION_ROOTS``. destination() walked that table
with ``items()``, so a close inside the walk raised "dictionary changed size
during iteration". Under ``-n 10`` it surfaced as an artifact collection
error in test_verifier_own_failures (the private_folder of _capture calls
destination()).

The test makes the GC's call itself: it closes the dropped coroutine at
every line the walk executes, in destination() and in weakref.py, one
point per call. No global gc.collect(): that would also run every other
test's leftover finalizers.
"""

from __future__ import annotations

import asyncio
import sys
import weakref
from pathlib import Path

import pytest

from benchflow._utils import reachable_write as rw


def _held_stage(folder: Path):
    """A coroutine suspended inside private_folder, and its stage's path."""
    held: list[Path] = []

    async def holds(gate: asyncio.Future) -> None:
        with rw.private_folder(folder) as stage:
            held.append(stage)
            await gate

    loop = asyncio.new_event_loop()
    try:
        coro = holds(loop.create_future())
        coro.send(None)
    finally:
        loop.close()
    return coro, held[0]


class _CloseAtLine:
    """A tracer that closes ``coro`` (as the GC would) at the ``point``-th traced line."""

    def __init__(self, point: int, coro) -> None:
        self.point = point
        self.coro = coro
        self.lines = 0
        self.fired = False

    def __call__(self, frame, event, arg):
        code = frame.f_code
        if code.co_filename == weakref.__file__ or (
            code.co_filename == rw.__file__ and code.co_name == "destination"
        ):
            return self._line
        return None

    def _line(self, frame, event, arg):
        if event == "line" and not self.fired:
            if self.lines == self.point:
                self.fired = True
                self.coro.close()
            self.lines += 1
        return self._line


@pytest.mark.skipif(sys.gettrace() is not None, reason="needs the trace hook")
def test_a_close_anywhere_in_the_walk_leaves_destination_whole(tmp_path):
    root = tmp_path / "root"
    (root / "out").mkdir(parents=True)
    pinned = rw.pin_destination(root)
    target = root / "out" / "file.txt"
    points = 0
    for point in range(500):
        coro, stage = _held_stage(root / "out")
        assert stage in rw._DESTINATION_ROOTS
        tracer = _CloseAtLine(point, coro)
        sys.settrace(tracer)
        try:
            chosen = rw.destination(target)
        finally:
            sys.settrace(None)
            coro.close()
        assert chosen._anchor is pinned._anchor
        assert chosen._parts == ("out", "file.txt")
        # The closed stage's root left the table.
        assert stage not in rw._DESTINATION_ROOTS
        if not tracer.fired:
            break
        points += 1
    assert points >= 5, points
