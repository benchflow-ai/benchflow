"""Sandbox teardown off a trial's critical path (PERF-B).

Removing a trial's containers takes the daemon about a second (``docker rm
-f`` alone is 0.6 s on the engine bench's build VM) and the whole stop 1-3 s,
and the trial's result waited for it. Inside a job, a trial now hands its
sandbox's stop (and the removal of its task copy, which the stop's Compose
files live in) to the job's queue and goes on to write its result; the next
trial can start meanwhile.

- **Bounded.** At most *limit* stops run at once (the job's concurrency, at
  least 2); a trial that would start one more waits for the oldest.
- **Drained.** The job waits for every queued stop before it sweeps Docker
  and writes its summary, and on any way out: an error, or Ctrl-C, which
  cannot cut the drain short (a second Ctrl-C is held until it ends; each
  stop has its own deadlines, so the drain does not hang on a wedged daemon).
- **Counted.** Under a budget, a stop's seconds are added to its trial's
  sandbox seconds when it ends, before the job's drain ends.
- **Docker only.** Only Docker and remote Docker sandboxes are queued
  (``background_stop``); a provider's sandbox stops inline.
- **Same work.** The stop is the one the trial ran itself, in the same order,
  and a failure is logged as it was ("Cleanup failed"). Outside a job (an SDK
  caller driving a ``Rollout``) nothing is queued and the stop runs inline.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextvars import ContextVar

logger = logging.getLogger(__name__)

_QUEUE: ContextVar[TeardownQueue | None] = ContextVar(
    "benchflow_teardown_queue", default=None
)
# Set by the job for each trial under a budget: called with a queued stop's
# seconds when it ends, so the trial's sandbox time includes its stop.
_STOP_SECONDS: ContextVar[Callable[[float], None] | None] = ContextVar(
    "benchflow_teardown_stop_seconds", default=None
)


@contextlib.contextmanager
def counting_stop_seconds(add: Callable[[float], None]):
    """Inside, a queued stop's seconds go to *add* when it ends."""
    token = _STOP_SECONDS.set(add)
    try:
        yield
    finally:
        _STOP_SECONDS.reset(token)


class TeardownQueue:
    """The job's queued sandbox stops."""

    def __init__(self, limit: int) -> None:
        self._limit = max(1, limit)
        self._running: set[asyncio.Task[None]] = set()

    async def submit(
        self, work: Callable[[], Awaitable[None]], *, what: str
    ) -> asyncio.Task[None]:
        """Start *work* in the background once fewer than *limit* run; the
        task, for a caller that must wait for this stop (a warm-pool trial
        the job discards removes its folder only after it, PERF-ALL §2)."""
        while len(self._running) >= self._limit:
            await asyncio.wait(self._running, return_when=asyncio.FIRST_COMPLETED)
        task = asyncio.get_running_loop().create_task(
            self._run(work, what, _STOP_SECONDS.get())
        )
        self._running.add(task)
        task.add_done_callback(self._running.discard)
        return task

    @staticmethod
    async def _run(
        work: Callable[[], Awaitable[None]],
        what: str,
        count: Callable[[float], None] | None,
    ) -> None:
        started = time.monotonic()
        try:
            await work()
        except Exception as exc:
            logger.warning(f"Cleanup of {what} failed: {exc}")
        else:
            logger.debug(
                "Background teardown of %s took %.2fs", what, time.monotonic() - started
            )
        finally:
            if count is not None:
                try:
                    count(time.monotonic() - started)
                except Exception as exc:
                    logger.warning(f"Counting the stop of {what} failed: {exc}")

    async def drain(self) -> None:
        """Wait for every queued stop; a cancellation (Ctrl-C) is held until
        they end, then raised."""
        cancelled = False
        while self._running:
            try:
                await asyncio.shield(asyncio.wait(set(self._running)))
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError


def teardown_queue() -> TeardownQueue | None:
    """The running job's queue, or None outside a job."""
    return _QUEUE.get()


@contextlib.asynccontextmanager
async def background_teardown(limit: int) -> AsyncIterator[TeardownQueue]:
    """Queue the sandbox stops of the trials run inside; drain them on exit."""
    queue = TeardownQueue(max(2, limit))
    token = _QUEUE.set(queue)
    try:
        yield queue
    finally:
        _QUEUE.reset(token)
        await queue.drain()
