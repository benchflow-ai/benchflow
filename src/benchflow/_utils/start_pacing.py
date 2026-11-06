"""Pace a job's trial starts with a token bucket, outside the concurrency slot.

Above ``--concurrency 16`` a job used to sleep each trial a random
``uniform(0, max(c / 2, 8))`` seconds after it took its slot. That spread the
starts to about 2 per second, which keeps compose-up and SSH storms off the
Docker daemon, but every trial idled its slot for c/4 seconds on average,
on the first wave and on every later one.

:class:`StartPacer` keeps the rate and frees the slot. Trials take their start
turn one at a time, in the order they asked; the trial whose turn it is waits
for a token with no slot held, then waits for a free slot, and spends the
token only when it has the slot and budget admission succeeds. The start
turn stays held across admission. A bucket of one token cannot bank starts
while every slot is busy, so slots that free together still start their
trials at most ``rate`` per second.

A trial that will not start (the job's budget, usage limit or API-error
breaker has stopped it) skips the token wait: a stopped job empties its queue
at once instead of at the start rate.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from benchflow._utils import timing_spans

#: Above this ``--concurrency`` the job paces its starts; at or below it,
#: trials start as soon as they have a slot (as before the pacer).
PACE_ABOVE_CONCURRENCY = 16

#: The start rate the old jitter averaged: its window, c/2 seconds for c
#: trials, gave 2 starts per second at every concurrency above 16.
STARTS_PER_SECOND = 2.0


class StartPacer:
    """At most ``rate`` trial starts per second, with no slot held while waiting.

    One pacer per schedule: :meth:`take_slot` hands out the schedule's
    semaphore slots, and the caller releases the slot as before.
    """

    def __init__(self, rate: float = STARTS_PER_SECOND) -> None:
        if rate <= 0:
            raise ValueError(f"start rate must be positive, got {rate}")
        self._rate = rate
        # One token at most: the first trial starts at once, and no burst is
        # saved up while the slots are busy.
        self._tokens = 1.0
        self._stamp: float | None = None
        self._turn = asyncio.Lock()

    def _refill(self) -> None:
        now = timing_spans.clock()
        if self._stamp is not None:
            self._tokens = min(1.0, self._tokens + (now - self._stamp) * self._rate)
        self._stamp = now

    async def take_slot(
        self,
        slot: asyncio.Semaphore,
        will_not_start: Callable[[], bool],
        *,
        admit: Callable[[], Awaitable[bool]] | None = None,
    ) -> float:
        """Acquire ``slot`` at the pacer's rate; the seconds slept for a token.

        ``will_not_start`` is asked before and after each token wait: once it
        is true the trial takes its slot without a token (it only records
        that it was skipped, which needs no pacing). The caller re-checks its
        stop conditions in ``admit`` (or after this returns, for callers
        without admission). Admission runs with the slot and FIFO turn held;
        debit only on success, immediately before returning to the trial.
        If it raises or is cancelled, release the slot as well as the turn.
        """
        async with self._turn:
            waited = 0.0
            has_token = False
            while not will_not_start():
                self._refill()
                if self._tokens >= 1.0:
                    has_token = True
                    break
                asleep = timing_spans.clock()
                await asyncio.sleep((1.0 - self._tokens) / self._rate)
                waited += timing_spans.clock() - asleep
            # No slot is held above. Waiting for one here keeps the turn, so
            # the trials behind cannot take the token this one is owed.
            await slot.acquire()
            try:
                admitted = admit is None or await admit()
                if has_token and admitted:
                    self._refill()
                    self._tokens -= 1.0
            except BaseException:
                slot.release()
                raise
            return waited
