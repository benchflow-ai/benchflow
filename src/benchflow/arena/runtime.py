"""Concurrent seat driver — Seam 1 + the eval glue.

:func:`run_arena` runs N seats CONCURRENTLY against one shared env (via
``asyncio.gather``); each seat is an independent observe/act pull loop. The
shared env serializes turns, so N clocks still produce one well-ordered hand. A
worker cap bounds concurrent *decisions* (the expensive policy/LLM call) without
blocking seats that are merely waiting — so interdependent seats never deadlock.
A wall-clock deadline reaps stragglers (a seat with no partner, or a stalled
one). Pure ``asyncio``; the sequential scene path is left untouched.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable

from benchflow.arena.protocol import Observation, SeatClient, SeatPolicy, SeatStatus

__all__ = ["run_arena"]


OnTurn = Callable[[str, int, Observation, dict[str, object]], None]


async def _run_seat(
    seat_id: str,
    client: SeatClient,
    policy: SeatPolicy,
    *,
    sem: asyncio.Semaphore,
    poll_s: float,
    deadline: float,
    on_turn: OnTurn | None,
) -> dict[str, object]:
    acts = 0
    while True:
        if time.monotonic() > deadline:
            return {"seat": seat_id, "status": "deadline", "acts": acts}
        obs = Observation.from_payload(await client.observe(seat_id))
        if obs.done:
            return {"seat": seat_id, "status": "done", "acts": acts}
        if obs.status is SeatStatus.YOUR_TURN:
            async with sem:  # cap concurrent decisions only — never the polling
                action = await policy.act(obs)
            await client.act(seat_id, obs.request_id or "", action)
            acts += 1
            if on_turn is not None:  # the bench's per-seat decision trajectory
                on_turn(seat_id, acts, obs, action)
        else:  # waiting / not_your_turn — yield and poll again
            await asyncio.sleep(poll_s)


async def run_arena(
    seat_ids: Iterable[str],
    client: SeatClient,
    policy_for: Callable[[str], SeatPolicy],
    *,
    workers: int = 16,
    deadline_s: float = 120.0,
    poll_s: float = 0.05,
    on_turn: OnTurn | None = None,
) -> dict[str, dict[str, object]]:
    """Drive ``seat_ids`` concurrently against ``client`` until each is done or
    the deadline fires. ``policy_for(seat_id)`` supplies that seat's brain;
    ``on_turn(seat, turn, obs, action)`` (optional) is called after each move —
    the hook the bench uses to capture per-seat trajectories. Returns
    ``{seat_id: {status, acts, ...}}`` (status ∈ done | deadline | error).
    """
    sem = asyncio.Semaphore(max(1, workers))
    deadline = time.monotonic() + deadline_s

    async def _guarded(seat_id: str) -> dict[str, object]:
        try:
            return await _run_seat(
                seat_id,
                client,
                policy_for(seat_id),
                sem=sem,
                poll_s=poll_s,
                deadline=deadline,
                on_turn=on_turn,
            )
        except Exception as exc:  # one bad seat must not kill the floor
            return {"seat": seat_id, "status": "error", "error": repr(exc), "acts": 0}

    results = await asyncio.gather(*[_guarded(s) for s in seat_ids])
    return {str(r["seat"]): r for r in results}
