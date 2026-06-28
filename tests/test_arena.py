"""The inter-agent concurrent (arena) scaffold: N seats drive ONE shared,
turn-gated env at once. Exercised with an in-memory fake env + scripted policies
— no ACP, no sandbox, no LLM — so the concurrency, turn-gating, reaping, and
per-seat scoring are unit-testable on their own.
"""

from __future__ import annotations

import asyncio

from benchflow.arena import (
    FloorMode,
    Observation,
    SharedEnvReward,
    run_arena,
)


class _FakeFloor:
    """A tiny turn-gated 2-seat env implementing the SeatClient contract.

    Each seat, on its turn, bids a number; the higher bid wins ``stake`` from the
    lower (zero-sum). Seats lazy-join on first ``observe``; the table forms once
    ``n_seats`` are present; turns are issued one at a time with a stable
    ``request_id`` (so a stale id is rejected)."""

    def __init__(self, n_seats: int = 2, stake: int = 100, start: int = 1000) -> None:
        self.n, self.stake, self.start = n_seats, stake, start
        self.bankroll: dict[str, int] = {}
        self.seated: list[str] = []
        self.bids: dict[str, int] = {}
        self.pending: str | None = None
        self.cur_rid: str | None = None
        self.turn = 0
        self.formed = self.done = False
        self.lock = asyncio.Lock()

    def _open_turn(self, seat: str) -> None:
        self.pending = seat
        self.turn += 1
        self.cur_rid = f"{seat}#{self.turn}"

    async def observe(self, seat_id: str) -> dict:
        async with self.lock:
            self.bankroll.setdefault(seat_id, self.start)
            if not self.formed and seat_id not in self.seated:
                self.seated.append(seat_id)
            if not self.formed and len(self.seated) >= self.n:
                self.formed = True
                self._open_turn(self.seated[0])
            if self.done:
                return {"status": "done", "bankroll": self.bankroll[seat_id]}
            if not self.formed:
                return {"status": "waiting"}
            if self.pending != seat_id:
                return {"status": "not_your_turn", "current_actor": self.pending}
            return {
                "status": "your_turn", "request_id": self.cur_rid,
                "observation": {"public": {"pot": self.stake}, "private": {}},
                "legal_actions": [{"verb": "bid", "args": {"n": k}} for k in range(10)],
            }

    async def act(self, seat_id: str, request_id: str, action: dict) -> dict:
        async with self.lock:
            if not self.formed or self.pending != seat_id:
                return {"ok": False, "status": "not_your_turn"}
            if request_id != self.cur_rid:
                return {"ok": False, "status": "stale_request_id"}
            self.bids[seat_id] = int(action.get("args", {}).get("n", 0))
            idx = self.seated.index(seat_id)
            if idx + 1 < len(self.seated):
                self._open_turn(self.seated[idx + 1])
            else:
                self._settle()
            return {"ok": True, "status": "applied"}

    def _settle(self) -> None:
        a, b = self.seated[0], self.seated[1]
        if self.bids[a] != self.bids[b]:
            win, lose = (a, b) if self.bids[a] > self.bids[b] else (b, a)
            self.bankroll[win] += self.stake
            self.bankroll[lose] -= self.stake
        self.pending = self.cur_rid = None
        self.done = True

    def standings(self) -> dict[str, int]:
        return dict(self.bankroll)


class _FixedBid:
    """A scripted seat brain: always bid the same number."""

    def __init__(self, n: int) -> None:
        self.n = n

    async def act(self, obs: Observation) -> dict:
        return {"verb": "bid", "args": {"n": self.n}}


def test_two_seats_play_concurrently_and_settle_zero_sum() -> None:
    env = _FakeFloor()
    bids = {"seat-0": 7, "seat-1": 3}                 # seat-0 wins
    res = asyncio.run(run_arena(
        ["seat-0", "seat-1"], env, lambda s: _FixedBid(bids[s]),
        deadline_s=5.0, poll_s=0.001,
    ))
    assert all(r["status"] == "done" for r in res.values())
    st = env.standings()
    assert st == {"seat-0": 1100, "seat-1": 900}
    assert sum(st.values()) == 2000                   # zero-sum, conserved


def test_per_seat_reward_vector() -> None:
    env = _FakeFloor()
    bids = {"seat-0": 7, "seat-1": 3}
    asyncio.run(run_arena(["seat-0", "seat-1"], env, lambda s: _FixedBid(bids[s]),
                          deadline_s=5.0, poll_s=0.001))
    pvp = SharedEnvReward(mode=FloorMode.PVP).score(env.standings())
    assert pvp == {"seat-0": 100.0, "seat-1": -100.0}
    assert sum(pvp.values()) == 0.0
    coop = SharedEnvReward(mode=FloorMode.COOP).score(env.standings())
    assert coop == {"seat-0": -100.0, "seat-1": -100.0}   # joint = worst seat


def test_not_your_turn_is_rejected() -> None:
    async def scenario() -> None:
        env = _FakeFloor()
        await env.observe("seat-0")                   # seat-0 seats, pending
        await env.observe("seat-1")                   # forms; seat-0 is pending
        o0 = await env.observe("seat-0")
        assert o0["status"] == "your_turn"
        bad = await env.act(
            "seat-1", o0["request_id"], {"verb": "bid", "args": {"n": 9}})
        assert bad == {"ok": False, "status": "not_your_turn"}  # out of turn
        stale = await env.act("seat-0", "bogus", {"verb": "bid", "args": {"n": 9}})
        assert stale["status"] == "stale_request_id"  # wrong request_id

    asyncio.run(scenario())


def test_lone_seat_is_reaped_by_deadline() -> None:
    env = _FakeFloor(n_seats=2)                        # needs a partner
    res = asyncio.run(run_arena(["solo"], env, lambda s: _FixedBid(5),
                                deadline_s=0.2, poll_s=0.01))
    assert res["solo"]["status"] == "deadline"         # never formed, reaped
    assert env.standings() == {"solo": 1000}           # no chips moved
