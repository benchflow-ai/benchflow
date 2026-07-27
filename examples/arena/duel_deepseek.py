"""Real run of the arena scaffold with two deepseek-v4 seats.

Two independent agents play one round of rock-paper-scissors **concurrently**
against a single shared, turn-gated environment — driven entirely by
``benchflow.arena.run_arena``. The environment here is an in-process object that
implements the same ``SeatClient`` turn-poll contract a networked co-tenant
service would (so the demo needs no FastAPI/sandbox); each seat's brain is a real
``deepseek-v4-pro`` call.

    set -a; . ./sb-run.env; set +a        # DEEPSEEK_API_KEY (and optional _BASE_URL)
    uv run python examples/arena/duel_deepseek.py

This is the inter-agent (concurrent) axis: N agents, one shared world, per-seat
reward — none of which the framework hosts natively today. The scaffold is
opt-in and touches no existing scored path.
"""

from __future__ import annotations

import asyncio
import os
import random

import httpx

from benchflow.arena import Observation, SharedEnvReward, run_arena

RPS_BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}


class DuelFloor:
    """In-process shared env implementing the SeatClient contract: a turn-gated,
    2-seat rock-paper-scissors round. Seats lazy-join on first ``observe``;
    throws stay private until both have acted, then it settles zero-sum."""

    def __init__(self, stake: int = 100, start: int = 1000) -> None:
        self.stake, self.start = stake, start
        self.bankroll: dict[str, int] = {}
        self.seated: list[str] = []
        self.throws: dict[str, str] = {}
        self.pending: str | None = None
        self.cur_rid: str | None = None
        self.turn = 0
        self.formed = self.done = False
        self.lock = asyncio.Lock()

    def _open(self, seat: str) -> None:
        self.pending, self.turn = seat, self.turn + 1
        self.cur_rid = f"{seat}#{self.turn}"

    async def observe(self, seat_id: str) -> dict:
        async with self.lock:
            self.bankroll.setdefault(seat_id, self.start)
            if not self.formed and seat_id not in self.seated:
                self.seated.append(seat_id)
            if not self.formed and len(self.seated) >= 2:
                self.formed = True
                self._open(self.seated[0])
            if self.done:
                return {"status": "done", "bankroll": self.bankroll[seat_id]}
            if not self.formed:
                return {"status": "waiting"}
            if self.pending != seat_id:
                return {"status": "not_your_turn", "current_actor": self.pending}
            return {
                "status": "your_turn", "request_id": self.cur_rid,
                "observation": {
                    "public": {"pot": self.stake, "game": "rock-paper-scissors"},
                    "private": {},
                },
                "legal_actions": [
                    {"verb": "throw", "args": {"hand": h}}
                    for h in ("rock", "paper", "scissors")
                ],
            }

    async def act(self, seat_id: str, request_id: str, action: dict) -> dict:
        async with self.lock:
            if not self.formed or self.pending != seat_id:
                return {"ok": False, "status": "not_your_turn"}
            if request_id != self.cur_rid:
                return {"ok": False, "status": "stale_request_id"}
            self.throws[seat_id] = str(action.get("args", {}).get("hand", "rock"))
            idx = self.seated.index(seat_id)
            if idx + 1 < len(self.seated):
                self._open(self.seated[idx + 1])
            else:
                self._settle()
            return {"ok": True, "status": "applied"}

    def _settle(self) -> None:
        a, b = self.seated
        ta, tb = self.throws[a], self.throws[b]
        if ta != tb:
            win, lose = (a, b) if RPS_BEATS.get(ta) == tb else (b, a)
            self.bankroll[win] += self.stake
            self.bankroll[lose] -= self.stake
        self.pending = self.cur_rid = None
        self.done = True

    def standings(self) -> dict[str, int]:
        return dict(self.bankroll)


class DeepSeekPolicy:
    """A seat brain backed by a real deepseek-v4-pro call."""

    def __init__(self, seat: str, http: httpx.AsyncClient, model: str = "deepseek-v4-pro") -> None:
        self.seat, self.http, self.model = seat, http, model
        self.base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.key = os.environ["DEEPSEEK_API_KEY"]
        self.choice: str | None = None

    async def act(self, obs: Observation) -> dict:
        hands = [a["args"]["hand"] for a in obs.legal_actions]
        prompt = (
            f"You are {self.seat} in ONE round of rock-paper-scissors for a pot of "
            f"{obs.public.get('pot')} chips. Choose exactly one of: {', '.join(hands)}. "
            "Reply with ONLY that single word."
        )
        try:
            r = await self.http.post(
                f"{self.base}/chat/completions",
                headers={"Authorization": f"Bearer {self.key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256, "temperature": 0.8,
                },
                timeout=90.0,
            )
            text = (r.json()["choices"][0]["message"]["content"] or "").lower()
            hand = next((h for h in hands if h in text), random.choice(hands))
        except Exception as exc:  # a flaky call falls back to a random legal throw
            print(f"  [{self.seat}] deepseek call failed ({exc!r}); random fallback")
            hand = random.choice(hands)
        self.choice = hand
        return {"verb": "throw", "args": {"hand": hand}}


async def _main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY required (source your env file first)")
    floor = DuelFloor()
    seats = ["seat-0", "seat-1"]
    async with httpx.AsyncClient() as http:
        policies = {s: DeepSeekPolicy(s, http) for s in seats}
        print("running arena: 2 deepseek-v4-pro seats, one shared RPS table…", flush=True)
        res = await run_arena(
            seats, floor, lambda s: policies[s], deadline_s=120.0, poll_s=0.05,
        )
    st = floor.standings()
    print("\npicks      :", {s: p.choice for s, p in policies.items()})
    print("standings  :", st)
    print("reward (pvp):", SharedEnvReward().score(st))
    print("seat status :", {s: r["status"] for s, r in res.items()})
    print("conserved   :", sum(st.values()), "(== 2000)")


if __name__ == "__main__":
    asyncio.run(_main())
