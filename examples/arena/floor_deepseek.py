"""Real arena run: N deepseek-v4 seats, routed through BenchFlow's provider proxy,
with a per-seat trajectory written for each.

Each seat's raw LLM call goes through ``BENCHFLOW_PROVIDER_BASE_URL`` /
``BENCHFLOW_PROVIDER_API_KEY`` / ``BENCHFLOW_PROVIDER_MODEL`` (the proxy the SDK
injects in a real eval — there it also writes ``llm_trajectory.jsonl`` and
attributes usage per seat via the ``x-bf-seat`` tag). The bench's own per-seat
decision trajectory is written to ``out/arena-floor/<seat>.trajectory.jsonl``.

    # point the provider vars at the proxy (or, standalone, at the model API):
    export BENCHFLOW_PROVIDER_BASE_URL=https://api.deepseek.com
    export BENCHFLOW_PROVIDER_API_KEY=$DEEPSEEK_API_KEY
    export BENCHFLOW_PROVIDER_MODEL=deepseek-v4-pro
    uv run python examples/arena/floor_deepseek.py 3
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import httpx

from benchflow.arena import (
    ProxyChatPolicy,
    SeatTrajectory,
    SharedEnvReward,
    provider_config,
    run_arena,
)


class HighCardFloor:
    """N-seat turn-gated env (SeatClient): each seat antes ``stake`` and picks
    0-9; the highest pick wins the pot (ties split). Seats lazy-join on observe."""

    def __init__(self, n_seats: int, stake: int = 50, start: int = 1000) -> None:
        self.n, self.stake, self.start = n_seats, stake, start
        self.bankroll: dict[str, int] = {}
        self.seated: list[str] = []
        self.picks: dict[str, int] = {}
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
            if not self.formed and len(self.seated) >= self.n:
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
                "observation": {"public": {"pot": self.stake * self.n}, "private": {}},
                "legal_actions": [{"verb": "pick", "args": {"n": k}} for k in range(10)],
            }

    async def act(self, seat_id: str, request_id: str, action: dict) -> dict:
        async with self.lock:
            if not self.formed or self.pending != seat_id:
                return {"ok": False, "status": "not_your_turn"}
            if request_id != self.cur_rid:
                return {"ok": False, "status": "stale_request_id"}
            self.picks[seat_id] = int(action.get("args", {}).get("n", 0))
            idx = self.seated.index(seat_id)
            if idx + 1 < len(self.seated):
                self._open(self.seated[idx + 1])
            else:
                self._settle()
            return {"ok": True, "status": "applied"}

    def _settle(self) -> None:
        hi = max(self.picks.values())
        winners = [s for s, v in self.picks.items() if v == hi]
        share = (self.stake * self.n) // len(winners)
        for s in self.seated:
            self.bankroll[s] -= self.stake
        for w in winners:
            self.bankroll[w] += share
        self.pending = self.cur_rid = None
        self.done = True

    def standings(self) -> dict[str, int]:
        return dict(self.bankroll)


def render_for(seat: str):
    def render(obs) -> str:
        ns = [a["args"]["n"] for a in obs.legal_actions]
        return (
            f"You are {seat} in a one-round high-card game for a pot of "
            f"{obs.public.get('pot')} chips. Pick ONE number from {ns[0]}..{ns[-1]}; "
            "the single highest pick wins the whole pot (ties split it). "
            "Reply with ONLY your number."
        )
    return render


def pick_number(text: str, legal: list[dict]) -> dict:
    m = re.search(r"\d", text or "")
    n = int(m.group()) if m else None
    for a in legal:
        if a["args"]["n"] == n:
            return a
    import random
    return random.choice(legal)


async def _main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    base, key, model = provider_config()
    if not key:
        raise SystemExit("no provider key (BENCHFLOW_PROVIDER_API_KEY / DEEPSEEK_API_KEY)")
    run_dir = Path("out/arena-floor")
    tr = SeatTrajectory(run_dir)
    floor = HighCardFloor(n)
    seats = [f"seat-{i}" for i in range(n)]
    print(f"arena: {n} seats · provider {base} · model {model}", flush=True)
    async with httpx.AsyncClient() as http:
        policies = {
            s: ProxyChatPolicy(s, http, render=render_for(s), pick=pick_number,
                               temperature=0.9, recorder=tr)
            for s in seats
        }
        res = await run_arena(seats, floor, lambda s: policies[s],
                              deadline_s=120.0, poll_s=0.05)
    st = floor.standings()
    print("picks       :", floor.picks)
    print("standings   :", st)
    print("reward (pvp):", SharedEnvReward().score(st))
    print("seat status :", {s: r["status"] for s, r in res.items()})
    print("conserved   :", sum(st.values()), f"(== {n * 1000})")
    print(f"trajectories: {run_dir}/<seat>.trajectory.jsonl")
    for s in seats:
        rec = json.loads(tr.path(s).read_text().strip().splitlines()[-1])
        print(f"  {s}: pick={rec['action']['args']['n']}  llm.usage={rec['llm']['usage']}")


if __name__ == "__main__":
    asyncio.run(_main())
