"""Separate trajectories per agent — ONE LiteLLM proxy PER seat.

The shared-proxy run writes a single `llm_trajectory.jsonl` with every seat's
calls mixed in (and the callback records only model+messages, so a per-call agent
tag does NOT survive). To track each concurrent agent separately, give each its
OWN `ensure_litellm_runtime` → its own callback log → its own
`<seat>/trajectory/llm_trajectory.jsonl` + its own usage/cost. This is exactly how
a BenchFlow multi-role rollout isolates roles.

    set -a; . ./sb-run.env; set +a
    uv run python examples/arena/run_per_seat_proxy.py 3
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import httpx

from benchflow.arena import ProxyChatPolicy, SeatTrajectory, run_arena
from benchflow.providers import (
    ensure_litellm_runtime,
    extract_usage,
    stop_provider_runtime,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from floor_deepseek import HighCardFloor, pick_number, render_for


async def _main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY required")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seats = [f"seat-{i}" for i in range(n)]
    run_dir = Path("out/arena-per-seat")

    # 1) ONE proxy per seat — separate callback log → separate trajectory + usage
    runtimes: dict = {}
    envs: dict = {}
    print(f"starting {n} per-seat LiteLLM proxies…", flush=True)
    for s in seats:
        env, rt = await ensure_litellm_runtime(
            agent="deepagents",
            agent_env={"DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"]},
            model="deepseek/deepseek-v4-pro", runtime=None, environment="local",
            session_id=f"arena-{s}",
        )
        runtimes[s], envs[s] = rt, env
        print(f"  {s}: {env['BENCHFLOW_PROVIDER_BASE_URL']}", flush=True)

    floor = HighCardFloor(n)
    tr = SeatTrajectory(run_dir)
    usages: dict = {}
    try:
        async with httpx.AsyncClient() as http:
            policies = {
                s: ProxyChatPolicy(
                    s, http, render=render_for(s), pick=pick_number,
                    base=envs[s]["BENCHFLOW_PROVIDER_BASE_URL"],      # this seat's proxy
                    api_key=envs[s]["BENCHFLOW_PROVIDER_API_KEY"],
                    model=envs[s]["BENCHFLOW_PROVIDER_MODEL"],
                    temperature=0.9, max_tokens=256, recorder=tr,
                )
                for s in seats
            }
            res = await run_arena(seats, floor, lambda s: policies[s],
                                  deadline_s=180.0, poll_s=0.05)
        await asyncio.sleep(1.5)  # let each proxy's async callback flush
    finally:
        for s in seats:
            await stop_provider_runtime(runtimes[s])  # stop FIRST → parses callback log
    for s in seats:  # then read each proxy's usage + trajectory (populated on stop)
        usages[s] = extract_usage(runtimes[s])
        traj = getattr(getattr(runtimes[s], "server", None), "trajectory", None)
        if traj is not None and traj.exchanges:
            d = run_dir / s / "trajectory"
            d.mkdir(parents=True, exist_ok=True)
            (d / "llm_trajectory.jsonl").write_text(traj.to_jsonl(redact_keys=True))

    st = floor.standings()
    print("\nstandings :", st, "· conserved", sum(st.values()),
          "· status", {s: r["status"] for s, r in res.items()})
    print("=== SEPARATE per-agent trajectories + usage ===")
    for s in seats:
        p = run_dir / s / "trajectory" / "llm_trajectory.jsonl"
        ex = len(p.read_text().splitlines()) if p.exists() else 0
        u = usages.get(s, {})
        print(f"  {s}: {p}  ({ex} exchange)  "
              f"tokens={u.get('total_tokens')} cost=${u.get('cost_usd')}")


if __name__ == "__main__":
    asyncio.run(_main())
