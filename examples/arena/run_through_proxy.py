"""Real run THROUGH the BenchFlow LiteLLM proxy.

Three deepseek-v4 seats play one shared high-card round concurrently via
``run_arena``; each seat's raw LLM call is routed through a loopback LiteLLM proxy
started by ``ensure_litellm_runtime`` — so per-agent **usage/cost** and the proxy's
**llm trajectory** are captured by BenchFlow, and the seats never see the raw
provider key (the proxy isolation invariant). A per-seat *decision* trajectory is
also written to ``out/arena-floor-proxy/<seat>.trajectory.jsonl``.

    set -a; . ./sb-run.env; set +a          # DEEPSEEK_API_KEY (the real upstream key)
    uv run python examples/arena/run_through_proxy.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from benchflow.arena import ProxyChatPolicy, SeatTrajectory, SharedEnvReward, run_arena
from benchflow.providers import (
    ensure_litellm_runtime,
    extract_usage,
    stop_provider_runtime,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from floor_deepseek import HighCardFloor, pick_number, render_for


async def _main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY required (the real upstream key)")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seats = [f"seat-{i}" for i in range(n)]

    print("starting BenchFlow LiteLLM proxy (environment=local)…", flush=True)
    agent_env, runtime = await ensure_litellm_runtime(
        agent="deepagents",
        agent_env={"DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"]},
        model="deepseek/deepseek-v4-pro",  # provider-prefixed → routes to deepseek
        runtime=None,
        environment="local",
        session_id="arena-floor",
    )
    os.environ.update(agent_env)  # each seat now reads BENCHFLOW_PROVIDER_* → the proxy
    print("  proxy base :", agent_env.get("BENCHFLOW_PROVIDER_BASE_URL"))
    print("  model alias:", agent_env.get("BENCHFLOW_PROVIDER_MODEL"))
    print("  raw key hidden from seats:",
          "DEEPSEEK_API_KEY" not in agent_env, flush=True)

    run_dir = Path("out/arena-floor-proxy")
    tr = SeatTrajectory(run_dir)
    floor = HighCardFloor(n)
    res: dict = {}
    try:
        async with httpx.AsyncClient() as http:
            policies = {
                s: ProxyChatPolicy(s, http, render=render_for(s), pick=pick_number,
                                   temperature=0.9, max_tokens=2048, recorder=tr)
                for s in seats
            }
            res = await run_arena(seats, floor, lambda s: policies[s],
                                  deadline_s=180.0, poll_s=0.05)
    finally:
        await asyncio.sleep(1.5)  # let the proxy's async callback flush the last call
        await stop_provider_runtime(runtime)  # parses the proxy callback log
    usage = extract_usage(runtime)            # aggregate tokens/cost after stop

    # persist the proxy's raw-LLM trajectory in BenchFlow's canonical format
    # (mirrors rollout._write_llm_trajectory).
    proxy_traj = getattr(getattr(runtime, "server", None), "trajectory", None)
    if proxy_traj is not None and proxy_traj.exchanges:
        traj_dir = run_dir / "trajectory"
        traj_dir.mkdir(parents=True, exist_ok=True)
        (traj_dir / "llm_trajectory.jsonl").write_text(proxy_traj.to_jsonl(redact_keys=True))

    st = floor.standings()
    print("\npicks       :", floor.picks)
    print("standings   :", st)
    print("reward (pvp):", SharedEnvReward().score(st))
    print("seat status :", {s: r["status"] for s, r in res.items()})
    print("conserved   :", sum(st.values()), f"(== {n * 1000})")
    print("proxy usage :", json.dumps(usage))  # tokens + cost from the proxy callback log
    if proxy_traj is not None and proxy_traj.exchanges:
        print(f"llm_trajectory: {run_dir}/trajectory/llm_trajectory.jsonl "
              f"({len(proxy_traj.exchanges)} raw exchanges, canonical format)")
    print(f"decision traj : {run_dir}/<seat>.trajectory.jsonl")
    for s in seats:
        line = tr.path(s).read_text().strip().splitlines()[-1]
        rec = json.loads(line)
        print(f"  {s}: pick={rec['action']['args']['n']}  llm.usage={rec['llm']['usage']}")


if __name__ == "__main__":
    asyncio.run(_main())
