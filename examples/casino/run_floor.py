"""Real multi-agent casino floor on BenchFlow — heterogeneous, all games.

N autonomous ACP agents play ONE shared casinobench World concurrently, each in
its OWN DockerSandbox (the casino-agent-seat image: node ACP agents + the `casino`
seven-tool CLI). The default roster is 4 subscription seats:
  - 2x codex-acp on gpt-5.5            (ChatGPT subscription)
  - 2x claude-agent-acp on sonnet-4-6  (Claude subscription)

Each subscription agent calls its provider directly (oauth) → it produces an
`acp_trajectory.jsonl` (its `casino` tool-calls/thinking); there is NO raw
`llm_trajectory` for subscription seats (that needs an API key fronted by the
proxy — only proxy-routed deepseek seats get one). The World runs on the host;
agents reach it over the docker bridge gateway (the same path the proxy uses).

    set -a; . ~/sb-run.env; set +a
    uv run python examples/casino/run_floor.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
from pathlib import Path

import httpx

from benchflow.acp.runtime import AgentPromptTimeoutError, connect_acp, execute_prompts
from benchflow.agents.credentials import upload_subscription_auth
from benchflow.agents.registry import AGENTS
from benchflow.providers import (
    ensure_litellm_runtime,
    extract_usage,
    stop_provider_runtime,
)
from benchflow.providers.litellm_runtime import _docker_host_address
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.trajectories._capture import TrajectoryWriter, make_trajectory_sink

HERE = Path(__file__).resolve().parent
AGENT_ENV_DIR = HERE / "agent_env"
CASINOBENCH = Path(os.environ.get("CASINOBENCH_DIR", "/home/liu.10379/casinobench"))
BRIDGE = _docker_host_address()  # the host gateway agent containers can reach

# roster: (agent, model, label, count) → seats <label>-<i>. 10 agents using only the
# models the subscriptions actually expose: codex→gpt-5.5; claude→sonnet-4-6 + haiku-4-5
# (gpt-5/gpt-5-codex/gpt-5.1-codex and claude-opus-4-8 are rejected by the plans).
DEFAULT_ROSTER = [
    ("codex-acp", "gpt-5.5", "codex-gpt-5.5", 4),
    ("claude-agent-acp", "claude-sonnet-4-6", "claude-sonnet-4-6", 3),
    ("claude-agent-acp", "claude-haiku-4-5", "claude-haiku-4-5", 3),
]

PROMPT = """Play the casino games and win as many chips as you can, using the
`casino` command (your seat is already configured):

  casino lobby                       — open games and your bankroll
  casino rules <game_id>             — a game's rules
  casino join <game_id>              — take a seat at a game
  casino observe                     — {request_id, observation, legal_actions, done}
  casino act <request_id> '<json>'   — play ONE of the legal actions
  casino cashier                     — your bankroll

Play through the `casino` CLI. Begin with `casino lobby`."""


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def _start_world(port: int) -> subprocess.Popen:
    # CASINOBENCH_GAMES="" => all 18 ENABLED_GAMES open (build_world_from_env);
    # "all" would be treated as a literal game id and rejected.
    proc = subprocess.Popen(  # noqa: S603,S607
        ["uv", "run", "casino-service", "--host", "0.0.0.0", "--port", str(port)],
        cwd=str(CASINOBENCH),
        env={**os.environ, "CASINO_MULTIPLAYER": "1", "CASINOBENCH_GAMES": "",
             "CASINO_PORT": str(port), "CASINOBENCH_HANDS": "1"},
    )
    for _ in range(200):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                print(f"world: healthy on :{port} (all games)", flush=True)
                return proc
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.2)
    proc.kill()
    raise SystemExit("world: service never became healthy")


def _is_subscription(agent: str) -> bool:
    return getattr(AGENTS.get(agent), "subscription_auth", None) is not None


async def _run_seat(agent: str, model: str, seat_id: str, world_port: int,
                    run_dir: Path) -> dict:
    out = run_dir / seat_id
    (out / "trajectory").mkdir(parents=True, exist_ok=True)
    world_url = f"http://{BRIDGE}:{world_port}"
    sandbox = DockerSandbox(
        environment_dir=AGENT_ENV_DIR, environment_name="casinoseat",
        session_id=seat_id, rollout_paths=None,
        task_env_config=SandboxConfig(allow_internet=True),
    )
    runtime = None
    n_tools = 0
    status = "ok"
    try:
        await sandbox.start(force_build=False)
        if _is_subscription(agent):
            # upload host ~/.codex/auth.json or ~/.claude/.credentials.json into
            # the container BEFORE connect (the rollout would do this; we bypass it)
            await upload_subscription_auth(sandbox, agent, "/root")
            agent_env = {"CASINO_URL": world_url, "CASINOBENCH_SEAT_ID": seat_id}
            print(f"{seat_id}: subscription [{agent}/{model}] world={world_url}", flush=True)
        else:  # proxy-routed (e.g. deepseek) → per-seat raw llm_trajectory
            provider_env, runtime = await ensure_litellm_runtime(
                agent=agent,
                agent_env={"DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", "")},
                model="deepseek/deepseek-v4-pro", runtime=None, environment="docker",
                session_id=f"casino-{seat_id}",
            )
            agent_env = {**provider_env, "CASINO_URL": world_url,
                         "CASINOBENCH_SEAT_ID": seat_id}
            print(f"{seat_id}: proxy [{agent}/{model}] world={world_url}", flush=True)

        client, session, _adapter, _name = await connect_acp(
            env=sandbox, agent=agent, agent_launch=AGENTS[agent].launch_cmd,
            agent_env=agent_env, sandbox_user=None, model=model,
            rollout_dir=out, environment="docker", agent_cwd="/app",
        )
        # stream the trajectory to disk on every ACP update — so it is visible
        # LIVE in the viewer and survives a wall-clock timeout (the agent may
        # play hundreds of moves without finishing all 18 games).
        writer = TrajectoryWriter(out / "trajectory" / "acp_trajectory.jsonl")
        session.on_change = make_trajectory_sink(writer, [])
        try:
            _traj, n_tools = await execute_prompts(
                acp_client=client, session=session, prompts=[PROMPT],
                timeout=1200, idle_timeout=300,
            )
        except AgentPromptTimeoutError as exc:
            n_tools = getattr(exc, "n_tool_calls", len(getattr(session, "tool_calls", [])))
            status = f"timeout (played {n_tools} moves)"  # non-fatal: it DID play
        finally:
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        status = f"error: {type(exc).__name__}: {str(exc)[:200]}"
        print(f"{seat_id}: {status}", flush=True)
    finally:
        if runtime is not None:
            await asyncio.sleep(1.0)
            await stop_provider_runtime(runtime)
    n_llm = 0
    rt_traj = getattr(getattr(runtime, "server", None), "trajectory", None)
    if rt_traj is not None and rt_traj.exchanges:
        (out / "trajectory" / "llm_trajectory.jsonl").write_text(
            rt_traj.to_jsonl(redact_keys=True))
        n_llm = len(rt_traj.exchanges)
    try:
        await sandbox.stop(delete=True)
    except Exception:  # noqa: BLE001
        pass
    return {"seat": seat_id, "agent": agent, "model": model, "status": status,
            "acp_tool_calls": n_tools, "llm_calls": n_llm,
            "subscription": _is_subscription(agent),
            "usage": extract_usage(runtime) if runtime is not None else {}}


def _seats(roster) -> list[tuple[str, str, str]]:
    specs = []
    for agent, model, label, count in roster:
        for i in range(count):
            specs.append((agent, model, f"{label}-{i}"))
    return specs


async def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/casino-floor/all-games")
    ap.add_argument("--world-port", type=int, default=0,
                    help="fixed World port (0 = ephemeral); set for a live viewer")
    args = ap.parse_args()

    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    specs = _seats(DEFAULT_ROSTER)
    # roster for the viewer (seat -> agent/model), written before play starts
    (run_dir / "roster.json").write_text(json.dumps(
        [{"seat": s, "agent": a, "model": m} for a, m, s in specs]))
    world_port = args.world_port or _free_port()
    world = await _start_world(world_port)
    standings = {}
    results = []
    try:
        print(f"floor: {len(specs)} seats · {', '.join(s[2] for s in specs)}", flush=True)
        results = await asyncio.gather(*[
            _run_seat(agent, model, seat_id, world_port, run_dir)
            for agent, model, seat_id in specs
        ])
        try:
            standings = httpx.get(f"http://127.0.0.1:{world_port}/_admin/standings",
                                  timeout=10).json()
        except httpx.HTTPError:
            pass
        # snapshot the merged event log so the viewer can be built --from this run
        try:
            ev = httpx.get(f"http://127.0.0.1:{world_port}/_admin/events", timeout=10).json()
            (run_dir / "events.jsonl").write_text(ev.get("jsonl", ""))
        except httpx.HTTPError:
            pass
    finally:
        world.terminate()
        try:
            world.wait(timeout=10)
        except subprocess.TimeoutExpired:
            world.kill()

    print("\n=== floor results ===")
    for r in results:
        print(f"  {r['seat']:<22} {r['status']:<30} acp_tool_calls={r['acp_tool_calls']} "
              f"llm={r['llm_calls']} sub={r['subscription']}")
    print("standings:", json.dumps(standings))
    (run_dir / "floor.json").write_text(json.dumps(
        {"results": results, "standings": standings}, indent=2))
    ok = sum(1 for r in results if r["status"] == "ok")
    played = sum(1 for r in results if r["acp_tool_calls"] > 0)
    print(f"\n{ok}/{len(results)} seats ok · {played} actually played · "
          f"per-seat trajectories under {run_dir}")
    return 0 if played else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
