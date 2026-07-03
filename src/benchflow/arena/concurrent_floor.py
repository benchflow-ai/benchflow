"""Shared-sandbox concurrent multi-agent runner — the native floor.

``run_concurrent_floor(roster, sandbox, service_url, config)`` runs every seat of
a decoupled ``Roster`` CONCURRENTLY against ONE shared task service, all inside
ONE shared sandbox, each agent pinned to its own ``/work/<seat>`` folder (cwd =
identity = I/O). Per seat it captures a separate ACP trajectory, and — for
proxy-routed seats — a separate raw ``llm_trajectory.jsonl`` from that seat's own
LiteLLM proxy. Subscription seats (codex/claude oauth) call their provider
directly, so they get an ACP trajectory only (flagged ``raw=false``).

The roster is the A/M axis only; the run-level config (out, drive, prompt,
deadlines, the service-env var names) arrives as a :class:`FloorConfig` built from
the standard ``bench eval run`` flags — NOT baked into the roster file. The
orchestrator stays pure over an *already-started* ``sandbox`` + ``service_url``
(the docker/daytona/in-sandbox-service bootstrap is the caller's job), which keeps
it testable with in-memory fakes.

Two drive modes (``FloorConfig.drive``), orthogonal to agent protocol:
  * ``auto-loop`` (default, verified) — one prompt; the agent runs its own
    observe→act loop via the in-sandbox CLI. Multi-round happens inside the prompt.
  * ``service-rounds`` (structural) — the mock service drives the rounds: poll the
    shared service per seat and re-prompt (nudge) the seat only on ``YOUR_TURN``,
    until ``DONE``/deadline. Re-entrant ``prompt_seat`` per round.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.acp.runtime import AgentPromptTimeoutError
from benchflow.agents.credentials import upload_subscription_auth
from benchflow.agents.env import uses_native_subscription_auth
from benchflow.arena.agent_driver import close_seat, connect_seat, prompt_seat
from benchflow.arena.agents_manifest import Seat
from benchflow.arena.instructions import write_agent_instructions
from benchflow.arena.protocol import Observation, SeatStatus
from benchflow.arena.roster import Roster
from benchflow.providers import ensure_litellm_runtime, stop_provider_runtime
from benchflow.trajectories._capture import TrajectoryWriter, make_trajectory_sink

__all__ = ["FloorConfig", "run_concurrent_floor", "HttpSeatClient"]


@dataclass
class FloorConfig:
    """Run-level config for a floor — the bits that come from CLI flags, not the
    roster file. ``url_env``/``seat_env`` name the env vars the task CLI reads for
    the service URL and this seat's id (e.g. ``CASINO_URL`` / ``CASINOBENCH_SEAT_ID``)."""

    out: str | Path
    drive: str = "auto-loop"
    prompt: str | None = None
    deadline_s: int = 1200
    idle_timeout_s: int = 300
    url_env: str | None = None
    seat_env: str | None = None
    standings_path: str | None = None  # in-sandbox path → {seat: score} for reward
    events_path: str | None = None  # in-sandbox path → service event log
    environment: str = "docker"


def _provider_keys() -> dict[str, str]:
    """The provider API keys present in env — handed to a proxy seat's runtime
    (the proxy strips them; the raw key never reaches the agent)."""
    return {k: v for k, v in os.environ.items() if k.endswith("_API_KEY") and v}


class HttpSeatClient:
    """Minimal ``SeatClient`` over the shared service's ``/observe`` + ``/act``
    (the arena turn-poll contract). Used by the ``service-rounds`` drive."""

    def __init__(self, base_url: str, http: Any) -> None:
        self.base = base_url.rstrip("/")
        self.http = http

    async def observe(self, seat_id: str) -> dict[str, Any]:
        r = await self.http.get(f"{self.base}/observe", params={"seat": seat_id}, timeout=10)
        return r.json()

    async def act(self, seat_id: str, request_id: str, action: dict) -> dict[str, Any]:
        r = await self.http.post(
            f"{self.base}/act",
            json={"seat": seat_id, "request_id": request_id, "action": action},
            timeout=30,
        )
        return r.json()


def _seat_env(cfg: FloorConfig, seat: Seat, service_url: str) -> dict[str, str]:
    env = {"BENCHFLOW_SERVICE_URL": service_url, "BENCHFLOW_SEAT_ID": seat.seat_id}
    if cfg.url_env:  # e.g. CASINO_URL — the var the task CLI reads
        env[cfg.url_env] = service_url
    if cfg.seat_env:  # e.g. CASINOBENCH_SEAT_ID — the player id
        env[cfg.seat_env] = seat.seat_id
    env.update(seat.spec.env)
    return env


def _render_round(base_prompt: str, obs: Observation, round_no: int) -> str:
    return (
        f"{base_prompt}\n\n[Round {round_no}] It is your turn "
        f"(request_id={obs.request_id}).\n"
        f"Observation: {json.dumps(obs.public)}\n"
        f"Legal actions: {json.dumps(obs.legal_actions)}\n"
        "Take exactly ONE action using your tools, then stop."
    )


async def _drive_service_rounds(
    conn, seat: Seat, cfg: FloorConfig, seat_client, deadline: float
) -> tuple[str, int, list[dict]]:
    """The mock service drives the rounds: nudge the seat only on its turn.

    Returns ``(status, total_tool_calls, last_trajectory)`` — tool calls summed
    across rounds (the status string carries the round count)."""
    base = cfg.prompt or "Play your turn."
    rounds, tools = 0, 0
    last_traj: list[dict] = []
    while time.monotonic() < deadline:
        obs = Observation.from_payload(await seat_client.observe(seat.seat_id))
        if obs.done:
            return (f"done ({rounds} rounds)", tools, last_traj)
        if obs.status is SeatStatus.YOUR_TURN:
            rounds += 1
            last_traj, n = await prompt_seat(
                conn, _render_round(base, obs, rounds),
                timeout=cfg.deadline_s, idle_timeout=cfg.idle_timeout_s,
            )
            tools += n
        else:
            await asyncio.sleep(0.1)
    return (f"deadline ({rounds} rounds)", tools, last_traj)


async def _run_seat(
    seat: Seat,
    *,
    roster: Roster,
    cfg: FloorConfig,
    sandbox: Any,
    service_url: str,
    run_dir: Path,
    http: Any | None,
    seat_client_factory,
) -> dict[str, Any]:
    out = run_dir / seat.seat_id
    (out / "trajectory").mkdir(parents=True, exist_ok=True)
    agent_cfg = seat.config
    runtime = None
    status, n_tools, n_llm = "ok", 0, 0
    try:
        await sandbox.exec(f"mkdir -p {shlex.quote(seat.agent_cwd)}", timeout_sec=20)
        await write_agent_instructions(
            sandbox, seat.agent_cwd, agent_cfg, roster.instructions_path(seat.spec)
        )
        # Decide subscription vs proxy by ACTUAL auth, not capability: a claude/codex
        # seat with an API key wants the proxy (raw+acp); only an oauth-only seat
        # (no key, host login present) goes provider-direct (acp-only).
        agent_env = {**_provider_keys(), **_seat_env(cfg, seat, service_url)}
        if uses_native_subscription_auth(agent_cfg.name, seat.spec.model, agent_env):
            await upload_subscription_auth(sandbox, agent_cfg.name, "/root")
        else:  # API-key seat → its OWN proxy → separate raw llm_trajectory
            provider_env, runtime = await ensure_litellm_runtime(
                agent=agent_cfg.name, agent_env=agent_env, model=seat.spec.model,
                runtime=None, environment=cfg.environment,
                session_id=f"floor-{seat.seat_id}",
                # daytona/modal are sandbox-local: the proxy must run INSIDE the
                # shared sandbox (the remote agent reaches it on localhost), so it
                # needs the sandbox handle. Ignored for docker (host proxy + bridge).
                sandbox=sandbox,
            )
            agent_env = {**agent_env, **provider_env}

        conn = await connect_seat(
            agent_cfg, env=sandbox, agent_cwd=seat.agent_cwd, agent_env=agent_env,
            model=seat.spec.model, rollout_dir=out, environment=cfg.environment,
            seat_id=seat.seat_id, reasoning_effort=seat.spec.reasoning_effort,
        )
        # stream the ACP trajectory live (survives a wall-clock timeout)
        writer = TrajectoryWriter(out / "trajectory" / "acp_trajectory.jsonl")
        # non-ACP sessions stream via write_final, so a missing/odd hook is fine
        with contextlib.suppress(Exception):
            conn.session.on_change = make_trajectory_sink(writer, [])
        try:
            if cfg.drive == "service-rounds":
                if seat_client_factory is not None:
                    client = seat_client_factory(seat.seat_id)
                else:
                    client = HttpSeatClient(service_url, http)
                deadline = time.monotonic() + cfg.deadline_s
                status, n_tools, traj = await _drive_service_rounds(
                    conn, seat, cfg, client, deadline
                )
            else:  # auto-loop
                if not cfg.prompt:
                    raise ValueError("auto-loop drive requires a prompt (--prompt)")
                traj, n_tools = await prompt_seat(
                    conn, cfg.prompt,
                    timeout=cfg.deadline_s, idle_timeout=cfg.idle_timeout_s,
                )
            if traj:
                writer.write_final(traj)
        # timeout is non-fatal ("it DID play") — keep it distinct from "error".
        # session-factory prompts time out via asyncio.TimeoutError, ACP via its own.
        except (TimeoutError, AgentPromptTimeoutError) as exc:
            n_tools = getattr(exc, "n_tool_calls", n_tools)
            status = f"timeout (played {n_tools} moves)"
        finally:
            await close_seat(conn)
    except Exception as exc:
        status = f"error: {type(exc).__name__}: {str(exc)[:200]}"
    finally:
        if runtime is not None:
            await asyncio.sleep(1.0)  # let the proxy callback flush before stop
            await stop_provider_runtime(runtime)
    # per-seat raw llm trajectory — only when a proxy actually started
    rt_traj = getattr(getattr(runtime, "server", None), "trajectory", None)
    if rt_traj is not None and getattr(rt_traj, "exchanges", None):
        (out / "trajectory" / "llm_trajectory.jsonl").write_text(
            rt_traj.to_jsonl(redact_keys=True)
        )
        n_llm = len(rt_traj.exchanges)
    return {
        "seat": seat.seat_id, "agent": agent_cfg.name, "model": seat.spec.model,
        "protocol": agent_cfg.protocol, "byoa": seat.is_byoa,
        "raw": runtime is not None,  # raw captured iff a proxy ran for this seat
        "status": status, "acp_tool_calls": n_tools, "llm_calls": n_llm,
    }


async def run_concurrent_floor(
    roster: Roster,
    *,
    sandbox: Any,
    service_url: str,
    config: FloorConfig,
    http: Any | None = None,
    seat_client_factory=None,
) -> dict[str, Any]:
    """Run all seats of ``roster`` concurrently in one shared ``sandbox`` against
    ``service_url``.

    ``sandbox`` is already started; ``service_url`` already reachable from inside
    it. Writes ``<config.out>/<seat>/trajectory/{acp,llm}_trajectory.jsonl`` +
    ``roster.json`` + ``floor.json``. Returns the floor summary.
    """
    run_dir = Path(config.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    seats = roster.seats()
    (run_dir / "roster.json").write_text(json.dumps(
        [{"seat": s.seat_id, "agent": s.config.name, "model": s.spec.model,
          "protocol": s.config.protocol, "byoa": s.is_byoa} for s in seats],
        indent=2,
    ))

    results = await asyncio.gather(*[
        _run_seat(
            s, roster=roster, cfg=config, sandbox=sandbox, service_url=service_url,
            run_dir=run_dir, http=http, seat_client_factory=seat_client_factory,
        )
        for s in seats
    ])

    summary = {"results": results, "drive": config.drive, "service_url": service_url}
    (run_dir / "floor.json").write_text(json.dumps(summary, indent=2))
    return summary
