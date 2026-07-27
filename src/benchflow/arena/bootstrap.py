"""Seed ONE shared sandbox + ONE IN-SANDBOX service, then run the native floor.

The casino (and any benchmark) service runs **inside** the rollout's own sandbox,
declared by an ``environment.toml`` ``[[environment.services]]`` manifest and
started by :class:`ManifestEnvironment` (``nohup`` over ``sandbox.exec``,
health-gated on in-sandbox ``curl localhost:<port>/health``) — NOT a host
subprocess, never reached over a docker bridge. Agents share the sandbox's network
namespace, so they reach the service on ``localhost``. This is the env-0/ClawsBench
contract; the runner is just the caller that seeds the env and hands the
already-started ``(sandbox, service_url)`` to :func:`run_concurrent_floor`.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from pathlib import Path
from typing import Any

from benchflow.arena.concurrent_floor import FloorConfig, run_concurrent_floor
from benchflow.arena.roster import Roster

__all__ = ["bootstrap_shared_env", "run_native_floor"]


def _make_sandbox(
    image: str, manifest_dir: Path, environment: str, service_env: dict[str, str] | None
) -> Any:
    """A ONE shared sandbox started from the manifest's prebuilt image.

    ``service_env`` (e.g. ``CASINO_MULTIPLAYER=1``) goes into the sandbox's
    persistent env, so it reaches EVERY exec — including ManifestEnvironment's
    ``nohup casino-service`` start, which is how the service comes up in floor
    (multiplayer) mode. ``environment`` selects docker (local) vs daytona (remote);
    both sandboxes share the same constructor shape."""
    from benchflow.task.config import SandboxConfig

    cfg = SandboxConfig(
        docker_image=image, allow_internet=True, env=dict(service_env or {})
    )
    # Daytona runs the whole floor (N ACP agents + proxy-seat LiteLLM + the service)
    # in ONE remote sandbox whose 1cpu/2GB default OOM-kills the agents mid-play;
    # size it up (daytona caps at 4 cpu/sandbox). Docker inherits the HOST's memory,
    # so an explicit cap there only over-constrains it — leave it unset for docker.
    if environment == "daytona":
        cfg.cpus, cfg.memory_mb = 4, 16384
    # Unique per-run session: a fixed name makes concurrent floor runs on one
    # host share a docker-compose project and tear each other down.
    run_id = f"native-floor-{uuid.uuid4().hex[:8]}"
    if environment == "daytona":
        from benchflow.sandbox.daytona import DaytonaSandbox

        # Keep the shared floor sandbox alive for the whole run: the defaults (0)
        # let Daytona stop+delete it during agent think-gaps mid-play, killing every
        # ACP session ("remote container killed / idle sleep"). Match the normal
        # rollout's 24h keep-alive; teardown() deletes it explicitly when done.
        return DaytonaSandbox(
            environment_dir=manifest_dir,
            environment_name=run_id,
            session_id=run_id,
            rollout_paths=None,
            task_env_config=cfg,
            auto_stop_interval_mins=1440,
            auto_delete_interval_mins=1440,
        )
    from benchflow.sandbox.docker import DockerSandbox

    return DockerSandbox(
        environment_dir=manifest_dir,
        environment_name=run_id,
        session_id=run_id,
        rollout_paths=None,
        task_env_config=cfg,
    )


async def bootstrap_shared_env(
    environment_manifest: str | Path,
    *,
    environment: str = "docker",
    game: str | None = None,
    service_env: dict[str, str] | None = None,
    _sandbox: Any | None = None,
    _env: Any | None = None,
):
    """Return ``(sandbox, service_url, teardown)`` — ONE shared sandbox with the
    manifest's service started IN-SANDBOX on ``localhost:<port>``.

    ``_sandbox``/``_env`` are injection seams for unit tests (no docker needed).
    ``game`` is the ``task_selection`` value (e.g. the casino game id).
    """
    from benchflow.environment.manifest import (
        load_manifest,
        resolve_manifest_image,
        resolve_manifest_runtime_env,
    )
    from benchflow.environment.manifest_env import ManifestEnvironment

    manifest = load_manifest(environment_manifest)
    sandbox = _sandbox
    if sandbox is None:
        image = resolve_manifest_image(manifest)
        if not image:
            raise SystemExit(
                f"{environment_manifest}: no runnable `image` in the manifest "
                "(set [environment].image to the prebuilt base)."
            )
        # Resolve the manifest's task_selection (game → CASINOBENCH_GAME) + forward_env,
        # then add the floor's service_env — all into the sandbox persistent env so the
        # in-sandbox `casino-service` start sees them.
        sandbox_env = {
            **resolve_manifest_runtime_env(manifest, task_id=game or ""),
            **(service_env or {}),
        }
        sandbox = _make_sandbox(
            image, Path(environment_manifest).resolve().parent, environment, sandbox_env
        )
        await sandbox.start(force_build=False)

    env = _env or ManifestEnvironment(manifest, sandbox=sandbox)
    await env.provision({"task_id": game})
    await env.readiness()

    if not manifest.services:
        raise ValueError(
            "the concurrent floor requires the manifest to declare an in-sandbox "
            "service ([[environment.services]]); none found"
        )
    service_url = f"http://localhost:{manifest.services[0].port}"

    async def teardown() -> None:
        with contextlib.suppress(Exception):
            await env.teardown()
        with contextlib.suppress(Exception):
            await sandbox.stop(delete=True)

    return sandbox, service_url, teardown


async def _read_service_json(sandbox: Any, service_url: str, path: str) -> Any:
    """GET a service endpoint FROM INSIDE the sandbox (the service is on the
    sandbox's localhost, unreachable from the host orchestrator)."""
    res = await sandbox.exec(f"curl -sf {service_url}{path}", timeout_sec=15)
    out = getattr(res, "stdout", res)
    return json.loads(out) if out and str(out).strip() else None


async def _attach_reward(
    sandbox: Any, summary: dict, service_url: str, cfg: FloorConfig
) -> None:
    """Opt-in: fetch final standings IN-SANDBOX and write a per-seat reward vector."""
    path = getattr(cfg, "standings_path", None)
    if not path:
        return
    with contextlib.suppress(Exception):
        from benchflow.arena.reward import SharedEnvReward

        standings = await _read_service_json(sandbox, service_url, path)
        if isinstance(standings, dict) and standings:
            summary["standings"] = standings
            summary["reward"] = SharedEnvReward().score(standings)
            (Path(cfg.out) / "floor.json").write_text(json.dumps(summary, indent=2))


def _count_actions(events_jsonl: str) -> dict[str, dict[str, int]]:
    """Per-seat REAL casino activity from the service event log — acp tool
    calls are polls+scripting, not game actions (a seat once showed 77 calls
    but 1066 applied actions)."""
    counts: dict[str, dict[str, int]] = {}
    for line in events_jsonl.splitlines():
        if not line.strip():
            continue
        with contextlib.suppress(Exception):
            e = json.loads(line)
            actor = e.get("actor")
            kind = e.get("type")
            if not actor or kind not in ("action_applied", "action_timeout"):
                continue
            c = counts.setdefault(actor, {"actions": 0, "timeouts": 0})
            c["actions" if kind == "action_applied" else "timeouts"] += 1
    return counts


async def _snapshot_events(sandbox: Any, service_url: str, cfg: FloorConfig) -> None:
    """Opt-in: snapshot the service event log IN-SANDBOX → events.jsonl (for the
    town viewer's animated board). The service is on the sandbox's localhost, so
    this reads it via sandbox.exec, not a host client."""
    path = getattr(cfg, "events_path", None)
    if not path:
        return
    with contextlib.suppress(Exception):
        data = await _read_service_json(sandbox, service_url, path)
        jsonl = data.get("jsonl") if isinstance(data, dict) else None
        if jsonl:
            (Path(cfg.out) / "events.jsonl").write_text(jsonl)


async def run_native_floor(
    roster: Roster,
    *,
    environment_manifest: str | Path,
    config: FloorConfig,
    game: str | None = None,
    service_env: dict[str, str] | None = None,
) -> dict:
    """Bootstrap the in-sandbox shared env, run every seat concurrently, tear down."""
    sandbox, service_url, teardown = await bootstrap_shared_env(
        environment_manifest,
        environment=config.environment,
        game=game,
        service_env=service_env,
    )
    try:
        summary = await run_concurrent_floor(
            roster,
            sandbox=sandbox,
            service_url=service_url,
            config=config,
        )
        await _attach_reward(sandbox, summary, service_url, config)
        await _snapshot_events(sandbox, service_url, config)
        _attach_activity(summary, config)
        with contextlib.suppress(Exception):
            outcomes = await _read_service_json(
                sandbox, service_url, "/_admin/outcomes"
            )
            if isinstance(outcomes, dict) and outcomes:
                summary["outcomes"] = outcomes
                (Path(config.out) / "floor.json").write_text(
                    json.dumps(summary, indent=2)
                )
        return summary
    finally:
        await teardown()


def _attach_activity(summary: dict, cfg: FloorConfig) -> None:
    """Honest per-seat metrics: merge real casino action counts (from the
    snapshotted event log) into the floor results + floor.json."""
    with contextlib.suppress(Exception):
        ev = Path(cfg.out) / "events.jsonl"
        if not ev.exists():
            return
        counts = _count_actions(ev.read_text())
        for r in summary.get("results", []):
            c = counts.get(r.get("seat"), {"actions": 0, "timeouts": 0})
            r["casino_actions"] = c["actions"]
            r["casino_timeouts"] = c["timeouts"]
        (Path(cfg.out) / "floor.json").write_text(json.dumps(summary, indent=2))
