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
from pathlib import Path
from typing import Any

from benchflow.arena.concurrent_floor import FloorConfig, run_concurrent_floor
from benchflow.arena.roster import Roster

__all__ = ["bootstrap_shared_env", "run_native_floor"]


def _make_sandbox(image: str, manifest_dir: Path, environment: str) -> Any:
    """A ONE shared sandbox started from the manifest's prebuilt image."""
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task.config import SandboxConfig

    return DockerSandbox(
        environment_dir=manifest_dir,
        environment_name="native-floor",
        session_id="native-floor",
        rollout_paths=None,
        task_env_config=SandboxConfig(docker_image=image, allow_internet=True),
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
    from benchflow.environment.manifest import load_manifest, resolve_manifest_image
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
        sandbox = _make_sandbox(image, Path(environment_manifest).resolve().parent, environment)
        await sandbox.start(force_build=False)

    # Inject the floor's service env (e.g. CASINO_MULTIPLAYER=1) before provision so
    # the in-sandbox service starts in the right mode.
    for key, value in (service_env or {}).items():
        with contextlib.suppress(Exception):
            await sandbox.exec(f"export {key}={value}", timeout_sec=10)

    env = _env or ManifestEnvironment(manifest, sandbox=sandbox)
    await env.provision({"task_id": game})
    await env.readiness()

    port = manifest.services[0].port if manifest.services else 9001
    service_url = f"http://localhost:{port}"

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


async def _attach_reward(sandbox: Any, summary: dict, service_url: str, cfg: FloorConfig) -> None:
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
        environment_manifest, environment=config.environment, game=game,
        service_env=service_env,
    )
    try:
        summary = await run_concurrent_floor(
            roster, sandbox=sandbox, service_url=service_url, config=config,
        )
        await _attach_reward(sandbox, summary, service_url, config)
        return summary
    finally:
        await teardown()
