"""Seed ONE shared sandbox + ONE shared service, then run the native floor.

The orchestrator (:func:`run_concurrent_floor`) is pure over an already-started
sandbox + service URL. This module is the runnable glue that produces them from an
``agents.yaml`` and tears them down — the proven docker + host-subprocess-service
path (one shared sandbox for all seats, the service reached over the docker
bridge). In-sandbox / daytona service bootstrap is deferred (see the PR notes).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import socket
import subprocess
from pathlib import Path

import httpx

from benchflow.arena.agents_manifest import AgentsManifest
from benchflow.arena.concurrent_floor import run_concurrent_floor

__all__ = ["bootstrap_shared_env", "run_native_floor"]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _kill_service_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the service's whole process group, not just the launcher.

    The host service (e.g. ``uv run …`` → uvicorn) spawns children in its own
    session; signalling only ``proc`` leaves the uvicorn workers running. We start
    it with ``start_new_session=True`` so it leads a process group we can reap."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(proc.pid), sig)


async def _start_host_service(manifest: AgentsManifest, port: int) -> subprocess.Popen:
    svc = manifest.services
    assert svc.command is not None
    cmd = svc.command.format(port=port).split()
    proc = subprocess.Popen(
        cmd,
        cwd=os.path.expanduser(svc.cwd) if svc.cwd else None,
        env={**os.environ, **svc.env, "PORT": str(port)},
        start_new_session=True,  # own process group → teardown reaps children
    )
    for _ in range(200):
        try:
            r = httpx.get(f"http://127.0.0.1:{port}{svc.health}", timeout=2)
            if r.status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.2)
    _kill_service_group(proc, signal.SIGKILL)
    with contextlib.suppress(Exception):
        proc.wait(timeout=5)
    raise SystemExit(f"service never became healthy on :{port}")


async def bootstrap_shared_env(manifest: AgentsManifest, *, environment: str = "docker"):
    """Return ``(sandbox, service_url, teardown)`` — ONE shared sandbox + service."""
    from benchflow.providers.litellm_runtime import _docker_host_address
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task.config import SandboxConfig

    if not manifest.sandbox.image_dir:
        raise SystemExit("agents.yaml needs `sandbox.image_dir` (the shared image)")

    proc: subprocess.Popen | None = None
    if manifest.services.url:  # external service → use as-is
        service_url = manifest.services.url
    elif manifest.services.command:  # host subprocess, reached over the bridge
        port = manifest.services.port or _free_port()
        proc = await _start_host_service(manifest, port)
        service_url = f"http://{_docker_host_address()}:{port}"
    else:
        raise SystemExit("agents.yaml needs services.url or services.command")

    sandbox = DockerSandbox(
        environment_dir=manifest.resolve_path(manifest.sandbox.image_dir),
        environment_name=manifest.sandbox.name,
        session_id="native-floor",
        rollout_paths=None,
        task_env_config=SandboxConfig(allow_internet=True),
    )
    await sandbox.start(force_build=False)

    async def teardown() -> None:
        with contextlib.suppress(Exception):
            await sandbox.stop(delete=True)
        if proc is not None:
            _kill_service_group(proc, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
            # sweep any worker that outlived the leader, then reap the leader
            _kill_service_group(proc, signal.SIGKILL)
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)

    return sandbox, service_url, teardown


async def run_native_floor(manifest: AgentsManifest, *, environment: str = "docker") -> dict:
    """Bootstrap the shared env, run every seat concurrently, then tear down."""
    sandbox, service_url, teardown = await bootstrap_shared_env(
        manifest, environment=environment
    )
    http = httpx.AsyncClient()
    try:
        summary = await run_concurrent_floor(
            manifest, sandbox=sandbox, service_url=service_url,
            environment=environment, http=http,
        )
        await _attach_reward(manifest, summary, service_url, http)
        return summary
    finally:
        await http.aclose()
        await teardown()


async def _attach_reward(manifest, summary, service_url, http) -> None:
    """Opt-in: fetch the shared service's final standings (before teardown) and
    write a per-seat reward vector into floor.json."""
    path = manifest.services.standings_path
    if not path:
        return
    with contextlib.suppress(Exception):
        from benchflow.arena.reward import SharedEnvReward

        standings = (await http.get(f"{service_url}{path}", timeout=10)).json()
        if isinstance(standings, dict) and standings:
            summary["standings"] = standings
            summary["reward"] = SharedEnvReward().score(standings)
            (Path(manifest.out) / "floor.json").write_text(json.dumps(summary, indent=2))
