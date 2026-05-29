"""v0.5 Phase 2 — snapshot/restore proven on a REAL SQLite DB in a REAL container.

The moat (Han: "Environment 总是要 roll out, roll back") was only stub-tested
against ``FakeSandbox`` — `tests/environment/test_manifest_env.py` asserts the
*commands* are issued, never that a real round-trip rolls state back. This is
the integration gate: a real ``DockerSandbox`` runs the real
``sqlite3 .backup`` / ``cp`` that ``ManifestEnvironment.snapshot``/``restore``
issue, and we assert a mutation made after the snapshot is gone after restore.

Docker-gated: skipped when no Docker daemon is reachable, so the default unit
suite stays fast and host-independent. Run locally / in a Docker-capable CI
lane to exercise it.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from benchflow.environment.manifest import EnvironmentManifest
from benchflow.environment.manifest_env import ManifestEnvironment
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task import RolloutPaths, SandboxConfig

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="Docker not available — real snapshot/restore round-trip is gated",
)

# Minimal stateful manifest: just [environment.state] over one sqlite file. We
# drive snapshot()/restore() directly against a real container, so no services
# / provision are needed.
_MANIFEST = EnvironmentManifest.model_validate_toml(
    """
[environment]
name           = "snap-roundtrip-test"
image          = "alpine:3.20"
owns_lifecycle = true

[environment.state]
kind  = "sqlite"
paths = ["/data/test.db"]
"""
)

_DOCKERFILE = textwrap.dedent(
    """\
    FROM alpine:3.20
    RUN apk add --no-cache sqlite
    CMD ["sleep", "infinity"]
    """
)


async def _count_rows(sandbox: DockerSandbox) -> int:
    result = await sandbox.exec(
        'sqlite3 /data/test.db "SELECT count(*) FROM t;"', timeout_sec=30
    )
    assert result.return_code == 0, result.stderr
    return int(result.stdout.strip())


@pytest.mark.asyncio
async def test_manifest_env_snapshot_restore_rolls_back_real_sqlite(tmp_path: Path):
    """Seed → snapshot → mutate → restore → assert the mutation rolled back."""
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text(_DOCKERFILE)

    rollout_dir = tmp_path / "rollout"
    for sub in ("agent", "verifier", "artifacts", "trajectory"):
        (rollout_dir / sub).mkdir(parents=True, exist_ok=True)

    sandbox = DockerSandbox(
        environment_dir=env_dir,
        environment_name="snap-roundtrip-test",
        session_id="phase2-gate",
        rollout_paths=RolloutPaths(rollout_dir=rollout_dir),
        task_env_config=SandboxConfig(),
    )
    await sandbox.start(force_build=True)
    try:
        # Seed a real DB with one row.
        seed = await sandbox.exec(
            'mkdir -p /data && sqlite3 /data/test.db '
            '"CREATE TABLE t(x INTEGER); INSERT INTO t VALUES (1);"',
            timeout_sec=30,
        )
        assert seed.return_code == 0, seed.stderr
        assert await _count_rows(sandbox) == 1

        env = ManifestEnvironment(_MANIFEST, sandbox=sandbox)

        # Real snapshot (sqlite3 .backup into an in-sandbox snapshot dir).
        snap = await env.snapshot()
        assert snap.id and snap.path

        # Mutate after the snapshot.
        mutate = await sandbox.exec(
            'sqlite3 /data/test.db "INSERT INTO t VALUES (2);"', timeout_sec=30
        )
        assert mutate.return_code == 0, mutate.stderr
        assert await _count_rows(sandbox) == 2  # mutation visible

        # Real restore (cp the backup back over the live file) rolls it back.
        await env.restore(snap)
        assert await _count_rows(sandbox) == 1  # mutation gone
    finally:
        await sandbox.stop(delete=True)
