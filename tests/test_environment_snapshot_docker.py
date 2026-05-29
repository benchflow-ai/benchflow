"""v0.5 Phase 2 — snapshot/restore proven on a REAL SQLite DB in a REAL container.

The moat (Han: "Environment 总是要 roll out, roll back") was only stub-tested
against ``FakeSandbox`` — `tests/environment/test_manifest_env.py` asserts the
*commands* are issued, never that a real round-trip rolls state back. This is
the integration gate: a real ``DockerSandbox`` runs the real backup/restore
that ``ManifestEnvironment.snapshot``/``restore`` issue, and we assert a
mutation made after the snapshot is gone after restore.

The image is ``python:3.12-slim`` *on purpose* — it has Python's ``sqlite3``
module but **not** the ``sqlite3`` CLI, exactly like the real ClawsBench
(smolclaws) image. The ClawsBench e2e run caught that the CLI is absent there
(``sqlite3: not found``), so this test now exercises the python3 fallback path
end-to-end — the real-benchmark regression, reproduced in CI.

Docker-gated via the ``docker`` marker (excluded from the default suite) and the
``docker_daemon`` fixture; run with ``pytest -m docker``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from benchflow.environment.manifest import EnvironmentManifest
from benchflow.environment.manifest_env import (
    ManifestEnvironment,
    _sqlite_backup_command,
)
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task import RolloutPaths, SandboxConfig

# Minimal stateful manifest: just [environment.state] over one sqlite file. We
# drive snapshot()/restore() directly against a real container, so no services
# / provision are needed.
_MANIFEST = EnvironmentManifest.model_validate_toml(
    """
[environment]
name           = "snap-roundtrip-test"
image          = "python:3.12-slim"
owns_lifecycle = true

[environment.state]
kind  = "sqlite"
paths = ["/data/test.db"]
"""
)

# python:3.12-slim has the sqlite3 *module* but not the sqlite3 *CLI* — the same
# shape as the smolclaws ClawsBench image that surfaced the regression.
_DOCKERFILE = textwrap.dedent(
    """\
    FROM python:3.12-slim
    CMD ["sleep", "infinity"]
    """
)


def _py_sqlite(stmt: str) -> str:
    """A shell command running one SQLite statement via python3 (no CLI)."""
    return (
        'python3 -c "'
        "import sqlite3; "
        "c = sqlite3.connect('/data/test.db'); "
        f"c.execute('{stmt}'); "
        "c.commit(); c.close()"
        '"'
    )


async def _count_rows(sandbox: DockerSandbox) -> int:
    result = await sandbox.exec(
        'python3 -c "'
        "import sqlite3; "
        "print(sqlite3.connect('/data/test.db').execute('SELECT count(*) FROM t').fetchone()[0])"
        '"',
        timeout_sec=30,
    )
    assert result.return_code == 0, result.stderr
    return int(result.stdout.strip())


def test_sqlite_backup_command_prefers_cli_falls_back_to_python():
    """The backup command tries the sqlite3 CLI, then python3 — guards the
    ClawsBench regression (smolclaws ships python3 but not the sqlite3 CLI)."""
    cmd = _sqlite_backup_command("/data/x.db", "/snap/x.db")
    assert "command -v sqlite3" in cmd
    assert ".backup" in cmd  # the CLI branch
    assert "python3 -c" in cmd  # the fallback branch
    assert "sqlite3.connect" in cmd


@pytest.mark.docker
@pytest.mark.asyncio
async def test_manifest_env_snapshot_restore_rolls_back_real_sqlite(
    docker_daemon: None, tmp_path: Path
):
    """Seed → snapshot → mutate → restore → assert rollback, on a python-only
    image (no sqlite3 CLI) — the real ClawsBench scenario via the python3 path."""
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text(_DOCKERFILE)

    rollout_paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    rollout_paths.mkdir()

    sandbox = DockerSandbox(
        environment_dir=env_dir,
        environment_name="snap-roundtrip-test",
        session_id="phase2-gate",
        rollout_paths=rollout_paths,
        task_env_config=SandboxConfig(),
    )
    await sandbox.start(force_build=True)
    try:
        # Prove we are exercising the fallback: the image has no sqlite3 CLI.
        cli = await sandbox.exec("command -v sqlite3", timeout_sec=10)
        assert cli.return_code != 0, (
            "test image must NOT ship the sqlite3 CLI — the point is to exercise "
            "the python3 backup fallback (the real ClawsBench/smolclaws shape)"
        )

        # Seed a real DB with one row (via python3 — no CLI available).
        seed = await sandbox.exec(
            "mkdir -p /data && " + _py_sqlite("CREATE TABLE t(x INTEGER)"),
            timeout_sec=30,
        )
        assert seed.return_code == 0, seed.stderr
        ins = await sandbox.exec(_py_sqlite("INSERT INTO t VALUES (1)"), timeout_sec=30)
        assert ins.return_code == 0, ins.stderr
        assert await _count_rows(sandbox) == 1

        env = ManifestEnvironment(_MANIFEST, sandbox=sandbox)

        # Real snapshot (python3 sqlite3.backup into an in-sandbox snapshot dir).
        snap = await env.snapshot()
        assert snap.id and snap.path

        # Mutate after the snapshot.
        mutate = await sandbox.exec(_py_sqlite("INSERT INTO t VALUES (2)"), timeout_sec=30)
        assert mutate.return_code == 0, mutate.stderr
        assert await _count_rows(sandbox) == 2  # mutation visible

        # Real restore (cp the backup back over the live file) rolls it back.
        await env.restore(snap)
        assert await _count_rows(sandbox) == 1  # mutation gone
    finally:
        await sandbox.stop(delete=True)
