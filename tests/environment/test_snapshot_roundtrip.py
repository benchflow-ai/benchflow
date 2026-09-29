"""Execute snapshot commands against real SQLite databases in a local test sandbox."""

import asyncio
import shutil
import sqlite3

import pytest

from benchflow.environment.manifest import EnvironmentManifest
from benchflow.environment.manifest_env import ManifestEnvironment
from benchflow.environment.protocol import StateSnapshot
from benchflow.sandbox.protocol import ExecResult


class LocalSandbox:
    """Runs sandbox commands on the host, redirecting the snapshot directory."""

    def __init__(self, snapshot_root):
        self.snapshot_root = str(snapshot_root)

    async def exec(self, command, *, timeout_sec):
        command = command.replace("/tmp/benchflow-snapshots", self.snapshot_root)
        proc = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout_sec)
        return ExecResult(
            return_code=proc.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
        )


async def test_snapshot_roundtrip_preserves_same_named_databases(tmp_path):
    """Guards the snapshot collision fix."""
    if shutil.which("sqlite3") is None:
        pytest.skip("SQLite CLI is required to execute the sandbox backup commands")

    paths = [tmp_path / name / "state.db" for name in ("mail", "calendar")]
    for index, path in enumerate(paths):
        path.parent.mkdir()
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE state (value INTEGER)")
            db.execute("INSERT INTO state VALUES (?)", (index,))
    manifest = EnvironmentManifest.model_validate(
        {
            "name": "roundtrip",
            "image": "local-test",
            "state": {"kind": "sqlite", "paths": list(map(str, paths))},
        }
    )
    env = ManifestEnvironment(manifest, sandbox=LocalSandbox(tmp_path / "snapshots"))
    snap = await env.snapshot()
    for path in paths:
        with sqlite3.connect(path) as db:
            db.execute("UPDATE state SET value = 99")
    await env.restore(snap)
    for index, path in enumerate(paths):
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT value FROM state").fetchall() == [(index,)]


async def test_composed_restore_does_not_replay_later_container_wal(tmp_path):
    """Guards selective PR #1046 against WAL replay over an earlier online backup."""
    live = tmp_path / "live.db"
    backup = tmp_path / "backup.db"
    restored = tmp_path / "restored.db"
    with sqlite3.connect(live) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE state(v)")
        db.execute("INSERT INTO state VALUES(11)")
        db.commit()
        with sqlite3.connect(backup) as destination:
            db.backup(destination)
        # The service writes after environment capture, before container capture.
        db.execute("UPDATE state SET v=99")
        db.commit()
        shutil.copyfile(backup, restored)
        later_wal = (tmp_path / "live.db-wal").read_bytes()
    db.close()
    # Negative control: ordinary file replacement replays the later WAL.
    restored.with_name("restored.db-wal").write_bytes(later_wal)
    with sqlite3.connect(restored) as db:
        assert db.execute("SELECT v FROM state").fetchone() == (99,)
    db.close()
    restored.with_name("restored.db-wal").write_bytes(later_wal)
    restored.with_name("restored.db-shm").write_bytes(b"stale shared memory")
    env = ManifestEnvironment(
        EnvironmentManifest.model_validate(
            {
                "name": "wal",
                "image": "test",
                "state": {"kind": "sqlite", "paths": [str(restored)]},
            }
        ),
        sandbox=LocalSandbox(tmp_path / "snapshots"),
    )
    await env.restore_after_sandbox_restore(
        StateSnapshot(
            id="before-write",
            path=str(tmp_path),
            files={str(restored): "backup.db"},
        )
    )
    assert not restored.with_name("restored.db-wal").exists()
    assert not restored.with_name("restored.db-shm").exists()
    with sqlite3.connect(restored) as db:
        assert db.execute("SELECT v FROM state").fetchone() == (11,)


@pytest.mark.parametrize(
    "files",
    [
        {},
        {"/mail/state.db": "0.sqlite"},
        {"/mail/state.db": "0.sqlite", "/calendar/state.db": "0.sqlite"},
        {"/mail/state.db": "../0.sqlite", "/calendar/state.db": "1.sqlite"},
    ],
)
async def test_invalid_snapshot_mapping_is_rejected_before_exec(files):
    """Guards the snapshot collision fix against unsafe restoration."""

    class NoExec:
        async def exec(self, *args, **kwargs):
            pytest.fail("invalid snapshot must not modify the sandbox")

    manifest = EnvironmentManifest.model_validate(
        {
            "name": "roundtrip",
            "image": "local-test",
            "state": {
                "kind": "sqlite",
                "paths": ["/mail/state.db", "/calendar/state.db"],
            },
        }
    )
    env = ManifestEnvironment(manifest, sandbox=NoExec())
    with pytest.raises(ValueError, match="snapshot"):
        await env.restore(StateSnapshot(id="legacy", path="/tmp/snap", files=files))


async def test_unambiguous_legacy_snapshot_is_restorable():
    """Guards compatibility with snapshots created before the snapshot collision fix."""
    commands = []

    class RecordingSandbox:
        async def exec(self, command, **kwargs):
            commands.append(command)
            return ExecResult(return_code=0, stdout="", stderr="")

    manifest = EnvironmentManifest.model_validate(
        {
            "name": "legacy",
            "image": "local-test",
            "state": {
                "kind": "sqlite",
                "paths": ["/data/mail.db", "/data/calendar.db"],
            },
        }
    )
    env = ManifestEnvironment(manifest, sandbox=RecordingSandbox())
    await env.restore(StateSnapshot(id="legacy", path="/tmp/snap"))
    assert commands == [
        "cp /tmp/snap/mail.db /data/mail.db && cp /tmp/snap/calendar.db /data/calendar.db"
    ]
