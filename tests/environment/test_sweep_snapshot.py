"""Guards FF-sw-evaluation snapshot/readiness fixes at 363d815b.

The shell adapter executes real commands only in the builder's labeled container.
"""

import asyncio
import sqlite3

import pytest

from benchflow.environment.manifest import (
    EnvironmentManifest,
    StateSpec,
)
from benchflow.environment.manifest_env import (
    EnvironmentSnapshotError,
    ManifestEnvironment,
)
from benchflow.sandbox.protocol import ExecResult


class ShellSandbox:
    async def exec(self, command, *, timeout_sec=120):
        proc = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout_sec)
        return ExecResult(
            return_code=proc.returncode, stdout=stdout.decode(), stderr=stderr.decode()
        )


def environment(paths):
    return ManifestEnvironment(
        EnvironmentManifest(
            name="fixture", image="fixture", state=StateSpec(paths=paths)
        ),
        sandbox=ShellSandbox(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pattern", ["*.db", "mail?.db", "mail[12].db", "*/*.db", "space dir/*.db"]
)
async def test_sqlite_globs_capture_and_restore_exact_existing_set(tmp_path, pattern):
    folder = tmp_path / (
        "space dir"
        if pattern == "space dir/*.db"
        else "nested"
        if "/" in pattern
        else ""
    )
    folder.mkdir(exist_ok=True)
    paths = [folder / "mail1.db", folder / "mail2.db"]
    for path in paths:
        with sqlite3.connect(path) as db:
            db.execute("create table marker (value int)")
            db.execute("insert into marker values (42)")
    env = environment([str(tmp_path / pattern)])
    snap = await env.snapshot()
    assert set(snap.files) == {str(path) for path in paths}
    for path in paths:
        with sqlite3.connect(path) as db:
            db.execute("update marker set value=0")
    extra = folder / "mail3.db"
    extra.write_bytes(b"created after snapshot")
    await env.restore(snap)
    for path in paths:
        with sqlite3.connect(path) as db:
            assert db.execute("select value from marker").fetchone() == (42,)
    assert extra.read_bytes() == b"created after snapshot"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["missing.db", "*.absent", "directory.db"])
async def test_missing_database_never_becomes_empty_snapshot(tmp_path, name):
    if name == "directory.db":
        (tmp_path / name).mkdir()
    env = environment([str(tmp_path / name)])
    with pytest.raises(EnvironmentSnapshotError):
        await env.snapshot()
    if name != "directory.db":
        assert not (tmp_path / name).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", ["space name.db", "quote'\".db", "line\nbreak.db", "$(touch UNEXPECTED).db"]
)
async def test_snapshot_paths_are_data_and_overlapping_patterns_deduplicate(
    tmp_path, name
):
    path = tmp_path / name
    with sqlite3.connect(path) as db:
        db.execute("create table marker (value int)")
        db.execute("insert into marker values (42)")
    env = environment([str(tmp_path / "*.db"), str(tmp_path / "*.db")])
    snap = await env.snapshot()
    assert list(snap.files) == [str(path)]
    await env.restore(snap)
    with sqlite3.connect(path) as db:
        assert db.execute("select value from marker").fetchone() == (42,)


@pytest.mark.asyncio
async def test_legacy_glob_snapshot_without_resolved_paths_is_refused():
    from unittest.mock import AsyncMock

    from benchflow.environment.protocol import StateSnapshot

    env = environment(["/tmp/state-fixture/*.db"])
    env._sandbox = type(
        "Sandbox",
        (),
        {
            "exec": AsyncMock(
                return_value=ExecResult(return_code=0, stdout="", stderr="")
            )
        },
    )()
    with pytest.raises(ValueError, match="captured database paths"):
        await env.restore(StateSnapshot(id="legacy", path="/tmp/legacy-snapshot"))
    env._sandbox.exec.assert_not_awaited()
