"""Guards FF-sw-evaluation-C6 against be37b5e8 unbounded shell arguments.

Since DAYTONA-EXEC the call site sends one command of any size and the
backend keeps the transport bound (benchflow.sandbox._transport): the
local shell backend here stages what does not fit, and every argv it sends
stays within the bound."""

import shutil
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from benchflow.environment.protocol import StateSnapshot
from benchflow.sandbox._transport import COMMAND_BOUND
from tests.environment.test_sweep_snapshot import environment
from tests.environment.test_sweep_snapshot_b22r1 import ShellSandbox


class RecordingShell(ShellSandbox):
    def __init__(self, shell):
        super().__init__(shell)
        self.commands = []

    async def exec(self, command, *, timeout_sec):
        self.commands.append(command)
        return await super().exec(command, timeout_sec=timeout_sec)


@pytest.mark.parametrize("count", [500, 1800])
@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/bash"])
@pytest.mark.parametrize("method", ["restore", "restore_after_sandbox_restore"])
async def test_large_snapshot_roundtrip(count, shell, method):
    # Keep the review's 500-file capture below MAX_ARG_STRLEN before the fix;
    # the larger case also covers capture/copy commands and Unicode byte sizes.
    with TemporaryDirectory(prefix="c6-") as folder:
        root = Path(folder)
        seed = root / "seed"
        with sqlite3.connect(seed) as db:
            db.execute("create table marker (v int)")
            db.execute("insert into marker values (42)")
        names = [
            root / (f"{i:04}.db" if count == 500 else f"{i:04}-" + "é'" * 30 + ".db")
            for i in range(count)
        ]
        for name in names:
            shutil.copyfile(seed, name)
        env = environment([str(root / "*.db")])
        sandbox = env._sandbox = RecordingShell(shell)
        snap = await env.snapshot()
        assert set(snap.files) == {str(name) for name in names}
        for name in names:
            name.write_bytes(b"changed")
        extra = root / "new.db"
        extra.write_bytes(b"later file")
        await getattr(env, method)(snap)
        for name in names:
            with sqlite3.connect(name) as db:
                assert db.execute("select v from marker").fetchone() == (42,)
        assert extra.read_bytes() == b"later file"
        assert max(sandbox.sent_sizes) <= COMMAND_BOUND


@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/bash"])
@pytest.mark.parametrize("method", ["restore", "restore_after_sandbox_restore"])
@pytest.mark.parametrize("mismatch", ["path", "pattern"])
async def test_complete_coverage_precedes_all_restore_writes(
    tmp_path, shell, method, mismatch
):
    paths = {str(tmp_path / f"{i:04}.db"): f"{i}.sqlite" for i in range(500)}
    if mismatch == "path":
        paths[str(tmp_path / "outside.txt")] = "outside.sqlite"
    patterns = [str(tmp_path / "*.db")]
    if mismatch == "pattern":
        patterns.append(str(tmp_path / "absent*.sqlite"))
    # Every destination and sidecar carries a dummy marker. Missing backups
    # must never be consulted: a validation failure precedes even the first rm.
    for path in paths:
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(path + suffix).write_bytes(b"untouched")
    env = environment(patterns)
    env._sandbox = RecordingShell(shell)
    snap = StateSnapshot(id="dummy", path=str(tmp_path / "no-backups"), files=paths)
    with pytest.raises(ValueError, match="snapshot database paths"):
        await getattr(env, method)(snap)
    for path in paths:
        for suffix in ("", "-wal", "-shm", "-journal"):
            assert Path(path + suffix).read_bytes() == b"untouched"
    assert all(
        "cp " not in command and "rm " not in command
        for command in env._sandbox.commands
    )


@pytest.mark.parametrize("method", ["restore", "restore_after_sandbox_restore"])
@pytest.mark.parametrize("legacy", [False, True])
async def test_large_literal_and_legacy_restores(tmp_path, method, legacy):
    backups = tmp_path / "backups"
    backups.mkdir()
    paths = {}
    for i in range(900):
        name = f"{i:04}-" + "quote'é" * 12 + ".db"
        path = tmp_path / name
        path.write_bytes(b"changed")
        (backups / name).write_bytes(b"captured")
        paths[str(path)] = name
    env = environment(list(paths))
    env._sandbox = RecordingShell("/bin/sh")
    snap = StateSnapshot(id="dummy", path=str(backups), files={} if legacy else paths)
    await getattr(env, method)(snap)
    assert all(Path(path).read_bytes() == b"captured" for path in paths)
    assert max(env._sandbox.sent_sizes) <= COMMAND_BOUND


@pytest.mark.parametrize("response", ["", "p0\n", "g0\n", "p0\ng0\nunknown\n"])
async def test_incomplete_or_malformed_coverage_fails_closed(response):
    from unittest.mock import AsyncMock

    from benchflow.sandbox.protocol import ExecResult

    env = environment(["/tmp/c6-state/*.db"])
    env._sandbox.exec = AsyncMock(
        return_value=ExecResult(return_code=0, stdout=response, stderr="")
    )
    snap = StateSnapshot(
        id="dummy", path="/tmp/c6-backups", files={"/tmp/c6-state/a.db": "0"}
    )
    with pytest.raises(ValueError, match="snapshot database paths"):
        await env.restore(snap)
    assert env._sandbox.exec.await_count == 1


@pytest.mark.parametrize("method", ["restore", "restore_after_sandbox_restore"])
@pytest.mark.parametrize("covered", [False, True])
async def test_many_patterns_require_complete_coverage(tmp_path, method, covered):
    path = tmp_path / "a.db"
    path.write_bytes(b"changed")
    (tmp_path / "backup").write_bytes(b"captured")
    patterns = [str(tmp_path / "*.db")] * 600
    if not covered:
        patterns.append(str(tmp_path / "missing*.db"))
    env = environment(patterns)
    env._sandbox = RecordingShell("/bin/sh")
    snap = StateSnapshot(id="dummy", path=str(tmp_path), files={str(path): "backup"})
    if covered:
        await getattr(env, method)(snap)
        assert path.read_bytes() == b"captured"
    else:
        with pytest.raises(ValueError, match="snapshot database paths"):
            await getattr(env, method)(snap)
        assert path.read_bytes() == b"changed"
    assert max(env._sandbox.sent_sizes) <= COMMAND_BOUND
