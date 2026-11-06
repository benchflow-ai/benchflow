"""Guards C7 against d1efc9ab rejecting captured paths by component count."""

import asyncio
import shlex
import sqlite3
from pathlib import Path

import pytest

from benchflow.environment.manifest import EnvironmentManifest, StateSpec
from benchflow.environment.manifest_env import ManifestEnvironment, _shell_path_match
from benchflow.environment.protocol import StateSnapshot
from benchflow.sandbox.protocol import ExecResult


class Shell:
    def __init__(self, transport):
        self.transport = transport
        self.payload_sizes = []
        self.argument_sizes = []

    async def exec(self, command, *, timeout_sec):
        self.payload_sizes.append(len(command.encode()))
        if self.transport == "wrapped":
            for _ in range(2):
                command = "/bin/sh -c " + shlex.quote(command)
        self.argument_sizes.append(len(command.encode()))
        proc = await asyncio.create_subprocess_exec(
            "/bin/sh" if self.transport == "wrapped" else self.transport,
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout_sec)
        return ExecResult(
            return_code=proc.returncode, stdout=out.decode(), stderr=err.decode()
        )


def environment(paths, transport):
    return ManifestEnvironment(
        EnvironmentManifest(
            name="fixture", image="fixture", state=StateSpec(paths=paths)
        ),
        sandbox=Shell(transport),
    )


@pytest.mark.parametrize("transport", ["/bin/sh", "/bin/bash", "wrapped"])
@pytest.mark.parametrize("depth", [0, 1, 150, 500, 1500])
async def test_matcher_syntax_size_is_independent_of_depth(transport, depth):
    # Exercise depth beyond SQLite's own pathname limit without asking SQLite
    # to open these paths. The shell matcher neither expands nor opens them.
    path, pattern = "/data/" + "a/" * depth + "x.db", "/data/" + "*/" * depth + "*.db"
    command = _shell_path_match(path, pattern)
    assert (
        len(command.encode())
        <= len(shlex.quote(path).encode()) + len(shlex.quote(pattern).encode()) + 512
    )
    assert len(command.encode()) <= 8192
    assert (await Shell(transport).exec(command, timeout_sec=120)).return_code == 0


@pytest.mark.parametrize("transport", ["/bin/sh", "/bin/bash", "wrapped"])
@pytest.mark.parametrize("method", ["restore", "restore_after_sandbox_restore"])
@pytest.mark.parametrize(
    "depth,component,pattern",
    [
        (150, "a", "*"),
        (200, "a", "?"),
        (100, "q'\n", "[[:alpha:]]*"),
        (80, "é\\'", "*"),
    ],
)
async def test_deep_snapshot_roundtrip(
    tmp_path, transport, method, depth, component, pattern
):
    path = tmp_path.joinpath(*([component] * depth), "x.db")
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as db:
        db.execute("create table marker (v int)")
        db.execute("insert into marker values (42)")
    env = environment(
        [str(tmp_path) + "/" + (pattern + "/") * depth + "*.db"], transport
    )
    sandbox = env._sandbox
    snap = await env.snapshot()
    assert set(snap.files) == {str(path)}
    with sqlite3.connect(path) as db:
        db.execute("update marker set v=0")
    extra = path.with_name("new.db")
    extra.write_bytes(b"not captured")
    await getattr(env, method)(snap)
    with sqlite3.connect(path) as db:
        assert db.execute("select v from marker").fetchone() == (42,)
    assert extra.read_bytes() == b"not captured"
    assert max(sandbox.payload_sizes) <= 8192
    assert max(sandbox.argument_sizes) < 131072


@pytest.mark.parametrize("transport", ["/bin/sh", "/bin/bash", "wrapped"])
@pytest.mark.parametrize("method", ["restore", "restore_after_sandbox_restore"])
@pytest.mark.parametrize("position", [0, 75, 149])
@pytest.mark.parametrize("bad_component", ["b", ".a", "a/extra"])
async def test_every_deep_component_is_checked_before_writes(
    tmp_path, transport, method, position, bad_component
):
    components = ["a"] * 150
    components[position] = bad_component
    invalid = tmp_path.joinpath(*components, "x.db")
    valid = tmp_path / "valid.db"
    for path in (valid, invalid):
        path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(str(path) + suffix).write_bytes(b"untouched")
    env = environment(
        [str(valid), str(tmp_path) + "/" + "[a]/" * 150 + "*.db"], transport
    )
    snap = StateSnapshot(
        id="dummy",
        path=str(tmp_path / "absent-backups"),
        files={str(valid): "0.sqlite", str(invalid): "1.sqlite"},
    )
    with pytest.raises(ValueError, match="snapshot database paths"):
        await getattr(env, method)(snap)
    for path in (valid, invalid):
        for suffix in ("", "-wal", "-shm", "-journal"):
            assert Path(str(path) + suffix).read_bytes() == b"untouched"
