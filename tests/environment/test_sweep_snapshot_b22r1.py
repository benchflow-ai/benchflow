"""Guards FF-sw-evaluation-C2 against d97bf9f6 shell glob validation drift."""

import asyncio
import sqlite3

import pytest

from benchflow.environment.protocol import StateSnapshot
from benchflow.sandbox.protocol import ExecResult
from tests.environment.test_sweep_snapshot import environment


class ShellSandbox:
    """A local shell backend that keeps the transport contract: a command
    over the bound goes on stdin and the reader runs it
    (benchflow.sandbox._transport)."""

    def __init__(self, shell):
        self.shell = shell
        self.sent_sizes = []

    async def exec(self, command, *, timeout_sec):
        from benchflow.sandbox._transport import fit, sent, size

        fitted = fit(lambda c: [self.shell, "-c", c], command, cat="/bin/cat")
        sent("test shell", fitted.sent)
        self.sent_sizes.append(size(fitted.sent))
        proc = await asyncio.create_subprocess_exec(
            *fitted.sent,
            stdin=(
                asyncio.subprocess.DEVNULL
                if fitted.payload is None
                else asyncio.subprocess.PIPE
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(fitted.payload), timeout_sec
        )
        return ExecResult(
            return_code=proc.returncode, stdout=stdout.decode(), stderr=stderr.decode()
        )


@pytest.mark.parametrize(
    "pattern,name",
    [
        ("[[:digit:]].db", "1.db"),
        ("[[:alpha:]].db", "a.db"),
        ("[[:alnum:]].db", "7.db"),
        ("[[:lower:]].db", "z.db"),
        ("[[:upper:]].db", "Z.db"),
        ("[[:xdigit:]].db", "F.db"),
        ("[[:blank:]].db", " .db"),
        ("[[:space:]].db", "\t.db"),
        ("[[:cntrl:]].db", "\t.db"),
        ("[[:punct:]].db", "'.db"),
        ("[[:graph:]].db", "!.db"),
        ("[[:print:]].db", " .db"),
        ("[![:digit:]].db", "a.db"),
        ("[[:digit:]a].db", "a.db"),
        ("[[]x].db", "[x].db"),
        (".*.db", ".hidden.db"),
        (r"\.*.db", ".hidden.db"),
        ("[a-z]/[[:digit:]].db", "q/2.db"),
        ("[*].db", "*.db"),
        ("?.db", "a.db"),
        ("[abc.db", "[abc.db"),
        ("quote'*/line\n*.db", "quote'dir/line\nfile.db"),
        (r"back\\*.db", r"back\file.db"),
    ],
)
@pytest.mark.parametrize("restore_method", ["restore", "restore_after_sandbox_restore"])
@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/bash"])
async def test_shell_glob_snapshot_restores_captured_set(
    tmp_path, pattern, name, restore_method, shell
):
    path = tmp_path / name
    path.parent.mkdir(exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute("create table marker (v int)")
        db.execute("insert into marker values (42)")
    env = environment([str(tmp_path / pattern)])
    env._sandbox = ShellSandbox(shell)
    snap = await env.snapshot()
    assert set(snap.files) == {str(path)}
    with sqlite3.connect(path) as db:
        db.execute("update marker set v=0")
    # No restore-time discovery: a newly matching database is left untouched.
    extra = tmp_path / "9.db"
    extra.write_bytes(b"new database marker")
    await getattr(env, restore_method)(snap)
    with sqlite3.connect(path) as db:
        assert db.execute("select v from marker").fetchone() == (42,)
    assert extra.read_bytes() == b"new database marker"


@pytest.mark.parametrize(
    "pattern,path",
    [
        ("*.db", "nested/1.db"),
        ("*.db", ".hidden.db"),
        ("[.]*.db", ".hidden.db"),
        ("[[:digit:]].db", "a.db"),
        ("[![:digit:]].db", "1.db"),
        ("?.db", "ab.db"),
    ],
)
@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/bash"])
async def test_restore_rejects_paths_outside_shell_selection(
    tmp_path, pattern, path, shell
):
    env = environment([str(tmp_path / pattern)])
    env._sandbox = ShellSandbox(shell)
    snap = StateSnapshot(
        id="dummy", path=str(tmp_path), files={str(tmp_path / path): "0.sqlite"}
    )
    with pytest.raises(ValueError, match="snapshot database paths"):
        await env.restore(snap)
