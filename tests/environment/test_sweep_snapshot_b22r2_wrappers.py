"""Guards C6 against provider quoting expansion at be37b5e8 and 724db804.

Since DAYTONA-EXEC the backend, not the call site, keeps the command line
bounded (benchflow.sandbox._transport): this shell wraps a command the way
Daytona's named-user execs do and stages what would not fit, so no argument
it starts ever passes Linux's limit, however large the snapshot's command."""

import asyncio
import shlex
import shutil
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from benchflow.sandbox._transport import COMMAND_BOUND
from benchflow.sandbox.protocol import ExecResult
from tests.environment.test_sweep_snapshot import environment


class WrappedShell:
    """Replay the nested command shells used by named-user provider execs."""

    def __init__(self):
        self.payload_sizes = []
        self.argument_sizes = []

    async def exec(self, command, *, timeout_sec):
        from benchflow.sandbox._transport import fit, sent

        self.payload_sizes.append(len(command.encode()))

        # Daytona quotes the inner command shell, then the named-user shell.
        # Replay both layers without su: the fixture already runs unprivileged.
        def wrapped(inner):
            for _ in range(2):
                inner = "/bin/sh -c " + shlex.quote(inner)
            return inner

        fitted = fit(wrapped, command, cat="/bin/cat")
        sent("test wrapped shell", fitted.sent)
        self.argument_sizes.append(len(fitted.sent.encode()))
        proc = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            fitted.sent,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["restore", "restore_after_sandbox_restore"])
async def test_snapshot_commands_leave_room_for_provider_shells(method):
    with TemporaryDirectory(prefix="c6-") as folder:
        root = Path(folder)
        seed = root / "seed"
        with sqlite3.connect(seed) as db:
            db.execute("create table marker (v int)")
            db.execute("insert into marker values (42)")
        names = [root / (f"{i:04}-" + "é'" * 30 + ".db") for i in range(500)]
        for name in names:
            shutil.copyfile(seed, name)
        env = environment([str(root / "*.db")])
        env._sandbox = shell = WrappedShell()
        snap = await env.snapshot()
        for name in names:
            name.write_bytes(b"changed")
        await getattr(env, method)(snap)
        for name in names:
            with sqlite3.connect(name) as db:
                assert db.execute("select v from marker").fetchone() == (42,)
        # The snapshot's own command is one, of any size; what is sent is not.
        assert max(shell.payload_sizes) > COMMAND_BOUND
        assert max(shell.argument_sizes) <= COMMAND_BOUND
