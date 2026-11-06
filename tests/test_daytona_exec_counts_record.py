"""``daytona-execs.json``: what a Daytona sandbox's execs cost (PY8 U5b).

When a Daytona sandbox stops, it writes in the trial folder how many execs it
ran, how many status requests their polls made and how long they took, so a
run can be compared by its round trips (RM's B4f reads it from each trial).
Transient execs (session deleted afterward: the live token counter's log
reads, which repeat for the whole agent run, and short probes) are counted in
their own ``transient_`` fields, so the first three cover the other execs.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox.daytona_dind import VM_ROUTE

RECORD = "daytona-execs.json"


class _Process:
    """Each command is done at its second status request."""

    def __init__(self) -> None:
        self.asked: dict[str, int] = {}

    async def create_session(self, session_id):
        pass

    async def execute_session_command(self, session_id, request, timeout=None):
        return SimpleNamespace(cmd_id=session_id)

    async def get_session_command(self, session_id, command_id):
        self.asked[command_id] = self.asked.get(command_id, 0) + 1
        done = self.asked[command_id] >= 2
        return SimpleNamespace(id=command_id, exit_code=0 if done else None)

    async def get_session_command_logs(self, session_id, command_id):
        return SimpleNamespace(stdout="ok", stderr="")

    async def delete_session(self, session_id):
        pass


class _Strategy:
    def __init__(self, sandbox) -> None:
        from benchflow.sandbox.daytona_strategies import _DaytonaDirect

        self._direct = _DaytonaDirect(sandbox)
        self.stopped = 0

    async def exec(self, *args, **kwargs):
        return await self._direct.exec(*args, **kwargs)

    async def exec_transient(self, *args, **kwargs):
        return await self._direct.exec_transient(*args, **kwargs)

    async def stop(self, delete):
        self.stopped += 1


class _ComposeStrategy(_Strategy):
    """Runs one exec while it stops, like a compose task's ``compose down``."""

    async def stop(self, delete):
        # As the DinD VM's own command (daytona_dind.VM_ROUTE).
        await self._direct.exec("compose down", timeout_sec=5, route=VM_ROUTE)
        self.stopped += 1


def _sandbox(rollout_dir: Path | None, strategy=_Strategy):
    pytest.importorskip("daytona")
    from benchflow.sandbox.daytona import DaytonaSandbox, _load_daytona_sdk

    _load_daytona_sdk()
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox._sandbox = SimpleNamespace(process=_Process())
    sandbox._strategy = strategy(sandbox)
    sandbox.rollout_paths = (
        SimpleNamespace(rollout_dir=rollout_dir) if rollout_dir is not None else None
    )
    return sandbox


async def test_stop_writes_the_exec_counts_in_the_trial_folder(
    tmp_path, monkeypatch
) -> None:
    from benchflow.sandbox import daytona as daytona_module

    monkeypatch.setattr(
        daytona_module, "_DAYTONA_COMMAND_POLL_FIRST_SEC", 0.001, raising=False
    )
    sandbox = _sandbox(tmp_path)

    for command in ("true", "mkdir -p /app", "id -u"):
        assert (await sandbox.exec(command, timeout_sec=5)).return_code == 0
    await sandbox.stop(delete=True)

    record = json.loads((tmp_path / RECORD).read_text())
    assert record["exec_calls"] == 3
    assert record["status_requests"] == 6
    assert 0 < record["exec_seconds"] < 5
    assert sandbox._strategy.stopped == 1


@pytest.fixture
def fast_poll(monkeypatch):
    from benchflow.sandbox import daytona as daytona_module

    monkeypatch.setattr(
        daytona_module, "_DAYTONA_COMMAND_POLL_FIRST_SEC", 0.001, raising=False
    )


async def test_transient_execs_are_counted_apart(tmp_path, fast_poll) -> None:
    """Guards U5b-R1-B-1: the live token counter's reads (transient execs)
    repeat for the whole agent run, faster under the new poll, so they must
    not change the ordinary exec count RM compares between arms."""
    sandbox = _sandbox(tmp_path)

    assert (await sandbox.exec("mkdir -p /app", timeout_sec=5)).return_code == 0
    for _ in range(4):
        result = await sandbox.exec_transient("stat log", timeout_sec=5)
        assert result.return_code == 0
    await sandbox.stop(delete=True)

    record = json.loads((tmp_path / RECORD).read_text())
    assert record["exec_calls"] == 1
    assert record["status_requests"] == 2
    assert record["transient_exec_calls"] == 4
    assert record["transient_status_requests"] == 8
    assert 0 < record["exec_seconds"] < 5
    assert 0 < record["transient_exec_seconds"] < 5


async def test_a_transient_read_alongside_an_exec_is_counted_apart(
    tmp_path, fast_poll
) -> None:
    """The live token counter reads in its own task while the trial's other
    execs run: the ordinary exec polled while a transient read is mid-poll
    still counts as ordinary."""
    sandbox = _sandbox(tmp_path)
    process = sandbox._sandbox.process
    read_polling = asyncio.Event()
    release_read = asyncio.Event()
    real_status = process.get_session_command

    async def execute(session_id, request, timeout=None):
        return SimpleNamespace(
            cmd_id="read" if "stat" in request.command else "foreground"
        )

    async def status(session_id, command_id):
        # The read's first status request holds until the ordinary exec is
        # done, so the ordinary exec polls while the read is being polled.
        if command_id == "read" and not read_polling.is_set():
            read_polling.set()
            await release_read.wait()
        return await real_status(session_id, command_id)

    process.execute_session_command = execute
    process.get_session_command = status

    read = asyncio.create_task(sandbox.exec_transient("stat log", timeout_sec=5))
    await read_polling.wait()
    assert (await sandbox.exec("build", timeout_sec=5)).return_code == 0
    release_read.set()
    assert (await read).return_code == 0
    await sandbox.stop(delete=True)

    record = json.loads((tmp_path / RECORD).read_text())
    assert (record["exec_calls"], record["status_requests"]) == (1, 2)
    assert (
        record["transient_exec_calls"],
        record["transient_status_requests"],
    ) == (1, 2)


async def test_the_execs_of_stopping_are_counted(tmp_path, fast_poll) -> None:
    """Guards U5b-R1-A-2: a compose task runs ``compose down`` (an exec)
    while it stops; the record is written after that."""
    sandbox = _sandbox(tmp_path, _ComposeStrategy)

    await sandbox.exec("true", timeout_sec=5)
    await sandbox.stop(delete=True)

    record = json.loads((tmp_path / RECORD).read_text())
    assert record["exec_calls"] == 2
    assert sandbox._strategy.stopped == 1


async def test_a_sandbox_outside_a_trial_writes_nothing(tmp_path) -> None:
    sandbox = _sandbox(None)

    await sandbox.exec("true", timeout_sec=5)
    await sandbox.stop(delete=True)

    assert list(tmp_path.iterdir()) == []
    assert sandbox._strategy.stopped == 1


async def test_a_failed_write_does_not_stop_the_stop(tmp_path) -> None:
    sandbox = _sandbox(tmp_path / "missing-folder")

    await sandbox.exec("true", timeout_sec=5)
    await sandbox.stop(delete=True)

    assert sandbox._strategy.stopped == 1
