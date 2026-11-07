"""The wait between a Daytona exec's status requests (PY8 U5b).

``DaytonaSandbox._poll_response`` asks Daytona whether a session command has
finished until it has. It used to wait a fixed 1.0 s between two requests, so
an exec that finished in 50 ms still took a second, and a trial makes dozens
of execs one after another. The waits now start at 0.05 s and double up to
1.0 s. These tests drive the poll loop on a fake clock: ``loop.time()`` is a
number the fake ``asyncio.sleep`` advances by exactly the delay asked for, so
every wait the loop makes is recorded and the deadline arithmetic is exact.
Exit codes, captured output and the two timeout errors stay as they were;
``tests/test_daytona_poll_deadline.py`` and
``tests/test_daytona_command_polling.py`` keep checking those unchanged.
"""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest

from benchflow.sandbox import daytona as daytona_module
from benchflow.sandbox.daytona import DaytonaSandbox

SCHEDULE = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0]


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.waits: list[float] = []


def _fake_clock(monkeypatch) -> _Clock:
    """Call inside the test, so the patched loop is the one the test runs on."""
    clock = _Clock()
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "time", lambda: clock.now)

    async def fake_sleep(delay):
        clock.waits.append(float(delay))
        clock.now += float(delay)

    monkeypatch.setattr(daytona_module.asyncio, "sleep", fake_sleep)
    return clock


class _TimedProcess:
    """A ``sandbox.process`` whose command finishes ``runs_for`` seconds of
    fake time after the poll starts (``None``: never), with exit code
    ``exit_code`` and the given output. It counts the status and log
    requests the poll makes."""

    def __init__(
        self,
        clock: _Clock,
        runs_for: float | None,
        exit_code: int = 0,
        stdout: str = "out",
        stderr: str = "err",
    ) -> None:
        self.clock = clock
        self.done_at = None if runs_for is None else clock.now + runs_for
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.status_requests = 0
        self.log_requests = 0

    async def get_session_command(self, session_id, command_id):
        self.status_requests += 1
        done = self.done_at is not None and self.clock.now >= self.done_at
        return SimpleNamespace(
            id=command_id, exit_code=self.exit_code if done else None
        )

    async def get_session_command_logs(self, session_id, command_id):
        self.log_requests += 1
        return SimpleNamespace(stdout=self.stdout, stderr=self.stderr)


def _sandbox(process: _TimedProcess) -> DaytonaSandbox:
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox._sandbox = SimpleNamespace(process=process)
    return sandbox


async def _poll(process: _TimedProcess, timeout_sec: float | None = 900):
    return await asyncio.wait_for(
        _sandbox(process)._poll_response("session", "cmd", timeout_sec=timeout_sec),
        timeout=30,
    )


async def test_waits_start_at_50_ms_and_double_up_to_one_second(monkeypatch) -> None:
    """The schedule itself: 0.05, 0.1, 0.2, 0.4, 0.8, then 1.0 s each."""
    clock = _fake_clock(monkeypatch)
    process = _TimedProcess(clock, runs_for=4.5)

    result = await _poll(process)

    assert clock.waits == SCHEDULE
    assert sum(clock.waits) == pytest.approx(4.55)
    assert process.status_requests == len(SCHEDULE) + 1
    assert process.log_requests == 1
    assert (result.return_code, result.stdout, result.stderr) == (0, "out", "err")


async def test_a_quick_command_waits_50_ms_not_a_second(monkeypatch) -> None:
    """A mkdir or a ``test -f`` is done by the second request, 50 ms in;
    the fixed wait held it for a whole second."""
    clock = _fake_clock(monkeypatch)
    process = _TimedProcess(clock, runs_for=0.03)

    await _poll(process)

    assert clock.waits == [0.05]
    assert process.status_requests == 2


async def test_a_command_done_at_the_first_request_waits_nothing(monkeypatch) -> None:
    clock = _fake_clock(monkeypatch)
    process = _TimedProcess(clock, runs_for=0.0, exit_code=3, stdout="", stderr="no")

    result = await _poll(process)

    assert clock.waits == []
    assert process.status_requests == 1
    assert (result.return_code, result.stdout, result.stderr) == (3, "", "no")


async def test_each_exec_starts_the_schedule_again(monkeypatch) -> None:
    clock = _fake_clock(monkeypatch)
    first = _TimedProcess(clock, runs_for=3.0)
    await _poll(first)
    clock.waits.clear()

    second = _TimedProcess(clock, runs_for=0.3)
    await _poll(second)

    assert clock.waits == [0.05, 0.1, 0.2]


async def test_the_last_wait_stops_at_the_deadline(monkeypatch) -> None:
    """An explicit timeout ends the poll exactly when it ends, with the
    same error as before; the last wait is cut to the time left."""
    clock = _fake_clock(monkeypatch)
    process = _TimedProcess(clock, runs_for=None)

    with pytest.raises(RuntimeError) as excinfo:
        await _poll(process, timeout_sec=1)

    assert str(excinfo.value) == "Command timed out after 1 seconds"
    assert clock.waits[:4] == [0.05, 0.1, 0.2, 0.4]
    assert clock.waits[4] == pytest.approx(0.25)
    assert sum(clock.waits) == pytest.approx(1.0)
    assert process.log_requests == 0


async def test_the_safety_net_cap_still_ends_a_poll_without_a_timeout(
    monkeypatch,
) -> None:
    clock = _fake_clock(monkeypatch)
    monkeypatch.setattr(daytona_module, "_DAYTONA_EXEC_HARD_CAP_SEC", 4)
    process = _TimedProcess(clock, runs_for=None)

    with pytest.raises(RuntimeError) as excinfo:
        await _poll(process, timeout_sec=None)

    assert "Command timed out after 4 seconds" in str(excinfo.value)
    assert "safety-net cap" in str(excinfo.value)
    assert sum(clock.waits) == pytest.approx(4.0)


def _old_status_requests(runs_for: float) -> int:
    """Status requests under the fixed 1.0 s wait: one at the start, then one
    a second until the command is done."""
    return 1 + math.ceil(runs_for)


@pytest.mark.parametrize(
    ("runs_for", "requests", "waited"),
    [
        (0.03, 2, 0.05),
        (0.3, 4, 0.35),
        (0.76, 6, 1.55),
        (1.0, 6, 1.55),
        (1.6, 7, 2.55),
        (2.0, 7, 2.55),
        (60.0, 65, 60.55),
        (600.0, 605, 600.55),
    ],
)
async def test_requests_and_time_waited_by_command_length(
    monkeypatch, runs_for, requests, waited
) -> None:
    """How long an exec waits after its command is done, and how many status
    requests it makes, for commands of a few lengths. Under the fixed wait
    every command waited until the next whole second after it started; now
    a command waits at most its own length plus 0.05 s past its end, and
    never more than a second, as before. It makes at most four status
    requests more than before; after 1.55 s it asks once a second, the old
    rate."""
    clock = _fake_clock(monkeypatch)
    process = _TimedProcess(clock, runs_for=runs_for)

    await _poll(process)

    assert process.status_requests == requests
    assert sum(clock.waits) == pytest.approx(waited)
    after_done = sum(clock.waits) - runs_for
    assert after_done <= runs_for + 0.05 + 1e-9
    assert after_done < 1.0
    assert requests - _old_status_requests(runs_for) <= 4


class _SlowStatusProcess(_TimedProcess):
    """Each status request answers as the command was when it was asked,
    ``latency`` seconds of fake time before the answer arrives."""

    def __init__(self, clock: _Clock, runs_for: float, latency: float, **kw) -> None:
        super().__init__(clock, runs_for, **kw)
        self.latency = latency

    async def get_session_command(self, session_id, command_id):
        response = await super().get_session_command(session_id, command_id)
        self.clock.now += self.latency
        return response


@pytest.mark.parametrize(
    ("runs_for", "timeout_sec", "latency", "exit_code", "stdout"),
    [
        # Status answers take 100 ms; the command ends
        # 0.82 s before its 3 s budget.
        (2.18, 3, 0.1, 0, "done"),
        # Answers take 0.6 s; the command ends with its
        # own exit 124 at 1.5 s of a 2 s budget.
        (1.5, 2, 0.6, 124, "early"),
    ],
)
async def test_a_command_done_within_its_budget_is_never_a_timeout(
    monkeypatch, runs_for, timeout_sec, latency, exit_code, stdout
) -> None:
    """The last pending answer was asked for before the last wait; the
    poll used to raise at the deadline on it without asking again, so a
    finished command (and its reward file) became a timeout."""
    clock = _fake_clock(monkeypatch)
    process = _SlowStatusProcess(
        clock, runs_for, latency, exit_code=exit_code, stdout=stdout
    )

    result = await _poll(process, timeout_sec=timeout_sec)

    assert (result.return_code, result.stdout) == (exit_code, stdout)
    assert process.log_requests == 1


async def test_a_command_still_running_at_the_deadline_times_out_after_one_more_ask(
    monkeypatch,
) -> None:
    clock = _fake_clock(monkeypatch)
    process = _SlowStatusProcess(clock, runs_for=10.0, latency=0.1)

    with pytest.raises(RuntimeError) as excinfo:
        await _poll(process, timeout_sec=3)

    assert str(excinfo.value) == "Command timed out after 3 seconds"
    assert process.log_requests == 0
    assert clock.now - 1000.0 < 3 + 2 * 0.1 + 1e-9
