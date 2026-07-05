"""BF-6 regression tests for ``DaytonaSandbox._poll_response`` deadlines.

A Daytona *session* command only reports its ``exit_code`` once it completes,
and Daytona treats the command as still-running while any child holds the
session's stdout/stderr stream open. Before BF-6, ``_poll_response`` looped
``while response.exit_code is None`` with ``deadline=None`` whenever the caller
passed no ``timeout_sec`` — so a backgrounded daemon that never released the
stream wedged ``exec`` forever. The fix applies a generous safety-net cap
(``_DAYTONA_EXEC_HARD_CAP_SEC``) on the no-``timeout_sec`` path so the loop can
never spin indefinitely, while leaving the explicit-``timeout_sec`` path
byte-for-byte unchanged.

These tests fake the SDK poll boundary directly (no real Daytona / SDK / creds —
the ``daytona`` extra is absent in the fork venv), mirroring the faking style of
``tests/test_daytona_command_polling.py`` and ``tests/test_daytona_list.py``.
``_poll_response`` touches no SDK types, so the sandbox is built via ``__new__``
with a hand-rolled fake ``process`` and the method is exercised directly.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from benchflow.sandbox import daytona as daytona_module
from benchflow.sandbox.daytona import DaytonaSandbox


class _FakeDaytonaError(Exception):
    pass


_FakeDaytonaError.__module__ = "daytona.common.errors"


def _install_fake_clock(monkeypatch) -> dict[str, float]:
    """Make the poll loop time-deterministic, independent of wall clock.

    ``_poll_response`` measures deadlines with ``loop.time()`` and waits with
    ``asyncio.sleep``. Under a large parallel/loaded suite the real event loop
    can be starved, making wall-clock-based timing flaky. We instead drive a
    fake monotonic clock that advances by exactly the requested delay on every
    ``asyncio.sleep`` call (which becomes a no-op coroutine), mirroring the
    ``fake_sleep`` pattern in ``tests/test_sandbox.py``. The deadline arithmetic
    in ``_poll_response`` is then exercised exactly, but deterministically.
    """
    clock = {"now": 1000.0}

    loop = asyncio.get_event_loop()
    monkeypatch.setattr(loop, "time", lambda: clock["now"])

    async def fake_sleep(delay):
        clock["now"] += float(delay)

    monkeypatch.setattr(daytona_module.asyncio, "sleep", fake_sleep)
    return clock


class _FakeProcess:
    """Fake Daytona ``sandbox.process`` for the poll boundary.

    ``exit_code_sequence`` is consumed one entry per ``get_session_command``
    call; once exhausted it repeats its last value forever. ``None`` models a
    still-running session command (the wedge case); an int models completion.
    """

    def __init__(self, exit_code_sequence: list[int | None]) -> None:
        self._sequence = list(exit_code_sequence)
        self.get_command_calls = 0
        self.logs_fetched = False

    async def get_session_command(self, session_id, command_id):
        self.get_command_calls += 1
        # Consume one entry per call; once exhausted, default to ``None`` (a
        # never-completing session command — the wedge case).
        exit_code = self._sequence.pop(0) if self._sequence else None
        return SimpleNamespace(id="cmd-1", exit_code=exit_code)

    async def get_session_command_logs(self, session_id, command_id):
        self.logs_fetched = True
        return SimpleNamespace(stdout="out", stderr="err")


def _make_sandbox(process: _FakeProcess) -> DaytonaSandbox:
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox._sandbox = SimpleNamespace(process=process)
    return sandbox


def test_daytona_empty_exit_code_parse_error_is_retryable() -> None:
    exc = _FakeDaytonaError(
        "Failed to get session command: failed to convert exit code to int: "
        'strconv.Atoi: parsing "": invalid syntax'
    )

    assert daytona_module._is_daytona_transient_retry_error(exc)


class TestPollDeadline:
    @pytest.mark.asyncio
    async def test_none_timeout_hits_hard_cap_instead_of_looping_forever(
        self, monkeypatch
    ) -> None:
        """BF-6: ``timeout_sec=None`` + never-resolving exit_code now raises.

        The hard cap is monkeypatched tiny so the test is fast; without the fix
        this poll loop would never terminate.
        """
        _install_fake_clock(monkeypatch)
        # A small cap so the fake clock crosses it after a few poll iterations.
        monkeypatch.setattr(daytona_module, "_DAYTONA_EXEC_HARD_CAP_SEC", 5)
        monkeypatch.setattr(daytona_module, "_DAYTONA_COMMAND_POLL_INTERVAL_SEC", 1.0)

        process = _FakeProcess([None])  # exit_code never resolves
        sandbox = _make_sandbox(process)

        # Outer guard exists only to fail loud (rather than hang the suite) if
        # the fix regressed to an unbounded loop. With the fake clock the loop
        # is instant, so any real-time hang means an unbounded loop.
        with pytest.raises(RuntimeError) as excinfo:
            await asyncio.wait_for(
                sandbox._poll_response("sess-1", "cmd-1", timeout_sec=None),
                timeout=30.0,
            )

        msg = str(excinfo.value)
        # Same "Command timed out" RuntimeError type/wording as the bounded
        # path, but reports the cap and points at the likely cause + fix.
        assert "Command timed out after 5 seconds" in msg
        assert "safety-net cap" in msg
        assert "</dev/null >log 2>&1" in msg
        # It must have actually polled, not returned immediately, and never
        # reached the logs fetch (no successful completion).
        assert process.get_command_calls >= 1
        assert process.logs_fetched is False

    @pytest.mark.asyncio
    async def test_explicit_timeout_still_raises_on_its_own_deadline(
        self, monkeypatch
    ) -> None:
        """Explicit ``timeout_sec`` path is unchanged — byte-identical message.

        The hard cap is set huge so that if behavior had regressed onto the cap
        path the message would differ; the explicit deadline must fire first
        with the legacy wording.
        """
        _install_fake_clock(monkeypatch)
        monkeypatch.setattr(daytona_module, "_DAYTONA_EXEC_HARD_CAP_SEC", 9999)
        monkeypatch.setattr(daytona_module, "_DAYTONA_COMMAND_POLL_INTERVAL_SEC", 1.0)

        process = _FakeProcess([None])  # never resolves
        sandbox = _make_sandbox(process)

        with pytest.raises(RuntimeError) as excinfo:
            await asyncio.wait_for(
                sandbox._poll_response("sess-1", "cmd-1", timeout_sec=3),
                timeout=30.0,
            )

        msg = str(excinfo.value)
        # Legacy wording, no safety-net annotation.
        assert msg == "Command timed out after 3 seconds"
        assert "safety-net cap" not in msg
        assert process.logs_fetched is False

    @pytest.mark.asyncio
    async def test_returns_normally_when_exit_code_resolves_in_time(
        self, monkeypatch
    ) -> None:
        """A command that completes within the deadline returns its result."""
        _install_fake_clock(monkeypatch)
        monkeypatch.setattr(daytona_module, "_DAYTONA_COMMAND_POLL_INTERVAL_SEC", 1.0)

        # First poll: still running; second poll: completed with code 0.
        process = _FakeProcess([None, 0])
        sandbox = _make_sandbox(process)

        result = await asyncio.wait_for(
            sandbox._poll_response("sess-1", "cmd-1", timeout_sec=5),
            timeout=30.0,
        )

        assert result.return_code == 0
        assert result.stdout == "out"
        assert result.stderr == "err"
        assert process.logs_fetched is True

    @pytest.mark.asyncio
    async def test_none_timeout_returns_normally_when_command_completes(
        self, monkeypatch
    ) -> None:
        """The safety-net cap does not disturb a normally-completing command.

        With ``timeout_sec=None`` and an exit_code that resolves quickly, the
        result is returned well before the (generous) cap — proving the cap only
        guards the wedge case and does not change normal no-timeout behavior.
        """
        _install_fake_clock(monkeypatch)
        monkeypatch.setattr(daytona_module, "_DAYTONA_EXEC_HARD_CAP_SEC", 9999)
        monkeypatch.setattr(daytona_module, "_DAYTONA_COMMAND_POLL_INTERVAL_SEC", 1.0)

        # Completes on the very first poll.
        process = _FakeProcess([0])
        sandbox = _make_sandbox(process)

        result = await asyncio.wait_for(
            sandbox._poll_response("sess-1", "cmd-1", timeout_sec=None),
            timeout=30.0,
        )

        assert result.return_code == 0
        assert process.logs_fetched is True

    def test_hard_cap_is_generous(self) -> None:
        """The cap must sit well above any legitimately long command."""
        from benchflow.sandbox.daytona import _DAYTONA_EXEC_HARD_CAP_SEC

        # Sized for many-minute agent rollouts; guards against an accidental
        # shrink to a short blanket timeout (the thing BF-6 says NOT to do).
        assert _DAYTONA_EXEC_HARD_CAP_SEC >= 1800
