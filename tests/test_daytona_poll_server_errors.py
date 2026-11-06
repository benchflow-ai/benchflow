"""Guards the fix against the command polls' retry rule (``_SDK_RETRY``), which
retried transport errors only: one 502 or 503 from Daytona's gateway while a
verifier ran crashed it ("verifier crashed: Failed to get session command",
2026-10-07). The status and logs polls of a running command are idempotent
reads; they now retry a 5xx for a bounded time, and a 4xx still fails at once.
Newer SDKs' transport timeout class (DaytonaConnectionTimeoutError, 0.201+)
is a transient error too.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from benchflow.sandbox import daytona as daytona_module
from benchflow.sandbox.daytona import DaytonaSandbox


class DaytonaError(Exception):
    """The SDK's error as far as benchflow reads it: module and status code."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class DaytonaConnectionTimeoutError(DaytonaError):
    pass


class DaytonaProcessExecutionTimeoutError(DaytonaError):
    pass


for _cls in (
    DaytonaError,
    DaytonaConnectionTimeoutError,
    DaytonaProcessExecutionTimeoutError,
):
    _cls.__module__ = "daytona.common.errors"


class _Process:
    """``sandbox.process``: each poll raises the next queued error, then answers."""

    def __init__(self, status_errors=(), log_errors=()) -> None:
        self.status_errors = list(status_errors)
        self.log_errors = list(log_errors)
        self.status_calls = 0
        self.log_calls = 0

    async def get_session_command(self, session_id, command_id):
        self.status_calls += 1
        if self.status_errors:
            raise self.status_errors.pop(0)
        return SimpleNamespace(id=command_id, exit_code=0)

    async def get_session_command_logs(self, session_id, command_id):
        self.log_calls += 1
        if self.log_errors:
            raise self.log_errors.pop(0)
        return SimpleNamespace(stdout="out", stderr="err")


def _sandbox(process: _Process) -> DaytonaSandbox:
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox._sandbox = SimpleNamespace(process=process)
    return sandbox


@pytest.fixture
def waits(monkeypatch) -> list[float]:
    """The retry's backoff, recorded instead of slept."""
    slept: list[float] = []

    async def sleep(delay, result=None):
        slept.append(delay)
        return result

    monkeypatch.setattr(asyncio, "sleep", sleep)
    return slept


@pytest.mark.parametrize("status", [500, 502, 503, 504])
async def test_a_status_poll_survives_a_server_error(waits, status):
    error = DaytonaError(f"Failed to get session command: HTTP {status}", status)
    process = _Process(status_errors=[error] * 4)
    result = await _sandbox(process)._poll_response("s", "c", timeout_sec=600)
    assert (result.return_code, result.stdout) == (0, "out")
    assert process.status_calls == 5


async def test_a_logs_poll_survives_a_gateway_page_without_a_status(waits):
    page = DaytonaError("<html><title>502 Bad Gateway</title></html>")
    process = _Process(log_errors=[page, page])
    result = await _sandbox(process)._poll_response("s", "c", timeout_sec=600)
    assert result.stderr == "err"
    assert process.log_calls == 3


async def test_server_errors_are_retried_for_a_bounded_time(waits):
    error = DaytonaError("Failed to get session command: 503", 503)
    process = _Process(status_errors=[error] * 50)
    with pytest.raises(DaytonaError):
        await _sandbox(process)._poll_response("s", "c", timeout_sec=600)
    assert process.status_calls == daytona_module._DAYTONA_POLL_SERVER_RETRY_ATTEMPTS
    assert 60 <= sum(waits) <= 120


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409])
async def test_a_client_error_fails_at_once(waits, status):
    process = _Process(status_errors=[DaytonaError("refused", status)])
    with pytest.raises(DaytonaError):
        await _sandbox(process)._poll_response("s", "c", timeout_sec=600)
    assert process.status_calls == 1


async def test_a_newer_sdks_transport_timeout_is_retried(waits):
    process = _Process(status_errors=[DaytonaConnectionTimeoutError("")] * 2)
    result = await _sandbox(process)._poll_response("s", "c", timeout_sec=600)
    assert result.return_code == 0
    assert process.status_calls == 3
    assert daytona_module._is_daytona_transient_retry_error(
        DaytonaConnectionTimeoutError("")
    )
    # A command that outran its own timeout is no transport failure.
    assert not daytona_module._is_daytona_transient_retry_error(
        DaytonaProcessExecutionTimeoutError("command timed out")
    )
