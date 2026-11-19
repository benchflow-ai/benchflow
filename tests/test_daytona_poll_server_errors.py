"""Guards the fix against the command polls' retry rule (``_SDK_RETRY``), which
retried transport errors only: one 502 or 503 from Daytona's gateway while a
verifier ran crashed it ("verifier crashed: Failed to get session command",
2026-10-07). The status and logs polls of a running command are idempotent
reads; they now retry a 5xx for a bounded time, and a 4xx other than 408 and
429 (a timeout and a rate limit, retried as before) still fails at once.
Newer SDKs' transport timeout class (DaytonaConnectionTimeoutError, 0.201+)
is a transient error too.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from benchflow.diagnostics import TransientSandboxTransportError
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


class DaytonaTimeoutError(DaytonaError):
    """SDK 0.207's class for HTTP 408 (and 504)."""


class DaytonaRateLimitError(DaytonaError):
    """SDK 0.207's class for HTTP 429."""


for _cls in (
    DaytonaError,
    DaytonaConnectionTimeoutError,
    DaytonaProcessExecutionTimeoutError,
    DaytonaTimeoutError,
    DaytonaRateLimitError,
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


@pytest.mark.parametrize(
    "error",
    [
        DaytonaError("Failed to get session command: 503", 503),
        DaytonaConnectionTimeoutError(""),
    ],
    ids=["5xx", "transport"],
)
async def test_poll_errors_are_retried_for_about_three_minutes(waits, error):
    """FP patch 37: waits of 1 s doubling to 60 s, until they add up to
    BENCHFLOW_DAYTONA_POLL_RETRY_SEC (180 s); then the error is raised."""
    process = _Process(status_errors=[error] * 50)
    with pytest.raises((DaytonaError, TransientSandboxTransportError)):
        await _sandbox(process)._poll_response("s", "c", timeout_sec=600)
    assert waits == [1, 2, 4, 8, 16, 32, 60, 60]
    assert process.status_calls == 9


async def test_a_transport_blip_longer_than_a_few_seconds_is_survived(waits):
    """FP patch 37: five timed-out status requests in a row (about 31 s of
    waits) no longer crash the poll after its third attempt."""
    process = _Process(status_errors=[DaytonaConnectionTimeoutError("")] * 5)
    result = await _sandbox(process)._poll_response("s", "c", timeout_sec=600)
    assert (result.return_code, result.stdout) == (0, "out")
    assert process.status_calls == 6


@pytest.mark.parametrize(("value", "calls"), [("10", 5), ("0", 1), ("soon", 9)])
async def test_the_retry_budget_is_set_by_the_environment(
    waits, monkeypatch, value, calls
):
    monkeypatch.setenv("BENCHFLOW_DAYTONA_POLL_RETRY_SEC", value)
    process = _Process(status_errors=[DaytonaConnectionTimeoutError("")] * 50)
    with pytest.raises(TransientSandboxTransportError):
        await _sandbox(process)._poll_response("s", "c", timeout_sec=600)
    assert process.status_calls == calls


@pytest.mark.parametrize(
    "value", ["inf", "Infinity", "1e999", "-inf", "nan", "-1", "-0.5"]
)
def test_a_budget_that_is_not_a_finite_count_of_seconds_is_refused(monkeypatch, value):
    """a review N2: ``inf`` was taken as is, and the polls then retried a
    dead API for ever. It and a negative value give the 180 s default."""
    monkeypatch.setenv("BENCHFLOW_DAYTONA_POLL_RETRY_SEC", value)
    assert daytona_module.daytona_poll_retry_sec() == 180.0


async def test_an_infinite_budget_still_gives_up(waits, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_DAYTONA_POLL_RETRY_SEC", "inf")
    process = _Process(status_errors=[DaytonaConnectionTimeoutError("")] * 50)
    with pytest.raises(TransientSandboxTransportError):
        await _sandbox(process)._poll_response("s", "c", timeout_sec=600)
    assert process.status_calls == 9


@pytest.mark.parametrize(
    "error",
    [
        DaytonaTimeoutError("Request Timeout", 408),
        DaytonaRateLimitError("Too Many Requests", 429),
    ],
    ids=["408", "429"],
)
async def test_a_timeout_or_rate_limit_4xx_is_retried(waits, error):
    """The docstring and the release note say so (a review N2): of the
    4xx, 408 and 429 are asked again, as before."""
    process = _Process(status_errors=[error] * 3)
    result = await _sandbox(process)._poll_response("s", "c", timeout_sec=600)
    assert result.return_code == 0
    assert process.status_calls == 4


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
