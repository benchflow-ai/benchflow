"""Deleting a Daytona sandbox survives a gateway error from the Daytona API.

Guards against a leaked sandbox: the trial finished normally, but the
delete got ``502 Bad Gateway``. ``_stop_sandbox`` retried only connection,
rate-limit and timeout errors, so it logged ``Error stopping sandbox …`` with
the gateway's HTML page and left the sandbox running.

The SDK raises the base ``DaytonaError`` with ``status_code`` 502 for that
response; it is faked here by module name, as in
``tests/test_daytona_failed_start_cleanup.py``. The sandbox is built via
``__new__``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDinD, _DaytonaDirect

pytest.importorskip("tenacity")  # the retry policy ships with sandbox-daytona

_ID = "00000000-0000-0000-0000-000000000001"
_GATEWAY_PAGE = (
    "Failed to remove sandbox: <html>\n<head><title>502 Bad Gateway</title></head>\n"
    "<body>\n<center><h1>502 Bad Gateway</h1></center>\n</body>\n</html>\n"
)


class DaytonaError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


DaytonaError.__module__ = "daytona.common.errors"


class DaytonaNotFoundError(DaytonaError):
    pass


DaytonaNotFoundError.__module__ = "daytona.common.errors"


def _bad_gateway() -> DaytonaError:
    return DaytonaError(_GATEWAY_PAGE, status_code=502)


class _Remote:
    """A Daytona sandbox whose delete() fails with the given errors in turn."""

    def __init__(self, *errors: BaseException) -> None:
        self.id = _ID
        self.errors = list(errors)
        self.deletes = 0

    async def delete(self) -> None:
        self.deletes += 1
        if self.errors:
            raise self.errors.pop(0)


def _stopper(strategy, remote: _Remote):
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox._sandbox = remote
    sandbox._client_manager = None
    sandbox._kwargs = {}
    sandbox._persistent_env = {}
    sandbox.environment_name = "task"
    sandbox.environment_dir = Path("environment")
    sandbox.task_env_config = SimpleNamespace(
        build_timeout_sec=60,
        cpus=1,
        memory_mb=1024,
        storage_mb=1024,
        docker_image=None,
        env={},
    )
    sandbox.logger = logging.getLogger("test.daytona.stop_retry")
    stopper = strategy(sandbox)
    if strategy is _DaytonaDinD:

        async def compose_down(*args, **kwargs):
            return None

        stopper._compose_exec = compose_down
    return sandbox, stopper


@pytest.fixture
def waits(monkeypatch) -> list[float]:
    """Backoff waits taken by the delete retry, without sleeping."""
    taken: list[float] = []

    async def sleep(seconds: float) -> None:
        taken.append(seconds)

    monkeypatch.setattr(DaytonaSandbox._stop_sandbox.retry, "sleep", sleep)
    return taken


def _errors(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "test.daytona.stop_retry" and record.levelno >= logging.ERROR
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", [_DaytonaDirect, _DaytonaDinD])
@pytest.mark.parametrize("status", [502, 503, 504])
async def test_a_gateway_error_on_delete_is_retried(
    strategy, status, waits, caplog
) -> None:
    remote = _Remote(DaytonaError(f"Failed to remove sandbox: {status}", status))
    sandbox, stopper = _stopper(strategy, remote)

    await stopper.stop(delete=True)

    assert remote.deletes == 2
    assert len(waits) == 1
    assert _errors(caplog) == []
    assert sandbox._sandbox is None


@pytest.mark.asyncio
async def test_a_delete_that_already_happened_behind_the_gateway_counts(
    waits, caplog
) -> None:
    """The gateway can fail the response after Daytona deleted the sandbox;
    the retry then finds it gone, which is what stop() wanted."""
    remote = _Remote(_bad_gateway(), DaytonaNotFoundError("not found", 404))
    _, stopper = _stopper(_DaytonaDirect, remote)

    await stopper.stop(delete=True)

    assert remote.deletes == 2
    assert _errors(caplog) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", [_DaytonaDirect, _DaytonaDinD])
async def test_a_gateway_error_on_every_attempt_is_reported_with_the_id(
    strategy, waits, caplog
) -> None:
    remote = _Remote(*(_bad_gateway() for _ in range(10)))
    sandbox, stopper = _stopper(strategy, remote)

    await stopper.stop(delete=True)

    # Bounded: four attempts, with a growing wait of at most 10 s between them.
    assert remote.deletes == 4
    assert len(waits) == 3 and waits == sorted(waits) and max(waits) <= 10
    errors = _errors(caplog)
    assert len(errors) == 1
    assert _ID in errors[0]
    assert "502 Bad Gateway" in errors[0]
    assert "<html>" not in errors[0]
    assert "bench sandbox cleanup" in errors[0]
    assert sandbox._sandbox is None


@pytest.mark.asyncio
async def test_other_delete_errors_keep_the_short_budget(waits, caplog) -> None:
    """A non-gateway API error is not retried, and a timeout keeps the old
    two-attempt budget: each attempt can take the SDK's 60 s delete timeout."""
    remote = _Remote(DaytonaError("Failed to remove sandbox: forbidden", 403))
    _, stopper = _stopper(_DaytonaDirect, remote)
    await stopper.stop(delete=True)
    assert remote.deletes == 1
    assert len(_errors(caplog)) == 1

    remote = _Remote(*(TimeoutError("delete timed out") for _ in range(10)))
    _, stopper = _stopper(_DaytonaDirect, remote)
    await stopper.stop(delete=True)
    assert remote.deletes == 2
