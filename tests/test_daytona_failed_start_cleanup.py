"""A Daytona sandbox that fails to start must not outlive the failed create.

The Daytona SDK creates the sandbox, then raises once it reaches an error state
(``BUILD_FAILED`` for a broken Dockerfile) without returning or deleting it.
A task with a missing ``COPY`` source leaves one ``BUILD_FAILED`` sandbox
behind this way per attempt.

The SDK is faked at the client boundary (the ``sandbox-daytona`` extra is not
part of the dev install), and the sandbox is built via ``__new__`` as in
``tests/test_daytona_command_polling.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import daytona as daytona_module
from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDinD, _DaytonaDirect
from benchflow.sandbox.protocol import SandboxStartupError

_FAILED_ID = "00000000-0000-0000-0000-000000000002"


class _FakeDaytonaError(Exception):
    pass


_FakeDaytonaError.__module__ = "daytona.common.errors"


def _build_failed() -> Exception:
    return _FakeDaytonaError(
        "Failed to create sandbox: Failure during waiting for sandbox to "
        f"start: Sandbox {_FAILED_ID} failed to start with state: "
        "SandboxState.BUILD_FAILED, error reason: failed to compute cache key"
    )


class _FakeClient:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.leftover = SimpleNamespace(id=_FAILED_ID, delete=AsyncMock())
        self.get = AsyncMock(return_value=self.leftover)

    async def create(self, params, timeout):
        raise self.error


def _sandbox(client: _FakeClient) -> DaytonaSandbox:
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox._sandbox = None
    sandbox._create_attempts = 0
    sandbox._client_manager = SimpleNamespace(get_client=AsyncMock(return_value=client))
    sandbox.task_env_config = SimpleNamespace(
        build_timeout_sec=60, cpus=1, memory_mb=1024, storage_mb=1024, docker_image=None
    )
    sandbox.logger = logging.getLogger("test.daytona.failed_start")
    return sandbox


@pytest.mark.asyncio
async def test_sandbox_that_failed_to_build_is_deleted() -> None:
    """Guards _create_sandbox cleanup, which only knew sandboxes create returned."""
    client = _FakeClient(_build_failed())
    sandbox = _sandbox(client)

    with pytest.raises(_FakeDaytonaError, match="BUILD_FAILED"):
        await sandbox._create_sandbox(params=object())

    client.get.assert_awaited_once_with(_FAILED_ID)
    client.leftover.delete.assert_awaited_once()
    assert sandbox._sandbox is None


@pytest.mark.asyncio
async def test_create_error_without_a_sandbox_deletes_nothing() -> None:
    """Guards the same cleanup: an error before any sandbox exists names none."""
    client = _FakeClient(
        _FakeDaytonaError("Failed to create sandbox: Path does not exist: /x/data")
    )
    sandbox = _sandbox(client)

    with pytest.raises(_FakeDaytonaError, match="Path does not exist"):
        await sandbox._create_sandbox(params=object())

    client.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_cleanup_keeps_the_original_create_error() -> None:
    """Guards the same cleanup: a delete failure must not mask the build error."""
    client = _FakeClient(
        _FakeDaytonaError(
            f"Sandbox {_FAILED_ID} failed to start with state: SandboxState.ERROR, "
            "error reason: no capacity"
        )
    )
    client.leftover.delete.side_effect = _FakeDaytonaError("delete refused")
    sandbox = _sandbox(client)

    with pytest.raises(_FakeDaytonaError, match="no capacity"):
        await sandbox._create_sandbox(params=object())

    client.leftover.delete.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", [_DaytonaDirect, _DaytonaDinD])
async def test_startup_error_reports_the_real_attempts_and_sandbox(
    strategy, monkeypatch
) -> None:
    """Guards the startup diagnostic, which always claimed 3 attempts and no id.

    A build failure is not a transient SDK error, so create runs once; the
    diagnostic said ``attempts=3`` / "failed after retries" and dropped the id
    of the sandbox that failed.
    """
    client = _FakeClient(_build_failed())
    sandbox = _sandbox(client)
    sandbox._kwargs = {}
    sandbox._persistent_env = {}
    sandbox.environment_name = "task"
    sandbox.task_env_config.env = {}
    sandbox._snapshot_template_name = None
    sandbox.environment_dir = Path("environment")
    sandbox._auto_delete_interval = 0
    sandbox._auto_stop_interval = 0
    sandbox._network_block_all = False
    for name, value in {
        "Resources": lambda **kwargs: kwargs,
        "Image": SimpleNamespace(from_dockerfile=str, base=str),
        "CreateSandboxFromImageParams": lambda **kwargs: kwargs,
        "DaytonaClientManager": SimpleNamespace(
            get_instance=AsyncMock(return_value=sandbox._client_manager)
        ),
    }.items():
        monkeypatch.setattr(daytona_module, name, value, raising=False)

    with pytest.raises(SandboxStartupError) as raised:
        await strategy(sandbox).start(force_build=True)

    diagnostic = raised.value.diagnostic
    assert diagnostic.attempts == 1
    assert diagnostic.sandbox_id == _FAILED_ID
    assert "after 1 attempt:" in str(raised.value)


# Teardown after a failed or unfinished create. Regression test:
# stop() after a create that failed
# before any sandbox existed logged "Sandbox not found. Please build the
# environment first." as a warning, advice that makes no sense at cleanup.


def _stop_logs(caplog) -> list[tuple[int, str]]:
    return [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == "test.daytona.failed_start"
    ]


_create = getattr(
    DaytonaSandbox._create_sandbox, "__wrapped__", DaytonaSandbox._create_sandbox
)


def _stopper(strategy, sandbox: DaytonaSandbox):
    """The strategy under test, with what its constructor reads."""
    sandbox._kwargs = {}
    sandbox._persistent_env = {}
    sandbox.environment_name = "task"
    sandbox.environment_dir = Path("environment")
    sandbox.task_env_config.env = {}
    return strategy(sandbox)


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", [_DaytonaDirect, _DaytonaDinD])
async def test_stop_after_a_failed_create_is_quiet(strategy, caplog) -> None:
    client = _FakeClient(
        _FakeDaytonaError("Failed to create sandbox: Path does not exist: /x/data")
    )
    sandbox = _sandbox(client)
    with pytest.raises(_FakeDaytonaError):
        await _create(sandbox, params=object())

    stopper = _stopper(strategy, sandbox)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="test.daytona.failed_start"):
        await stopper.stop(delete=True)
        # A second stop (the rollout's cleanup after an earlier one) too.
        await stopper.stop(delete=True)

    logs = _stop_logs(caplog)
    assert [level for level, _ in logs if level >= logging.WARNING] == []
    assert not any("Please build the environment" in text for _, text in logs)
    assert [level for level, text in logs if "No Daytona sandbox" in text] == [
        logging.DEBUG,
        logging.DEBUG,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", [_DaytonaDirect, _DaytonaDinD])
async def test_stop_after_an_unfinished_create_still_warns(
    strategy, caplog, monkeypatch
) -> None:
    """A create that timed out may still produce a sandbox on Daytona that
    this process has no handle for, so stop() says so as a warning."""
    import asyncio

    class _SlowClient(_FakeClient):
        async def create(self, params, timeout):
            await asyncio.sleep(30)

    monkeypatch.setattr(daytona_module, "_STARTUP_HARD_TIMEOUT_BUFFER_SEC", 0)
    sandbox = _sandbox(_SlowClient(_build_failed()))
    sandbox.task_env_config.build_timeout_sec = 0
    with pytest.raises(TimeoutError):
        await _create(sandbox, params=object())

    stopper = _stopper(strategy, sandbox)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="test.daytona.failed_start"):
        await stopper.stop(delete=True)

    warnings = [text for level, text in _stop_logs(caplog) if level >= logging.WARNING]
    assert len(warnings) == 1
    assert "may still exist" in warnings[0]
    assert "bench sandbox cleanup" in warnings[0]


class _GatewayFailureClient:
    """Create makes the sandbox, then the SDK's start wait gets a 502 whose
    message carries no sandbox id, which used to leak the sandbox."""

    def __init__(self) -> None:
        self.made: list[SimpleNamespace] = []
        self.queries: list[dict[str, str]] = []

    async def create(self, params, timeout):
        leftover = SimpleNamespace(
            id="00000000-0000-0000-0000-000000000005",
            labels=dict(params.labels),
            delete=AsyncMock(),
        )
        self.made.append(leftover)
        raise _FakeDaytonaError(
            "Failed to create sandbox: Failure during waiting for sandbox to "
            "start: Failed to refresh sandbox data: <html><title>502 Bad "
            "Gateway</title></html>"
        )

    async def list(self, query=None):
        wanted = dict(getattr(query, "labels", None) or {})
        self.queries.append(wanted)
        for sandbox in self.made:
            if wanted.items() <= sandbox.labels.items():
                yield sandbox


@pytest.mark.asyncio
async def test_sandbox_left_by_a_create_error_without_an_id_is_deleted() -> None:
    """The failed create names no sandbox; find it by its per-attempt label."""
    client = _GatewayFailureClient()
    sandbox = _sandbox(client)
    params = SimpleNamespace(labels={"benchflow.owner": "e2e"})

    with pytest.raises(_FakeDaytonaError, match="502"):
        await sandbox._create_sandbox(params=params)

    [leftover] = client.made
    leftover.delete.assert_awaited_once()
    [query] = client.queries
    assert query and "benchflow.owner" not in query  # the attempt's own label
    assert query.items() <= leftover.labels.items()
