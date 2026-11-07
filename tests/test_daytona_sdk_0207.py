"""The start path keeps its rules on Daytona SDK 0.207.

``sandbox-daytona`` capped the SDK below 0.197, which forced FP's production
0.207 down to 0.196. 0.207 words and types some errors differently, and each
difference changed what the start path did:

- a sandbox in an error state is "Sandbox <id> entered error state: ...", no
  longer "... failed to start with state: ...": it was neither deleted nor
  replaced by a fresh sandbox (DAYTONA-START rule E);
- a transport timeout is a DaytonaConnectionTimeoutError: a create or startup
  probe that hit one was no longer retried;
- an HTTP 504 is a DaytonaTimeoutError: a gateway 504 on create was taken for
  a create that outran its time, and the start failed for good;
- every client opens a Socket.IO event connection whose asyncio client takes
  over the loop's SIGINT handler.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import daytona as daytona_module
from benchflow.sandbox import daytona_start
from tests.test_daytona_start_r3 import _Client, _err, _params, _sandbox, fast

_FAILED_ID = "00000000-0000-0000-0000-000000000007"

__all__ = ["fast"]  # the fixture, used by name below


@pytest.mark.parametrize("state", ["SandboxState.ERROR", "SandboxState.BUILD_FAILED"])
def test_the_0207_error_state_wording_is_a_failed_start(state):
    error = _err(
        "DaytonaError",
        "Failed to create sandbox: "
        f"Sandbox {_FAILED_ID} entered error state: {state}, error reason: boom",
    )
    assert daytona_start.error_kind(error) == "start_failed"
    assert daytona_module._failed_start_sandbox_id(error) == _FAILED_ID
    assert bool(daytona_start._START_BUILD_FAILED_TEXT.search(str(error))) == (
        state.endswith("BUILD_FAILED")
    )


async def test_a_sandbox_in_an_error_state_is_replaced(fast):
    errors = [
        _err(
            "DaytonaError",
            f"Sandbox {_FAILED_ID} entered error state: SandboxState.ERROR, "
            "error reason: image entrypoint exited",
        )
        for _ in range(2)
    ]
    client = _Client(errors)
    leftover = SimpleNamespace(id=_FAILED_ID, delete=AsyncMock())
    client.get.return_value = leftover
    sandbox = _sandbox(client)
    await sandbox._create_sandbox(params=_params())
    assert sandbox._sandbox is client.made and client.creates == 3
    assert leftover.delete.await_count == 2
    await sandbox.stop(delete=True)


def test_newer_sdk_timeouts_keep_their_kind():
    assert daytona_start.error_kind(_err("DaytonaConnectionTimeoutError", "")) == (
        "timeout"
    )
    assert daytona_start.error_kind(_err("DaytonaTimeoutError", "timed out")) == (
        "timeout"
    )
    gateway = _err(
        "DaytonaTimeoutError", "Failed to create sandbox: 504 Gateway Time-out", 504
    )
    assert daytona_start.error_kind(gateway) == "transient"


async def test_a_gateway_504_on_create_is_retried(fast, caplog):
    """Guards PR #13: retry a 504 only after its allocation is deleted."""
    errors = [
        _err(
            "DaytonaTimeoutError", "Failed to create sandbox: 504 Gateway Time-out", 504
        )
        for _ in range(2)
    ]

    class GatewayClient(_Client):
        async def create(self, params, timeout):
            if self.errors:
                self.land(dict(params.labels), f"sb-gateway-{self.creates}")
            return await super().create(params, timeout)

    client = GatewayClient(errors)
    sandbox = _sandbox(client)
    with caplog.at_level("ERROR"):
        await sandbox._create_sandbox(params=_params())
    assert sandbox._sandbox is client.made and client.creates == 3
    assert client.remote == []
    assert "Sandbox creation timed out" not in caplog.text
    await sandbox.stop(delete=True)


async def test_the_event_client_leaves_ctrl_c_to_benchflow(monkeypatch):
    engineio = ModuleType("engineio")
    async_client = ModuleType("engineio.async_client")
    async_client.async_signal_handler_set = False
    engineio.async_client = async_client
    monkeypatch.setitem(sys.modules, "engineio", engineio)
    monkeypatch.setitem(sys.modules, "engineio.async_client", async_client)
    monkeypatch.setattr(daytona_module, "_load_daytona_sdk", lambda: None)
    monkeypatch.setattr(daytona_module, "AsyncDaytona", object)
    manager = daytona_module.DaytonaClientManager()
    manager._cleanup_registered = True  # no atexit hook for the dummy client
    await manager.get_client()
    assert async_client.async_signal_handler_set is True


def test_the_installed_engineio_still_has_the_flag():
    """If python-engineio renames its flag, Ctrl-C silently goes to it: fail."""
    try:
        from engineio import async_client
    except ImportError as exc:
        pytest.skip(f"python-engineio not installed (SDK below 0.197): {exc!r}")
    assert isinstance(async_client.async_signal_handler_set, bool)
