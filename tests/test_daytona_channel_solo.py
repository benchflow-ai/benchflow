"""The solo channel uses on-demand sessions for every fallback."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import daytona as module
from benchflow.sandbox._exec_channel import ChannelUnavailable
from benchflow.sandbox.daytona import DaytonaSandbox
from tests.daytona_channel_fakes import _lifecycle_sandbox, _Remote
from tests.test_daytona_channel_contract import _exec_box
from tests.test_daytona_channel_lifecycle import _channel

pytestmark = pytest.mark.timeout(30)


@pytest.mark.parametrize("fallback", ["setting", "environment", "never-started"])
@pytest.mark.parametrize("transient", [False, True])
async def test_fallback_creates_exactly_one_session_per_command(
    monkeypatch, fallback, transient
):
    """Fallback starts fresh sessions only when commands need them."""
    box, _pool, made = _exec_box(monkeypatch)
    provider = box._sandbox
    process = provider.process
    if fallback == "setting":
        box.exec_channel = False
    elif fallback == "environment":
        monkeypatch.setenv("BENCHFLOW_EXEC_CHANNEL", "0")
    else:

        async def not_sent(*args, **kwargs):
            raise ChannelUnavailable("never left host")

        made.run = not_sent
    box._on_sandbox_created()
    await asyncio.sleep(0)
    assert process.created == []
    execute = box.exec_transient if transient else box.exec
    results = await asyncio.gather(
        *(execute("true", user="root", timeout_sec=5) for _ in range(4))
    )
    await asyncio.sleep(0)
    assert all(result.return_code == 0 for result in results)
    assert all(result.stdout == "BF daemon-log root-only\n" for result in results)
    assert process.created == process.executed
    assert len(set(process.created)) == 4
    assert process.deleted == (process.created if transient else [])
    counts = box._exec_counts()
    assert counts.transient_exec_calls == (4 if transient else 0)
    assert counts.exec_calls == (0 if transient else 4)
    assert counts.channel_fallbacks == (4 if fallback == "never-started" else 0)
    made.pty.send_input.assert_not_awaited()
    provider.release.set()
    await box.stop(True)
    assert process.created == process.executed


@pytest.mark.parametrize("strategy", [module._DaytonaDirect, module._DaytonaDinD])
@pytest.mark.parametrize("delete_fails", [False, True])
async def test_stop_releases_only_its_captured_provider(
    monkeypatch, strategy, delete_fails
):
    """A provider published during delete keeps its handle and client."""
    sandbox, ledger = _lifecycle_sandbox(monkeypatch, strategy)
    old, new = sandbox._sandbox, _Remote("new-box")
    old.fail_delete = delete_fails
    work = asyncio.create_task(sandbox.stop(True))
    await old.deleting.wait()
    sandbox._sandbox = new
    sandbox._on_sandbox_created()
    client = sandbox._client_manager = object()
    old.release.set()
    await work
    assert sandbox._sandbox is new
    assert sandbox._client_manager is client
    assert new.deletes == 0
    assert ledger.entries == ([old.id] if delete_fails else [])


async def test_delete_retry_keeps_the_original_provider(monkeypatch):
    """Deletion retries cannot release a replacement's channel."""
    sandbox, ledger = _lifecycle_sandbox(monkeypatch)
    old, new = sandbox._sandbox, _Remote("new-box")
    old_pool, _ = _channel(sandbox)
    replacement = []
    monkeypatch.setattr(DaytonaSandbox._stop_sandbox.retry, "sleep", AsyncMock())

    async def delete(**kwargs):
        old.deletes += 1
        if old.deletes == 1:
            sandbox._sandbox = new
            sandbox._on_sandbox_created()
            replacement.append(_channel(sandbox))
            raise TimeoutError("delete answer lost")
        old.deleted = True

    old.delete = delete
    await sandbox.stop(True)
    assert old.deleted and old.deletes == 2
    assert sandbox._sandbox is new and new.deletes == 0
    assert old_pool.finalized
    new_pool, new_channel = replacement[0]
    assert sandbox._command_channels is new_pool
    assert not new_pool.closed and new_channel.alive
    # Releasing the old provider leaves the replacement's channel eligible (REVIEW-4 L-2).
    assert not sandbox._command_channels_closing
    assert ledger.entries == []
    new.release.set()
    await sandbox.stop(True)


async def test_dind_stop_captures_provider_before_compose_teardown(monkeypatch):
    """Compose teardown cannot make stop delete a replacement provider."""
    sandbox, _ = _lifecycle_sandbox(monkeypatch, module._DaytonaDinD)
    old, new = sandbox._sandbox, _Remote("new-box")
    old.release.set()

    async def teardown(**kwargs):
        sandbox._sandbox = new
        sandbox._on_sandbox_created()

    sandbox._strategy._compose_teardown = teardown
    await sandbox.stop(True)
    assert old.deleted and old.deletes == 1
    assert sandbox._sandbox is new and new.deletes == 0
