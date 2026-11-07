"""Every snapshot await shares the caller's complete allowance."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import daytona_content_snapshots as content


@pytest.mark.parametrize(
    "blocked", ["lock", "get", "create", "reread", "delete", "activate", "cleanup"]
)
async def test_snapshot_budget_covers_every_wait(blocked):
    """Guard 1bf9bd73: lock and API waits share the snapshot deadline."""
    name = f"deadline-{blocked}"
    failed = SimpleNamespace(id="failed", state="error")
    active = SimpleNamespace(id="active", state="active")
    service = SimpleNamespace(
        get=AsyncMock(return_value=active),
        create=AsyncMock(return_value=active),
        delete=AsyncMock(),
        activate=AsyncMock(),
    )

    async def hang(*args, **kwargs):
        await asyncio.Event().wait()

    class Missing(Exception):
        status_code = 404

    lock = content._name_lock(name)
    if blocked == "lock":
        await lock.acquire()
    elif blocked == "get":
        service.get.side_effect = hang
    elif blocked in ("create", "cleanup"):
        calls = 0

        async def missing_then_hang(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise Missing()
            await hang()

        service.get.side_effect = missing_then_hang
        service.create.side_effect = hang if blocked == "create" else RuntimeError()
    elif blocked == "reread":

        async def reread(*args):
            service.get.side_effect = hang
            return failed

        service.get.side_effect = reread
    elif blocked == "delete":
        service.get.return_value = failed
        service.delete.side_effect = hang
    elif blocked == "activate":
        service.get.return_value = SimpleNamespace(state="inactive")
        service.activate.side_effect = hang
    try:
        with pytest.raises(content.SnapshotFallback, match="within"):
            await asyncio.wait_for(
                content.ensure_snapshot(
                    SimpleNamespace(snapshot=service),
                    name,
                    lambda: None,
                    timeout_sec=0.03,
                ),
                timeout=0.3,
            )
        if blocked != "lock":
            assert not lock.locked()
    finally:
        if blocked == "lock":
            lock.release()


async def test_external_snapshot_cancellation_is_preserved():
    """Guard 1bf9bd73: external cancellation must still propagate."""
    entered = asyncio.Event()

    async def hang(name):
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(
        content.ensure_snapshot(
            SimpleNamespace(snapshot=SimpleNamespace(get=hang)),
            "external-cancel",
            lambda: None,
            timeout_sec=10,
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not content._name_lock("external-cancel").locked()
