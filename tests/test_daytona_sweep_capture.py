"""Guards FF-sw-daytona asynchronous context capture at 363d815b."""

import threading

import pytest

from benchflow.sandbox import daytona_content_snapshots as content
from tests.test_daytona_content_snapshots import (
    PINNED_BASE,
    Client,
    _record,
    _start,
    _task,
)
from tests.test_daytona_content_snapshots import server as snapshot_server


@pytest.fixture
def server(monkeypatch):
    return snapshot_server.__wrapped__(monkeypatch)


@pytest.mark.parametrize("eligible", [False, True])
async def test_capture_skips_ineligible_and_never_blocks_loop(
    monkeypatch, server, tmp_path, eligible
):
    """Guard 363d815b: recipe eligibility precedes off-thread disk work."""
    base = PINNED_BASE if eligible else "ubuntu:24.04"
    dockerfile = _task(
        tmp_path / "task", f"FROM {base}\nCOPY seed /seed\n", {"seed": "dummy"}
    )
    loop_thread = threading.get_ident()
    calls = []
    original = content._capture

    def capture(*args, **kwargs):
        calls.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(content, "_capture", capture)
    sandbox = await _start(
        monkeypatch,
        Client(server),
        tmp_path / "trial",
        image=None,
        dockerfile=dockerfile,
    )
    if eligible:
        assert calls and all(thread != loop_thread for thread in calls)
        assert _record(sandbox)["snapshot"] == "created"
    else:
        assert not calls
        assert _record(sandbox)["snapshot"] == "off"


@pytest.mark.parametrize("phase", ["capture", "hash"])
async def test_cancel_waits_for_worker_before_removing_staging(
    monkeypatch, server, tmp_path, phase
):
    """Guard FF-sw-daytona: cancelled off-thread work cannot outlive its files."""
    import asyncio
    from pathlib import Path

    dockerfile = _task(
        tmp_path / "task", f"FROM {PINNED_BASE}\nCOPY seed /seed\n", {"seed": "dummy"}
    )
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    paths = []
    name = "_capture" if phase == "capture" else "content_identity"
    original = getattr(content, name)

    def work(*args, **kwargs):
        result = original(*args, **kwargs)
        paths.extend(
            [Path(args[1])]
            if phase == "capture"
            else [Path(source) for source, _ in kwargs["context"]]
        )
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(10), "test did not release worker"
        assert all(path.exists() for path in paths)
        return result

    monkeypatch.setattr(content, name, work)
    task = asyncio.create_task(
        _start(
            monkeypatch,
            Client(server),
            tmp_path / "trial",
            image=None,
            dockerfile=dockerfile,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 3)
        task.cancel()
        await asyncio.sleep(0)
        assert all(path.exists() for path in paths)
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert paths and all(not path.exists() for path in paths)
