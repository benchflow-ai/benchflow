"""A content name and the SDK upload must read the same private copy."""

import asyncio
import errno
from pathlib import Path

import pytest

pytest.importorskip("daytona")

from benchflow.sandbox import daytona_content_snapshots as content
from tests.test_daytona_content_snapshots import (
    PINNED_BASE,
    Client,
    Snapshots,
    _record,
    _start,
    _task,
)
from tests.test_daytona_content_snapshots import (
    server as snapshot_server,
)


@pytest.fixture
def server(monkeypatch):
    return snapshot_server.__wrapped__(monkeypatch)


@pytest.mark.parametrize("entry", ["file", "directory"])
async def test_snapshot_upload_uses_the_content_that_named_it(
    monkeypatch, server, tmp_path, entry
):
    """Guard a6d5af73: name and upload consume one captured context."""
    source = "seed.txt" if entry == "file" else "seed/seed.txt"
    copied = "seed.txt" if entry == "file" else "seed"
    dockerfile = _task(
        tmp_path / "task",
        f"FROM {PINNED_BASE}\nCOPY {copied} /task/seed\n",
        {source: "marker-A"},
    )
    original = dockerfile.parent / source
    uploaded = []
    staged = []
    get = Snapshots.get
    create = Snapshots.create

    async def changed_get(self, name):
        original.write_text("marker-B")
        return await get(self, name)

    async def read_upload(self, params, timeout=None):
        # This is when SDK 0.184.0 opens the source paths for its archive.
        for context in params.image._context_list:
            path = Path(context.source_path)
            staged.append(path)
            if path.is_dir():
                uploaded.append((path / "seed.txt").read_text())
            else:
                uploaded.append(path.read_text())
        return await create(self, params, timeout)

    monkeypatch.setattr(Snapshots, "get", changed_get)
    monkeypatch.setattr(Snapshots, "create", read_upload)
    first = await _start(
        monkeypatch,
        Client(server),
        tmp_path / "trial-1",
        image=None,
        dockerfile=dockerfile,
    )
    assert uploaded == ["marker-A"]
    assert all(not path.exists() for path in staged)
    original.write_text("marker-A")
    second = await _start(
        monkeypatch,
        Client(server),
        tmp_path / "trial-2",
        image=None,
        dockerfile=dockerfile,
    )
    assert _record(first)["snapshot_name"] == _record(second)["snapshot_name"]
    assert _record(second)["snapshot"] == "reused"


@pytest.mark.parametrize("ending", ["failure", "cancel"])
async def test_captured_upload_is_removed_after_failure_or_cancellation(
    monkeypatch, server, tmp_path, ending
):
    """Guard a6d5af73: staging ends with the attempt, leaving sources intact."""
    dockerfile = _task(
        tmp_path / "task",
        f"FROM {PINNED_BASE}\nCOPY seed.txt /task/seed\n",
        {"seed.txt": "marker-A"},
    )
    staged = []
    entered = asyncio.Event()

    async def create(self, params, timeout=None):
        staged.extend(Path(c.source_path) for c in params.image._context_list)
        entered.set()
        if ending == "failure":
            raise RuntimeError("dummy build failure")
        await asyncio.Event().wait()

    monkeypatch.setattr(Snapshots, "create", create)
    task = asyncio.create_task(
        _start(
            monkeypatch,
            Client(server),
            tmp_path / "trial",
            image=None,
            dockerfile=dockerfile,
        )
    )
    await asyncio.wait_for(entered.wait(), 3)
    if ending == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        sandbox = await task
        assert _record(sandbox)["snapshot"] == "fallback"
    assert staged and all(not path.exists() for path in staged)
    assert (dockerfile.parent / "seed.txt").read_text() == "marker-A"


@pytest.mark.parametrize("with_context", [False, True])
async def test_staging_allocation_is_optional_and_falls_back_on_oserror(
    monkeypatch, server, tmp_path, with_context
):
    """Guard abb528a3/U13-CONFIRM-3: local ENOSPC cannot abort remote builds."""
    dockerfile = _task(
        tmp_path / "task",
        f"FROM {PINNED_BASE}\n" + ("COPY seed /seed\n" if with_context else ""),
        {"seed": "dummy-marker"} if with_context else None,
    )
    allocations = []

    def no_space(*args, **kwargs):
        allocations.append(True)
        raise OSError(errno.ENOSPC, "dummy staging has no inodes")

    monkeypatch.setattr(content.tempfile, "TemporaryDirectory", no_space)
    sandbox = await _start(
        monkeypatch,
        Client(server),
        tmp_path / "trial",
        image=None,
        dockerfile=dockerfile,
    )
    if with_context:
        assert _record(sandbox)["snapshot"] == "off"
        assert "captured" in _record(sandbox)["snapshot_reason"]
        assert allocations == [True]
        assert (dockerfile.parent / "seed").read_text() == "dummy-marker"
    else:
        assert _record(sandbox)["snapshot"] == "created"
        assert allocations == []
