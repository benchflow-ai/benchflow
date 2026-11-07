"""Cancellation, executor isolation and staged-reader lifetime."""

from __future__ import annotations

import asyncio
import builtins
import gc
import inspect
import io
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("daytona")

import aiofiles.threadpool
from daytona import (
    AsyncDaytona,
    CreateSandboxFromImageParams,
    CreateSnapshotParams,
    Image,
    Resources,
)
from daytona._async.object_storage import AsyncObjectStorage
from daytona._async.snapshot import AsyncSnapshotService

from benchflow.sandbox import daytona_content_snapshots as content
from benchflow.sandbox import daytona_context
from benchflow.sandbox.daytona_layers import LayeredRecipe
from benchflow.sandbox.daytona_strategies import _DaytonaDirect

_SDK_HASH = inspect.unwrap(AsyncObjectStorage._compute_hash_for_path_md5)


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    monkeypatch.setattr(AsyncObjectStorage, "_compute_hash_for_path_md5", _SDK_HASH)
    daytona_context.patch_context_hashing()
    source = tmp_path / "seed"
    source.write_bytes(b"x" * (2 * 1024 * 1024))
    image = Image.base("alpine@sha256:" + "1" * 64).add_local_file(str(source), "/seed")
    storage = object.__new__(AsyncObjectStorage)
    checks = AsyncMock(return_value=True)
    monkeypatch.setattr(AsyncObjectStorage, "_file_exists_in_s3", checks)
    monkeypatch.setattr(AsyncObjectStorage, "_upload_as_tar", AsyncMock())
    monkeypatch.setattr(AsyncObjectStorage, "__init__", lambda *a, **kw: None)
    service = object.__new__(AsyncSnapshotService)
    credentials = SimpleNamespace(
        storage_url="https://invalid.example",
        access_key="dummy",
        secret="dummy",
        session_token="dummy",
        bucket="dummy",
        region="us-east-1",
        organization_id="review",
    )
    service._AsyncSnapshotService__object_storage_api = SimpleNamespace(
        get_push_access=AsyncMock(return_value=credentials)
    )
    service._AsyncSnapshotService__default_region_id = None
    provider_create = AsyncMock(
        side_effect=AssertionError("no provider create expected")
    )
    service._AsyncSnapshotService__snapshots_api = SimpleNamespace(
        create_snapshot=provider_create
    )

    class Missing(Exception):
        status_code = 404

    service.get = AsyncMock(side_effect=Missing())
    return SimpleNamespace(
        source=source,
        image=image,
        storage=storage,
        service=service,
        checks=checks,
        provider_create=provider_create,
    )


@pytest.fixture
async def slow_reads(monkeypatch):
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    reads = []

    class SlowRaw(io.FileIO):
        def readinto(self, buffer):
            loop.call_soon_threadsafe(entered.set)
            time.sleep(len(buffer) / 8192 * 0.005)
            reads.append(len(buffer))
            return super().readinto(buffer)

    def slow_open(path, mode="r", *args, **kwargs):
        return io.BufferedReader(SlowRaw(path, mode))

    monkeypatch.setattr(daytona_context, "open", slow_open, raising=False)
    monkeypatch.setattr(aiofiles.threadpool, "sync_open", slow_open)
    return entered, reads


async def test_cancel_content_snapshot_does_not_fall_back_to_image_create(
    sdk, slow_reads, monkeypatch
):
    """Review: operator cancellation cannot become snapshot fallback."""
    entered, reads = slow_reads
    monkeypatch.setattr(content, "enabled", lambda: True)
    monkeypatch.setattr(content, "owner_prefix", lambda: "en229review")
    env = SimpleNamespace(
        _install_layers_record={},
        task_env_config=SimpleNamespace(build_timeout_sec=1.0),
        logger=Mock(),
    )
    strategy = _DaytonaDirect(env)
    image_create = AsyncMock()

    async def start():
        snapshot = await strategy._content_snapshot(
            SimpleNamespace(snapshot=sdk.service),
            sdk.image,
            Resources(cpu=1, memory=1, disk=1),
            LayeredRecipe(),
            build_allowance_sec=0,
            force_build=False,
            fresh=False,
        )
        if snapshot is None:
            await image_create()

    attempt = asyncio.create_task(start())
    await asyncio.wait_for(entered.wait(), 5)
    await asyncio.sleep(0.02)
    started = time.monotonic()
    attempt.cancel()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    assert time.monotonic() - started < 0.8
    assert len(reads) < 20
    assert "snapshot" not in env._install_layers_record
    assert "snapshot_reason" not in env._install_layers_record
    image_create.assert_not_awaited()
    sdk.provider_create.assert_not_awaited()
    sdk.checks.assert_not_awaited()


@pytest.mark.parametrize("nested", [False, True])
async def test_snapshot_deadline_keeps_its_error_and_stops_reading(
    sdk, slow_reads, nested
):
    """Review: nested deadlines retain their SDK/base error messages."""
    _, reads = slow_reads
    params = CreateSnapshotParams(name="review", image=sdk.image)
    started = time.monotonic()
    if nested:
        with pytest.raises(content.SnapshotFallback) as error:
            await content.ensure_snapshot(
                SimpleNamespace(snapshot=sdk.service),
                "review",
                lambda: params,
                timeout_sec=0.05,
            )
        assert str(error.value) == "snapshot review was not active within 0s"
    else:
        from daytona import DaytonaTimeoutError

        with pytest.raises(DaytonaTimeoutError) as error:
            await sdk.service.create(params, timeout=0.05)
        assert (
            str(error.value)
            == "Failed to create snapshot: Function 'create' exceeded timeout of 0.05 seconds."
        )
    assert time.monotonic() - started < 0.8
    assert len(reads) < 100
    sdk.provider_create.assert_not_awaited()
    sdk.checks.assert_not_awaited()


async def test_forty_hashes_leave_dns_and_default_executor_available(sdk, monkeypatch):
    """Review 2: bounded jobs leave DNS and executor work responsive."""
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    calls = []
    original_open = builtins.open

    class SlowReader(io.BufferedReader):
        def read(self, size=-1):
            calls.append(threading.get_ident())
            loop.call_soon_threadsafe(entered.set)
            data = super().read(size)
            time.sleep(len(data) / (2 * 1024 * 1024))
            return data

    def blocked_open(path, mode="r", *args, **kwargs):
        if mode == "rb" and str(path) == str(sdk.source):
            return SlowReader(io.FileIO(path, mode))
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", blocked_open)
    jobs = [
        asyncio.create_task(sdk.storage.upload(str(sdk.source), "review", "seed"))
        for _ in range(40)
    ]
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert (
            await asyncio.wait_for(asyncio.to_thread(lambda: "available"), 1)
            == "available"
        )
        assert await asyncio.wait_for(loop.getaddrinfo("localhost", 80), 1)
        for job in jobs:
            job.cancel()
        await asyncio.wait_for(asyncio.gather(*jobs, return_exceptions=True), 1)
        assert all(job.cancelled() for job in jobs)
    finally:
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
    assert calls
    sdk.checks.assert_not_awaited()


def test_runner_shutdown_keeps_staging_until_physical_reader_exits(sdk, monkeypatch):
    """Review: asyncio.run shutdown cannot release a live reader's staging."""
    entered = threading.Event()
    cleanup = threading.Event()
    finished = threading.Event()
    observed = {}

    class PausedReader(io.BufferedReader):
        def read(self, size=-1):
            entered.set()
            cleanup.wait(0.1)
            observed["exists_at_resume"] = Path(self.name).exists()
            return super().read(size)

        def close(self):
            try:
                super().close()
            finally:
                finished.set()

    monkeypatch.setattr(
        daytona_context,
        "open",
        lambda path, mode: PausedReader(io.FileIO(path, mode)),
        raising=False,
    )

    async def main():
        async def owner():
            try:
                async with content.captured_image_async(sdk.image) as staged:
                    entry = staged._context_list[0]
                    observed["path"] = entry.source_path
                    await sdk.storage.upload(
                        entry.source_path, "review", entry.archive_path
                    )
            finally:
                observed["cancel_before_reader_finished"] = not finished.is_set()
                observed["exists_at_cancel"] = Path(observed["path"]).exists()
                cleanup.set()

        task = asyncio.create_task(owner())
        while not entered.is_set():
            await asyncio.sleep(0.001)
        assert not task.done()
        # Returning makes the runner cancel every remaining Task, not just owner.

    asyncio.run(main())
    assert cleanup.is_set()
    assert finished.is_set()
    assert observed["exists_at_resume"] is True
    assert observed["cancel_before_reader_finished"] is True
    assert observed["exists_at_cancel"] is True
    assert Path(observed["path"]).exists() is False
    sdk.checks.assert_not_awaited()


def _plain_client(sdk):
    client = object.__new__(AsyncDaytona)
    client.default_language = "python"
    client._target = None
    client._object_storage_api = sdk.service._AsyncSnapshotService__object_storage_api
    client._sandbox_api = SimpleNamespace(create_sandbox=sdk.provider_create)
    return client


@pytest.mark.parametrize("strategy", [False, True])
async def test_cancel_returns_while_read_outlives_sdk_deadline(
    sdk, monkeypatch, strategy
):
    """Review 2: blocked raw I/O cannot turn cancellation into fallback."""
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    paths = []
    sizes = []

    class BlockedRaw(io.FileIO):
        def readinto(self, buffer):
            sizes.append(len(buffer))
            paths.append(Path(self.name))
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
            assert Path(self.name).exists(), "staging released under a physical read"
            return super().readinto(buffer)

        def close(self):
            super().close()
            finished.set()

    monkeypatch.setattr(
        daytona_context,
        "open",
        lambda p, m: io.BufferedReader(BlockedRaw(p, m)),
        raising=False,
    )
    monkeypatch.setattr(content, "enabled", lambda: True)
    monkeypatch.setattr(content, "owner_prefix", lambda: "en229review")
    env = SimpleNamespace(
        _install_layers_record={},
        task_env_config=SimpleNamespace(build_timeout_sec=1.0),
        logger=Mock(),
    )
    image_create = AsyncMock()

    async def start():
        if strategy:
            snapshot = await _DaytonaDirect(env)._content_snapshot(
                SimpleNamespace(snapshot=sdk.service),
                sdk.image,
                Resources(cpu=1, memory=1, disk=1),
                LayeredRecipe(),
                build_allowance_sec=0,
                force_build=False,
                fresh=False,
            )
            if snapshot is None:
                await image_create()
        else:
            await _plain_client(sdk).create(
                CreateSandboxFromImageParams(image=sdk.image), timeout=1
            )

    attempt = asyncio.create_task(start())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        attempt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(attempt, 0.2)
        assert not finished.is_set()
        assert paths[0].exists()
        assert sizes == [8192]
        assert "snapshot" not in env._install_layers_record
        assert "snapshot_reason" not in env._install_layers_record
        # Keep the real read pending beyond the SDK deadline, after cancellation.
        await asyncio.sleep(1.05)
        assert not finished.is_set()
        assert paths[0].exists()
    finally:
        release.set()
        await asyncio.gather(attempt, return_exceptions=True)
        async with asyncio.timeout(5):
            while not finished.is_set() or (strategy and paths[0].exists()):
                await asyncio.sleep(0.001)
    image_create.assert_not_awaited()
    sdk.provider_create.assert_not_awaited()
    sdk.checks.assert_not_awaited()


async def test_fifo_open_does_not_hold_create_cancellation(sdk, monkeypatch, tmp_path):
    """Review 2: a real FIFO open may outlive its cancelled create."""
    root = tmp_path / "fifo-context"
    root.mkdir()
    fifo = root / "pipe"
    os.mkfifo(fifo)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    closed = threading.Event()
    original_open = builtins.open

    def observed_open(path, *args, **kwargs):
        if str(path) == str(fifo):
            loop.call_soon_threadsafe(entered.set)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(daytona_context, "open", observed_open, raising=False)
    image = Image.base("alpine").add_local_dir(str(root), "/context")
    attempt = asyncio.create_task(
        _plain_client(sdk).create(CreateSandboxFromImageParams(image=image), timeout=1)
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        attempt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(attempt, 0.2)
    finally:

        def writer():
            try:
                with original_open(fifo, "wb") as out:
                    out.write(b"x")
            except BrokenPipeError:
                pass
            finally:
                closed.set()

        await asyncio.wait_for(asyncio.to_thread(writer), 5)
        await asyncio.gather(attempt, return_exceptions=True)
    assert closed.is_set()
    sdk.provider_create.assert_not_awaited()


async def test_cancelled_failed_read_consumes_every_exception_receipt(sdk, monkeypatch):
    """Review 2: physical completion before callback delivery stays quiet."""
    loop = asyncio.get_running_loop()
    diagnostics = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, message: diagnostics.append(message))
    entered = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()

    class FailedReader(io.BufferedReader):
        def read(self, size=-1):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
            try:
                raise OSError("read failed concurrently with cancellation")
            finally:
                finished.set()

    monkeypatch.setattr(
        daytona_context,
        "open",
        lambda p, m: FailedReader(io.FileIO(p, m)),
        raising=False,
    )
    task = asyncio.create_task(
        sdk.storage._compute_hash_for_path_md5(str(sdk.source), "seed")
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        release.set()
        assert finished.wait(5)
        time.sleep(0.02)
        with pytest.raises(asyncio.CancelledError):
            await task
        del task
        for _ in range(3):
            await asyncio.sleep(0.02)
            gc.collect()
        assert diagnostics == []
    finally:
        release.set()
        loop.set_exception_handler(previous)
