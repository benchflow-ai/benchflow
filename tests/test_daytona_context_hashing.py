"""Cancellable hashing preserves the SDK's upload identity and errors."""

from __future__ import annotations

import asyncio
import inspect
import io
import os
import threading
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("daytona")

from daytona._async.object_storage import AsyncObjectStorage
from daytona._sync.object_storage import ObjectStorage

from benchflow.sandbox import daytona_context
from benchflow.sandbox.daytona_build_keys import context_hash


async def _SDK_HASH(self, path_str, archive_base_path=None):
    """The SDK's context hash as BenchFlow installs it (#35, _sdk_ops): the
    SDK's function line for line with each folder listed sorted, so a key is
    the same on every filesystem (daytona_build_keys.context_hash)."""
    return context_hash(path_str, archive_base_path, self.compute_archive_base_path)


@pytest.fixture
def storage(monkeypatch):
    monkeypatch.setattr(AsyncObjectStorage, "_compute_hash_for_path_md5", _SDK_HASH)
    return object.__new__(AsyncObjectStorage)


@pytest.mark.parametrize(
    "size", [0, 8191, 8192, 8193, 98304, 1048575, 1048576, 1048577, 2097152]
)
@pytest.mark.parametrize("archive", [None, "context/seed"])
async def test_file_hash_matches_sdk_across_read_boundaries(
    storage, tmp_path, size, archive
):
    """Empty and multi-block files keep existing upload keys."""
    path = tmp_path / "seed"
    path.write_bytes((bytes(range(256)) * (size // 256 + 1))[:size])
    expected = await storage._compute_hash_for_path_md5(str(path), archive)
    daytona_context.patch_context_hashing()
    assert await storage._compute_hash_for_path_md5(str(path), archive) == expected


@pytest.mark.parametrize("archive", [None, "", "context/tree"])
async def test_tree_hash_keeps_sdk_order_links_and_empty_directories(
    storage, tmp_path, archive
):
    """Directory traversal retains the SDK's exact identity."""
    root = tmp_path / "tree"
    root.mkdir()
    (root / "z").write_bytes(b"z" * 9000)
    (root / "a").write_bytes(b"a")
    (root / "empty").mkdir()
    (root / "nested").mkdir()
    (root / "nested" / "seed-é").write_bytes(b"nested")
    (root / "link").symlink_to("a")
    (root / "dir-link").symlink_to("nested", target_is_directory=True)
    os.link(root / "a", root / "hard-link")
    expected = await storage._compute_hash_for_path_md5(str(root), archive)
    daytona_context.patch_context_hashing()
    assert await storage._compute_hash_for_path_md5(str(root), archive) == expected


async def test_upload_reuses_existing_key_and_rehashes_changed_bytes(
    storage, monkeypatch, tmp_path
):
    """A warm upload stays warm; edits cannot reuse stale hashes."""
    path = tmp_path / "seed"
    path.write_bytes(b"before")
    stamp = path.stat()
    cached = {}

    async def exists(key):
        return key in cached

    async def upload(key, source, archive):
        cached[key] = path.read_bytes()

    monkeypatch.setattr(storage, "_file_exists_in_s3", exists)
    uploaded = AsyncMock(side_effect=upload)
    monkeypatch.setattr(storage, "_upload_as_tar", uploaded)
    before = await storage.upload(str(path), "organization", "seed")
    daytona_context.patch_context_hashing()
    assert await storage.upload(str(path), "organization", "seed") == before
    assert uploaded.await_count == 1
    path.write_bytes(b"edited")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    after = await storage.upload(str(path), "organization", "seed")
    assert after != before
    assert cached[f"organization/{after}/context.tar"] == b"edited"
    assert uploaded.await_count == 2


async def test_unreadable_context_keeps_the_sdk_error(storage, monkeypatch, tmp_path):
    """A failed file read never checks or uploads a cache key."""
    (tmp_path / "dangling").symlink_to("missing")
    exists = AsyncMock()
    uploaded = AsyncMock()
    monkeypatch.setattr(storage, "_file_exists_in_s3", exists)
    monkeypatch.setattr(storage, "_upload_as_tar", uploaded)
    with pytest.raises(FileNotFoundError) as before:
        await storage.upload(str(tmp_path), "organization", "tree")
    daytona_context.patch_context_hashing()
    with pytest.raises(FileNotFoundError) as after:
        await storage.upload(str(tmp_path), "organization", "tree")
    assert after.value.args == before.value.args
    assert after.value.filename == before.value.filename
    exists.assert_not_awaited()
    uploaded.assert_not_awaited()


@pytest.mark.parametrize("cancel_count", [0, 1, 2])
@pytest.mark.parametrize("read_fails", [False, True])
async def test_worker_stops_after_cancellation_without_checking_cache(
    storage, monkeypatch, tmp_path, cancel_count, read_fails
):
    """Review 2: cancellation returns before a blocked read finishes."""
    path = tmp_path / "seed"
    path.write_bytes(b"input")
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    threads = []

    class BlockedReader(io.BufferedReader):
        def read(self, size=-1):
            threads.append(threading.get_ident())
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(10), "test did not release the context reader"
            assert path.exists(), "context was removed before its reader finished"
            if read_fails:
                raise OSError("context read failed")
            return super().read(size)

        def close(self):
            try:
                super().close()
            finally:
                finished.set()

    monkeypatch.setattr(
        daytona_context,
        "open",
        lambda *args: BlockedReader(io.FileIO(path, "r")),
        raising=False,
    )
    daytona_context.patch_context_hashing()
    exists = AsyncMock(return_value=True)
    monkeypatch.setattr(storage, "_file_exists_in_s3", exists)
    task = asyncio.create_task(storage.upload(str(path), "organization", "seed"))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert len(threads) == 1
        assert threads[0] != threading.get_ident()
        for _ in range(cancel_count):
            task.cancel()
            await asyncio.sleep(0)
        if cancel_count:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 0.2)
            assert not finished.is_set()
        exists.assert_not_awaited()
    finally:
        release.set()
    if cancel_count:
        with pytest.raises(asyncio.CancelledError):
            await task
        exists.assert_not_awaited()
    elif read_fails:
        with pytest.raises(OSError, match="context read failed"):
            await task
        exists.assert_not_awaited()
    else:
        assert await task
        exists.assert_awaited_once()
    async with asyncio.timeout(5):
        while not finished.is_set():
            await asyncio.sleep(0.001)
    path.unlink()


def test_patch_is_idempotent(storage):
    """Loading the SDK again cannot stack hash workers."""
    daytona_context.patch_context_hashing()
    patched = AsyncObjectStorage._compute_hash_for_path_md5
    assert patched is not _SDK_HASH
    daytona_context.patch_context_hashing()
    assert AsyncObjectStorage._compute_hash_for_path_md5 is patched
    assert inspect.unwrap(patched) is _SDK_HASH


@pytest.mark.parametrize("change", ["version", "async-layout", "sync-layout"])
def test_unfamiliar_sdk_keeps_its_own_hash(storage, monkeypatch, change):
    """Unknown SDK contracts retain the original implementation."""
    if change == "version":
        monkeypatch.setattr(daytona_context, "version", lambda _: "0.208.0")
    elif change == "async-layout":

        async def different(self, path):
            return "original"

        monkeypatch.setattr(AsyncObjectStorage, "_compute_hash_for_path_md5", different)
    else:
        monkeypatch.setattr(ObjectStorage, "_compute_hash_for_path_md5", lambda _: "")
    original = AsyncObjectStorage._compute_hash_for_path_md5
    daytona_context.patch_context_hashing()
    assert AsyncObjectStorage._compute_hash_for_path_md5 is original
