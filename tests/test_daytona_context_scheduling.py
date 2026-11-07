"""Bounded hash jobs preserve fork and concurrent-create behavior."""

from __future__ import annotations

import asyncio
import inspect
import io
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("daytona")

import aiofiles.threadpool
from daytona import AsyncDaytona, CreateSandboxFromImageParams, Image
from daytona._async.object_storage import AsyncObjectStorage
from daytona._async.snapshot import AsyncSnapshotService

from benchflow.sandbox import daytona_context

_SDK_HASH = inspect.unwrap(AsyncObjectStorage._compute_hash_for_path_md5)


@pytest.mark.parametrize("workload", ["equal", "mixed", "small-deadlines"])
def test_forty_creates_keep_throughput_and_small_deadlines(
    tmp_path, monkeypatch, workload
):
    """Review 2: equal/mixed contexts share workers between bounded jobs."""
    large = tmp_path / "large"
    large.write_bytes(b"x" * (2 * 1024 * 1024))
    small = tmp_path / "small"
    small.write_bytes(b"x" * 1024)
    credentials = SimpleNamespace(
        storage_url="https://invalid.example",
        access_key="dummy",
        secret="dummy",
        session_token="dummy",
        bucket="dummy",
        region="us-east-1",
        organization_id="review",
    )
    api = SimpleNamespace(get_push_access=AsyncMock(return_value=credentials))
    monkeypatch.setattr(AsyncObjectStorage, "__init__", lambda *a, **kw: None)
    checks = AsyncMock(return_value=True)
    monkeypatch.setattr(AsyncObjectStorage, "_file_exists_in_s3", checks)

    class ByteCostReader(io.BufferedReader):
        def read(self, size=-1):
            data = super().read(size)
            time.sleep(len(data) / (2 * 1024 * 1024))
            return data

    def slow_open(path, mode="r", *args, **kwargs):
        return ByteCostReader(io.FileIO(path, mode))

    monkeypatch.setattr(daytona_context, "open", slow_open, raising=False)
    monkeypatch.setattr(aiofiles.threadpool, "sync_open", slow_open)

    async def run():
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=20)
        )
        client = object.__new__(AsyncDaytona)

        async def create_after_context(params, **kwargs):
            return await AsyncSnapshotService.process_image_context(api, params.image)

        client._create = create_after_context

        async def create(source, timeout=0):
            image = Image.base("alpine").add_local_file(str(source), "/seed")
            started = time.monotonic()
            await client.create(
                CreateSandboxFromImageParams(image=image, language="python"),
                timeout=timeout,
            )
            return time.monotonic() - started

        started = time.monotonic()
        if workload == "small-deadlines":
            big = [asyncio.create_task(create(large)) for _ in range(4)]
            await asyncio.sleep(0.08)
            little = await asyncio.gather(*(create(small, 0.2) for _ in range(36)))
            assert max(little) < 0.2
            await asyncio.gather(*big)
        else:
            sources = [
                large if workload == "equal" or i % 2 == 0 else small for i in range(40)
            ]
            await asyncio.gather(*(create(source) for source in sources))
        return time.monotonic() - started

    results = {}
    for arm in ("base", "branch"):
        monkeypatch.setattr(AsyncObjectStorage, "_compute_hash_for_path_md5", _SDK_HASH)
        if arm == "branch":
            daytona_context.patch_context_hashing()
        checks.reset_mock()
        results[arm] = asyncio.run(run())
        assert checks.await_count == 40
    # The rejected four-context pool took 4.8x as long on this exact byte cost.
    assert results["branch"] < results["base"] * 1.75


def test_completed_hashes_do_not_leave_dead_workers_after_fork(tmp_path):
    """Review 2: a fresh child loop hashes after parent workers exit."""
    source = tmp_path / "seed"
    source.write_bytes(b"x" * 16384)
    script = r"""
import asyncio
import io
import json
import os
import sys
import threading
from unittest.mock import patch
from daytona._async.object_storage import AsyncObjectStorage
from benchflow.sandbox import daytona_context as dc
source = sys.argv[1]
dc.patch_context_hashing()
storage = object.__new__(AsyncObjectStorage)
barrier = threading.Barrier(4)
class Reader(io.BufferedReader):
    first = True
    def read(self, size=-1):
        if self.first:
            self.first = False
            barrier.wait(5)
        return super().read(size)
async def once():
    return await storage._compute_hash_for_path_md5(source, "seed")
async def prime():
    return await asyncio.gather(*(once() for _ in range(4)))
with patch.object(dc, "open", lambda p, m: Reader(io.FileIO(p, m)), create=True):
    expected = asyncio.run(prime())[0]
read_fd, write_fd = os.pipe()
pid = os.fork()
if pid == 0:
    os.close(read_fd)
    async def child():
        return [await asyncio.wait_for(once(), .5) for _ in range(2)]
    try:
        values = asyncio.run(child())
        os.write(write_fd, json.dumps(values).encode())
    finally:
        os.close(write_fd)
        os._exit(0)
os.close(write_fd)
with os.fdopen(read_fd) as receipt:
    values = json.load(receipt)
_, status = os.waitpid(pid, 0)
assert status == 0
assert values == [expected, expected]
print(json.dumps(values))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(source)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)) == 2
