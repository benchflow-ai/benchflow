"""Executor terminal states own rejected-reader cleanup."""

from __future__ import annotations

import asyncio
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("daytona")

from daytona import Image
from daytona._async.object_storage import AsyncObjectStorage

from benchflow.sandbox import daytona_content_snapshots as content
from benchflow.sandbox import daytona_context

_SDK_HASH = inspect.unwrap(AsyncObjectStorage._compute_hash_for_path_md5)


def _staged_fds(root):
    descriptors = []
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = str(fd.readlink())
        except FileNotFoundError:
            continue
        if target.startswith(str(root) + "/"):
            descriptors.append(fd.name)
    return descriptors


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("work", ["batch", "close"])
@pytest.mark.parametrize("outcome", ["cancelled", "broken", "runs"])
@pytest.mark.parametrize("closed_loop", [False, True])
def test_terminal_work_releases_after_caller_cancellation(
    tmp_path, monkeypatch, batch, work, outcome, closed_loop
):
    """Late rejection releases once without a caller or live loop."""
    monkeypatch.setattr(AsyncObjectStorage, "_compute_hash_for_path_md5", _SDK_HASH)
    daytona_context.patch_context_hashing()
    source = tmp_path / "seed"
    source.write_bytes(b"x" * 4 * 1024 * 1024)
    image = Image.base("alpine@sha256:" + "1" * 64).add_local_file(str(source), "/seed")
    storage = object.__new__(AsyncObjectStorage)
    checks = AsyncMock(return_value=True)
    uploaded = AsyncMock()
    monkeypatch.setattr(storage, "_file_exists_in_s3", checks)
    monkeypatch.setattr(storage, "_upload_as_tar", uploaded)
    loop = asyncio.new_event_loop()
    healthy = ThreadPoolExecutor(max_workers=2)
    loop.set_default_executor(healthy)
    diagnostics = []
    loop.set_exception_handler(lambda _, detail: diagnostics.append(detail))
    entered = threading.Event()
    release = threading.Event()
    state = {"batches": 0}
    releases = []
    original_finished = daytona_context._ContextReaders.finished

    def finished(self):
        releases.append(self)
        original_finished(self)

    monkeypatch.setattr(daytona_context._ContextReaders, "finished", finished)

    def initializer():
        entered.set()
        assert release.wait(10)
        if outcome == "broken":
            raise RuntimeError("initializer rejected after caller returned")

    fault = ThreadPoolExecutor(max_workers=1, initializer=initializer)
    original_submit = loop.run_in_executor

    def submit(executor, function, *args):
        reader = isinstance(
            getattr(function, "__self__", None), daytona_context._HashReader
        )
        if reader and function.__name__ == "_batch":
            state["batches"] += 1
        target = reader and (
            function.__name__ == "_close"
            or (work == "batch" and state["batches"] == batch)
        )
        if target:
            state["reader"] = function.__self__
            loop.set_default_executor(fault)
            try:
                completion = original_submit(executor, function, *args)
            finally:
                loop.set_default_executor(healthy)
            assert entered.wait(5)
            state["completion"] = completion
            if work == "batch":
                loop.call_soon(state["owner"].cancel, "operator requested")
            return completion
        completion = original_submit(executor, function, *args)
        if (
            reader
            and function.__name__ == "_batch"
            and work == "close"
            and state["batches"] == batch
        ):
            completion.add_done_callback(
                lambda _: state["owner"].cancel("operator requested")
            )
        return completion

    monkeypatch.setattr(loop, "run_in_executor", submit)

    async def upload():
        async with content.captured_image_async(image) as staged:
            path = Path(staged._context_list[0].source_path)
            state["root"] = next(
                parent
                for parent in path.parents
                if parent.name.startswith("benchflow-snapshot-")
            )
            await storage.upload(str(path), "review", "seed")

    async def run():
        state["owner"] = asyncio.create_task(upload())
        with pytest.raises(asyncio.CancelledError) as error:
            await asyncio.wait_for(state["owner"], 2)
        state["error"] = error.value
        assert state["error"].args == ("operator requested",)
        assert state["root"].exists()
        assert len(_staged_fds(state["root"])) == (work == "close" or batch == 2)
        assert not releases

    try:
        loop.run_until_complete(run())
        if closed_loop:
            loop.close()
        # The caller and optionally the loop have exited before rejection.
        if outcome == "cancelled":
            fault.shutdown(wait=False, cancel_futures=True)
        release.set()
        fault.shutdown(wait=True)
        healthy.shutdown(wait=True)
        # Deliberately do not deliver any asyncio callbacks before asserting.
        assert state["error"].__traceback__ is not None
        assert not state["root"].exists()
        assert _staged_fds(state["root"]) == []
        assert len(releases) == 1
        assert state["reader"]._readers._count == 0
        checks.assert_not_awaited()
        uploaded.assert_not_awaited()
        if not closed_loop:
            loop.run_until_complete(asyncio.sleep(0))
        assert diagnostics == []
    finally:
        release.set()
        fault.shutdown(wait=True, cancel_futures=True)
        healthy.shutdown(wait=True)
        # Failed assertions on the previous implementation must not leak fixtures.
        if "reader" in state:
            state["reader"]._submission_failed()
        if not loop.is_closed():
            loop.close()


async def test_nonstandard_loop_uses_original_sdk(tmp_path, monkeypatch):
    """Loops without the stdlib executor contract keep the SDK."""
    monkeypatch.setattr(AsyncObjectStorage, "_compute_hash_for_path_md5", _SDK_HASH)
    daytona_context.patch_context_hashing()
    path = tmp_path / "seed"
    path.write_bytes(b"seed")
    storage = object.__new__(AsyncObjectStorage)
    expected = await _SDK_HASH(storage, str(path), "seed")
    monkeypatch.setattr(asyncio, "BaseEventLoop", type("DifferentLoop", (), {}))
    reader = Mock(side_effect=AssertionError("must use original SDK"))
    monkeypatch.setattr(daytona_context, "_HashReader", reader)
    assert await storage._compute_hash_for_path_md5(str(path), "seed") == expected
    reader.assert_not_called()


@pytest.mark.parametrize("existing_pool", [False, True])
def test_hash_respects_default_executor_shutdown(tmp_path, existing_pool):
    """Fresh and existing pools preserve the shutdown error."""
    path = tmp_path / "seed"
    path.write_bytes(b"seed")

    async def run():
        if existing_pool:
            await asyncio.to_thread(lambda: None)
        await asyncio.get_running_loop().shutdown_default_executor()
        with daytona_context.context_readers() as readers:
            reader = daytona_context._HashReader(None, str(path), "seed")
            with pytest.raises(
                RuntimeError, match=r"^Executor shutdown has been called$"
            ):
                await reader.hash()
            assert readers._count == 0

    asyncio.run(run())
