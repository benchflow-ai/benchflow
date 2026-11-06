"""Failed executor submissions have one cleanup owner."""

from __future__ import annotations

import asyncio
import inspect
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("daytona")

from daytona import Image
from daytona._async.object_storage import AsyncObjectStorage

from benchflow.sandbox import daytona_content_snapshots as content
from benchflow.sandbox import daytona_context

_SDK_HASH = inspect.unwrap(AsyncObjectStorage._compute_hash_for_path_md5)


def _open_staged_fds(path):
    descriptors = []
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = str(fd.readlink())
        except FileNotFoundError:
            continue
        if target in (path, path + " (deleted)"):
            descriptors.append(fd.name)
    return descriptors


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize(
    "accepted", ["queued", "running", "rejected", "broken", "cancelled"]
)
def test_failed_submission_releases_captured_upload_with_retained_error(
    tmp_path, monkeypatch, batch, accepted
):
    """Failed submissions and completions release custody once."""
    monkeypatch.setattr(AsyncObjectStorage, "_compute_hash_for_path_md5", _SDK_HASH)
    daytona_context.patch_context_hashing()
    source = tmp_path / "seed"
    source.write_bytes(b"x" * 2 * 1024 * 1024)
    image = Image.base("alpine@sha256:" + "1" * 64).add_local_file(str(source), "/seed")
    storage = object.__new__(AsyncObjectStorage)
    checks = AsyncMock(return_value=True)
    uploaded = AsyncMock()
    monkeypatch.setattr(storage, "_file_exists_in_s3", checks)
    monkeypatch.setattr(storage, "_upload_as_tar", uploaded)
    errors = []
    releases = []
    readers = []
    closes = []
    release_blocker = threading.Event()
    blocker_entered = threading.Event()
    release_read = threading.Event()
    read_entered = threading.Event()
    state = {"batches": 0, "armed": False, "failures": 0}
    injected = RuntimeError("can't start new thread")
    original_start = threading.Thread.start
    original_finished = daytona_context._ContextReaders.finished

    def finished(self):
        releases.append(self)
        original_finished(self)

    monkeypatch.setattr(daytona_context._ContextReaders, "finished", finished)

    class ObservedReader(io.BufferedReader):
        def read(self, size=-1):
            if state["armed"] and accepted == "running":
                read_entered.set()
                assert release_read.wait(10)
            return super().read(size)

        def close(self):
            if not self.closed:
                closes.append(self)
            super().close()

    def observed_open(path, mode):
        reader = ObservedReader(io.FileIO(path, mode))
        readers.append(reader)
        return reader

    monkeypatch.setattr(daytona_context, "open", observed_open, raising=False)

    def start(thread, *args, **kwargs):
        if (
            state["armed"]
            and state["failures"] == 0
            and thread.name.startswith("en229-submit")
        ):
            state["failures"] += 1
            if accepted == "running":
                # The existing worker takes the queued job before submit raises.
                release_blocker.set()
                assert read_entered.wait(5)
            raise injected
        return original_start(thread, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "start", start)

    async def run():
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="en229-submit")
        loop.set_default_executor(pool)
        original_submit = loop.run_in_executor
        blockers = []
        fault_pools = []

        def blocker():
            blocker_entered.set()
            assert release_blocker.wait(15)

        def broken_initializer():
            raise RuntimeError("initializer rejected")

        def submit(executor, function, *args):
            if (
                isinstance(
                    getattr(function, "__self__", None), daytona_context._HashReader
                )
                and function.__name__ == "_batch"
            ):
                state["batches"] += 1
                if state["batches"] == batch:
                    # A fresh pool has no spare worker or stale idle receipt.
                    fault = ThreadPoolExecutor(
                        max_workers=1 if accepted in ("broken", "cancelled") else 2,
                        thread_name_prefix="en229-submit",
                        initializer=broken_initializer
                        if accepted == "broken"
                        else None,
                    )
                    fault_pools.append(fault)
                    loop.set_default_executor(fault)
                    if accepted in ("broken", "cancelled"):
                        if accepted == "cancelled":
                            blockers.append(original_submit(None, blocker))
                            assert blocker_entered.wait(5)
                        completion = original_submit(executor, function, *args)

                        def restore(future):
                            if not future.cancelled():
                                state["submission_error"] = future.exception()
                            loop.set_default_executor(pool)

                        completion.add_done_callback(restore)
                        if accepted == "cancelled":
                            fault.shutdown(wait=False, cancel_futures=True)
                        return completion
                    blockers.append(original_submit(None, blocker))
                    assert blocker_entered.wait(5)
                    if accepted == "rejected":
                        fault.shutdown(wait=False)
                    else:
                        state["armed"] = True
                    try:
                        return original_submit(executor, function, *args)
                    except RuntimeError as error:
                        state["submission_error"] = error
                        if accepted == "rejected":
                            loop.set_default_executor(pool)
                        raise
            return original_submit(executor, function, *args)

        monkeypatch.setattr(loop, "run_in_executor", submit)
        try:
            try:
                async with content.captured_image_async(image) as staged:
                    path = staged._context_list[0].source_path
                    await storage.upload(path, "review", "seed")
            except (RuntimeError, asyncio.CancelledError) as error:
                errors.append(error)
            assert len(errors) == 1
            if accepted == "cancelled":
                assert isinstance(errors[0], asyncio.CancelledError)
                assert errors[0].args == ()
            else:
                assert errors[0] is state["submission_error"]
            if accepted in ("queued", "running"):
                assert errors[0] is injected
                assert state["failures"] == 1
            elif accepted == "rejected":
                assert str(errors[0]) == "cannot schedule new futures after shutdown"
            elif accepted == "broken":
                assert type(errors[0]).__name__ == "BrokenThreadPool"
            if accepted == "running":
                assert Path(path).exists()
                assert _open_staged_fds(path)
                assert releases == []
            # Worker creation recovers; orphaned queue entries can now drain.
            assert await asyncio.to_thread(lambda: "recovered") == "recovered"
            release_read.set()
            release_blocker.set()
            await asyncio.gather(*blockers)
            async with asyncio.timeout(5):
                while Path(path).exists() or _open_staged_fds(path):
                    await asyncio.sleep(0.001)
            return path
        finally:
            release_read.set()
            release_blocker.set()
            await asyncio.gather(*blockers)
            pool.shutdown(wait=True)
            for fault in fault_pools:
                fault.shutdown(wait=True)

    path = asyncio.run(run())
    assert errors[0].__traceback__ is not None
    assert len(releases) == 1
    assert not Path(path).exists()
    assert _open_staged_fds(path) == []
    expected_readers = 0 if batch == 1 and accepted != "running" else 1
    assert len(readers) == len(closes) == expected_readers
    assert all(reader.closed for reader in readers)
    checks.assert_not_awaited()
    uploaded.assert_not_awaited()


@pytest.mark.parametrize("accepted", ["submission", "broken", "cancelled"])
def test_failed_cleanup_submission_preserves_cancellation(
    tmp_path, monkeypatch, accepted
):
    """Failed cleanup jobs release once and preserve cancellation."""
    monkeypatch.setattr(AsyncObjectStorage, "_compute_hash_for_path_md5", _SDK_HASH)
    daytona_context.patch_context_hashing()
    source = tmp_path / "seed"
    source.write_bytes(b"x" * 2 * 1024 * 1024)
    image = Image.base("alpine@sha256:" + "1" * 64).add_local_file(str(source), "/seed")
    storage = object.__new__(AsyncObjectStorage)
    checks = AsyncMock(return_value=True)
    monkeypatch.setattr(storage, "_file_exists_in_s3", checks)
    release = threading.Event()
    entered = threading.Event()
    state = {"armed": False, "failures": 0}
    stages = []
    releases = []
    original_start = threading.Thread.start
    original_finished = daytona_context._ContextReaders.finished

    def finished(self):
        releases.append(self)
        original_finished(self)

    monkeypatch.setattr(daytona_context._ContextReaders, "finished", finished)

    def start(thread, *args, **kwargs):
        if (
            state["armed"]
            and not state["failures"]
            and thread.name.startswith("en229-close")
        ):
            state["failures"] += 1
            raise RuntimeError("can't start new thread")
        return original_start(thread, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "start", start)

    async def run():
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor(max_workers=2)
        loop.set_default_executor(pool)
        original_submit = loop.run_in_executor
        blockers = []
        fault_pools = []

        def blocker():
            entered.set()
            assert release.wait(15)

        def broken_initializer():
            raise RuntimeError("initializer rejected")

        def submit(executor, function, *args):
            reader = isinstance(
                getattr(function, "__self__", None), daytona_context._HashReader
            )
            if reader and function.__name__ == "_close":
                fault = ThreadPoolExecutor(
                    max_workers=2 if accepted == "submission" else 1,
                    thread_name_prefix="en229-close",
                    initializer=broken_initializer if accepted == "broken" else None,
                )
                fault_pools.append(fault)
                loop.set_default_executor(fault)
                if accepted != "broken":
                    blockers.append(original_submit(None, blocker))
                    assert entered.wait(5)
                state["armed"] = accepted == "submission"
                try:
                    completion = original_submit(executor, function, *args)
                finally:
                    loop.set_default_executor(pool)
                if accepted == "cancelled":
                    fault.shutdown(wait=False, cancel_futures=True)
                return completion
            completion = original_submit(executor, function, *args)
            if reader and function.__name__ == "_batch":
                # Cancel after the physical batch suspended its still-open file.
                completion.add_done_callback(
                    lambda _: owner.cancel("operator requested")
                )
            return completion

        monkeypatch.setattr(loop, "run_in_executor", submit)

        async def upload():
            async with content.captured_image_async(image) as staged:
                path = staged._context_list[0].source_path
                stages.append(path)
                await storage.upload(path, "review", "seed")

        owner = asyncio.create_task(upload())
        try:
            with pytest.raises(asyncio.CancelledError) as error:
                await owner
            assert error.value.args == ("operator requested",)
            assert state["failures"] == (1 if accepted == "submission" else 0)
            assert await asyncio.to_thread(lambda: "recovered") == "recovered"
            async with asyncio.timeout(5):
                while Path(stages[0]).exists() or _open_staged_fds(stages[0]):
                    await asyncio.sleep(0.001)
        finally:
            release.set()
            await asyncio.gather(*blockers)
            pool.shutdown(wait=True)
            for fault in fault_pools:
                fault.shutdown(wait=True)

    asyncio.run(run())
    assert len(releases) == 1
    assert not Path(stages[0]).exists()
    assert _open_staged_fds(stages[0]) == []
    checks.assert_not_awaited()
