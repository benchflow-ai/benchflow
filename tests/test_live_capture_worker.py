"""Guards commit a821eb82b: C6 event-loop heartbeat and capture shutdown."""

import asyncio
import contextlib
import threading
import time

from benchflow.trajectories._llm_capture import LiveLLMTrajectoryWriter
from tests.test_llm_capture_memory import _chat_log, _process


async def _manual_process(tmp_path):
    log = tmp_path / "callback.jsonl"
    _chat_log(log, records=1)
    process = _process(tmp_path, log)
    process.start_live_capture(tmp_path / "trajectory/llm_trajectory.jsonl")
    assert process._live_capture_task is not None
    process._live_capture_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await process._live_capture_task
    process._live_capture_task = None
    return process


async def test_append_leaves_the_event_loop_free(tmp_path, monkeypatch):
    process = await _manual_process(tmp_path)
    original = LiveLLMTrajectoryWriter.append
    writing = threading.Event()
    beats = []

    def slow_append(writer, exchanges):
        writing.set()
        try:
            time.sleep(0.1)
            return original(writer, exchanges)
        finally:
            writing.clear()

    async def heartbeat():
        while True:
            if writing.is_set():
                beats.append(1)
            await asyncio.sleep(0.001)

    monkeypatch.setattr(LiveLLMTrajectoryWriter, "append", slow_append)
    ticker = asyncio.create_task(heartbeat())
    try:
        await process._capture_live_records()
    finally:
        ticker.cancel()
        await asyncio.gather(ticker, return_exceptions=True)
    assert beats, "append blocked the event loop for 100ms"
    assert process.live_usage_tokens() == 10


async def test_cancel_waits_for_writer_before_stop_drains(tmp_path, monkeypatch):
    process = await _manual_process(tmp_path)
    original = LiveLLMTrajectoryWriter.append
    entered = threading.Event()
    release = threading.Event()
    active = 0
    overlap = False

    def blocked_append(writer, exchanges):
        nonlocal active, overlap
        active += 1
        overlap |= active > 1
        entered.set()
        try:
            assert release.wait(5), "test failed to release writer"
            return original(writer, exchanges)
        finally:
            active -= 1

    monkeypatch.setattr(LiveLLMTrajectoryWriter, "append", blocked_append)
    process._live_capture_task = asyncio.create_task(process._capture_live_records())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        stopping = asyncio.create_task(process._stop_live_capture())
        await asyncio.sleep(0.02)
        assert not stopping.done(), "stop must await the outstanding worker"
        # A second cancellation must not detach the worker either.
        process._live_capture_task.cancel()
        await asyncio.sleep(0.02)
        assert not stopping.done()
    finally:
        release.set()
    await stopping
    assert not overlap and active == 0
    path = tmp_path / "trajectory/llm_trajectory.jsonl"
    assert len(path.read_text().splitlines()) == 1
    assert len(process._live_trajectory.exchanges) == 1
    assert process.live_usage_tokens() == 10


async def test_parse_and_request_event_capture_run_on_the_worker(tmp_path, monkeypatch):
    import benchflow.providers.litellm_runtime as runtime

    process = await _manual_process(tmp_path)
    main_thread = threading.get_ident()
    threads = []
    original = runtime.trajectory_from_litellm_callback_log

    def parse(*args, **kwargs):
        threads.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "trajectory_from_litellm_callback_log", parse)
    await process._capture_live_records()
    assert threads and all(thread != main_thread for thread in threads)


async def test_cancel_still_finishes_when_the_worker_fails(tmp_path, monkeypatch):
    process = await _manual_process(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def failed_append(writer, exchanges):
        entered.set()
        assert release.wait(5)
        raise OSError("test disk error")

    monkeypatch.setattr(LiveLLMTrajectoryWriter, "append", failed_append)
    process._live_capture_task = asyncio.create_task(process._live_capture_loop())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        stopping = asyncio.create_task(process._stop_live_capture())
        await asyncio.sleep(0.02)
    finally:
        release.set()
    await asyncio.wait_for(stopping, timeout=2)
    assert process._live_capture_task is None
