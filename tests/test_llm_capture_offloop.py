"""Guards the fix for public #1055: capture must not stall the event loop."""

import asyncio
import threading

import pytest

from benchflow.providers import litellm_runtime as runtime
from benchflow.trajectories._llm_capture import LiveLLMTrajectoryWriter
from tests.test_live_llm_trajectory import (
    _arm_capture,
    _callback_line,
    _sandbox_process,
)
from tests.test_llm_capture_memory import _process


async def _heartbeat_during(operation, monkeypatch, target, name):
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original = getattr(target, name)

    def blocking(*args, **kwargs):
        entered.set()
        try:
            release.wait(1)
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(target, name, blocking)

    async def heartbeat():
        while not entered.is_set():
            await asyncio.sleep(0.001)
        active = not finished.is_set()
        release.set()
        return active

    try:
        _, active = await asyncio.wait_for(
            asyncio.gather(operation(), heartbeat()), timeout=5
        )
        assert active, f"{name} blocked the coordinator heartbeat"
    finally:
        release.set()


@pytest.mark.parametrize("stage", ["parse", "append"])
async def test_live_capture_keeps_the_heartbeat_schedulable(
    tmp_path, monkeypatch, stage
):
    log = tmp_path / "callback.jsonl"
    log.write_text(_callback_line())
    process = _process(tmp_path, log)
    output = tmp_path / "llm_trajectory.jsonl"
    await _arm_capture(process, output)
    target, name = (
        (runtime, "trajectory_from_litellm_callback_log")
        if stage == "parse"
        else (LiveLLMTrajectoryWriter, "append")
    )
    await _heartbeat_during(process._capture_live_records, monkeypatch, target, name)
    assert len(output.read_text().splitlines()) == 1
    assert "sk-secret" not in output.read_text()


@pytest.mark.parametrize("backend", ["host", "sandbox-download", "sandbox-exec"])
async def test_final_import_keeps_the_heartbeat_schedulable(
    tmp_path, monkeypatch, backend
):
    text = _callback_line()
    if backend == "host":
        log = tmp_path / "callback.jsonl"
        log.write_text(text)
        process = _process(tmp_path, log)

        async def stable(*args, **kwargs):
            pass

        monkeypatch.setattr(runtime, "_await_log_stable", stable)
        operation = process.stop
    else:
        from types import SimpleNamespace

        async def download(source, destination):
            destination.write_text(text)

        async def execute(command, timeout_sec):
            return SimpleNamespace(return_code=0, stdout=text, stderr="")

        sandbox = SimpleNamespace(exec=execute)
        if backend == "sandbox-download":
            sandbox.download_file = download
        process = _sandbox_process(sandbox)
        operation = process._load_callback_log
    await _heartbeat_during(operation, monkeypatch, process, "_import_callback_lines")
    assert process.trajectory is not None
    assert len(process.trajectory.exchanges) == 1


async def test_cancellation_waits_for_the_writer_before_final_drain(
    tmp_path, monkeypatch
):
    log = tmp_path / "callback.jsonl"
    log.write_text(_callback_line())
    process = _process(tmp_path, log)
    entered = threading.Event()
    release = threading.Event()
    original = LiveLLMTrajectoryWriter.append

    def blocking(writer, exchanges):
        entered.set()
        assert release.wait(5), "test did not release the capture worker"
        return original(writer, exchanges)

    monkeypatch.setattr(LiveLLMTrajectoryWriter, "append", blocking)
    output = tmp_path / "llm_trajectory.jsonl"
    process.start_live_capture(output)
    stop = None
    try:
        async with asyncio.timeout(5):
            while not entered.is_set():
                await asyncio.sleep(0.001)
        stop = asyncio.create_task(process._stop_live_capture())
        await asyncio.sleep(0.02)
        assert not stop.done()
        assert process._live_capture_task is not None
        process._live_capture_task.cancel()  # a second cancellation also drains
        await asyncio.sleep(0.01)
        assert not stop.done()
    finally:
        release.set()
        if stop is not None:
            await stop
        else:
            await process._stop_live_capture()
    assert len(output.read_text().splitlines()) == 1
    assert not output.with_suffix(".jsonl.tmp").exists()
