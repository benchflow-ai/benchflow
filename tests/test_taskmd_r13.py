"""Publication regressions in 7c24f4d88 (TASKMD-R12, repaired by R13)."""

import asyncio
import os
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow._utils import reachable_write as rw
from benchflow.sandbox.docker import DockerSandbox


@pytest.mark.asyncio
async def test_cancel_stalled_download_keeps_loop_running(tmp_path, monkeypatch):
    """Guards R12-5 introduced by 7c24f4d88: join only after producer death."""
    from benchflow.sandbox import docker

    started = threading.Event()
    producer = None
    real_spawn = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        nonlocal producer
        if args[-2:] == ("system", "dial-stdio"):
            # The source is a regular file; this test stalls the copy stream.
            return SimpleNamespace(
                returncode=0,
                communicate=AsyncMock(
                    return_value=(
                        b"HTTP/1.1 200 OK\r\n"
                        b"X-Docker-Container-Path-Stat: eyJtb2RlIjo0MjB9\r\n\r\n",
                        b"",
                    )
                ),
            )
        producer = await real_spawn(
            "/bin/sh",
            "-c",
            "exec sleep 30",
            stdout=kwargs["stdout"],
            stderr=asyncio.subprocess.PIPE,
        )
        return producer

    def extract(self, reader):
        started.set()
        reader.read()

    monkeypatch.setattr(docker.asyncio, "create_subprocess_exec", spawn)
    # TASKMD-R14 B-M1: extraction is StagedTar.extract; commit follows exit 0.
    monkeypatch.setattr(docker.StagedTar, "extract", extract)
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._main_container = "c" * 64
    sandbox._docker_compose_global_args = Mock(return_value=[])
    sandbox._docker_compose_env = Mock(return_value={})
    sandbox._compose_transport_env = Mock(return_value={})
    root = rw.ReachablePath.directory(tmp_path)
    transfer = asyncio.create_task(
        sandbox._stream_download("/output", root / "out", directory=False)
    )
    assert await asyncio.to_thread(started.wait, 5)
    # Independent watchdog bounds the old event-loop deadlock. The assertion
    # distinguishes prompt cancellation from the watchdog's eventual rescue.
    rescued = threading.Event()

    def rescue():
        rescued.set()
        if producer is not None:
            with __import__("contextlib").suppress(ProcessLookupError):
                os.kill(producer.pid, 9)

    watchdog = threading.Timer(3, rescue)
    watchdog.start()
    try:
        transfer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await transfer
        assert not rescued.is_set(), (
            "cancellation blocked the event loop until watchdog"
        )
        assert producer.returncode is not None
    finally:
        watchdog.cancel()
