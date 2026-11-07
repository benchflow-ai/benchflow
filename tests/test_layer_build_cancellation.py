"""Cancelling Docker must reap its owned processes before a key is released."""

import asyncio
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox.env_layers import DockerLayers, LayerError


@pytest.mark.parametrize("ending", ["cancel", "timeout"])
async def test_real_docker_client_and_its_child_stop_before_return(
    monkeypatch, tmp_path, ending
):
    """Guard 02bbc02a/9bcc62da: reap the client and child before unlocking."""
    marker = tmp_path / "owned-pids"
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import os, sys, time\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    time.sleep(60)\n"
        "else:\n"
        "    with open(sys.argv[2], 'w') as f:\n"
        "        f.write(f'{os.getpid()} {child}')\n"
        "    time.sleep(60)\n"
    )
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    task = asyncio.create_task(
        DockerLayers()._docker(
            "build",
            str(marker),
            timeout=0.3 if ending == "timeout" else 30,
        )
    )
    owned = []

    def running(pid):
        try:
            # A killed orphan can remain a zombie until container init
            # reaps it. It cannot build or retain an output pipe then.
            return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
        except FileNotFoundError:
            return False

    try:
        async with asyncio.timeout(3):
            while not marker.exists() or len(marker.read_text().split()) != 2:
                await asyncio.sleep(0.01)
        owned = [int(pid) for pid in marker.read_text().split()]
        if ending == "cancel":
            task.cancel()
        expected = asyncio.CancelledError if ending == "cancel" else LayerError
        with pytest.raises(expected):
            await asyncio.wait_for(task, 2)
        for _ in range(50):
            if not any(running(pid) for pid in owned):
                break
            await asyncio.sleep(0.01)
        assert not any(running(pid) for pid in owned)
    finally:
        # Only these two dummy processes, inside the throwaway container.
        for pid in owned:
            if running(pid):
                os.kill(pid, signal.SIGKILL)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("ending", ["cancel", "timeout", "during-spawn"])
async def test_docker_reaps_before_unwinding_even_when_cancelled_twice(
    monkeypatch, ending
):
    """Guard 02bbc02a/U13-CONFIRM-2: reaping preserves external cancellation."""
    started = asyncio.Event()
    killed = asyncio.Event()
    allow_reap = asyncio.Event()
    reaped = asyncio.Event()
    allow_spawn = asyncio.Event()

    async def communicate(stdin=None):
        started.set()
        await killed.wait()
        await allow_reap.wait()
        reaped.set()
        return b"", b""

    def kill():
        killed.set()

    process = SimpleNamespace(
        pid=4000001,
        returncode=None,
        kill=kill,
        communicate=communicate,
    )

    async def spawn(*args, **kwargs):
        if ending == "during-spawn":
            started.set()
            await allow_spawn.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: kill())
    task = asyncio.create_task(
        DockerLayers()._docker(
            "build",
            timeout=0.03 if ending == "timeout" else 30,
        )
    )
    await started.wait()
    if ending != "timeout":
        task.cancel()
    allow_spawn.set()
    try:
        await asyncio.sleep(0.06)
        assert killed.is_set(), "cancelled Docker still runs after its lock unwinds"
        task.cancel()  # a second cancellation must not abandon reaping
        await asyncio.sleep(0)
        assert not task.done(), "key could be unlocked before Docker was reaped"
        allow_reap.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert reaped.is_set()
    finally:
        killed.set()
        allow_spawn.set()
        allow_reap.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
