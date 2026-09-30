"""The machine-wide sandbox cap holds across processes and frees itself on death."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time

import pytest

from benchflow_taskset.slots import sandbox_slot


async def test_one_slot_serializes_holders(tmp_path) -> None:
    order: list[str] = []

    async def hold(name: str, seconds: float) -> None:
        async with sandbox_slot(tmp_path, 1, poll_sec=0.05):
            order.append(f"{name}+")
            await asyncio.sleep(seconds)
            order.append(f"{name}-")

    await asyncio.gather(hold("a", 0.3), hold("b", 0.0))
    assert order in (["a+", "a-", "b+", "b-"], ["b+", "b-", "a+", "a-"])


async def test_two_slots_admit_two_holders(tmp_path) -> None:
    async with sandbox_slot(tmp_path, 2) as first:
        async with asyncio.timeout(2):
            async with sandbox_slot(tmp_path, 2) as second:
                assert {first, second} == {0, 1}


async def test_an_error_releases_the_slot(tmp_path) -> None:
    with pytest.raises(RuntimeError):
        async with sandbox_slot(tmp_path, 1):
            raise RuntimeError("episode failed")
    async with asyncio.timeout(2):
        async with sandbox_slot(tmp_path, 1):
            pass


async def test_a_dead_process_frees_its_slot(tmp_path) -> None:
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import asyncio, sys\n"
            "from benchflow_taskset.slots import sandbox_slot\n"
            "async def main():\n"
            f"    async with sandbox_slot({str(tmp_path)!r}, 1):\n"
            "        print('held', flush=True)\n"
            "        await asyncio.sleep(3600)\n"
            "asyncio.run(main())\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.5):
                async with sandbox_slot(tmp_path, 1, poll_sec=0.05):
                    pass
    finally:
        holder.kill()
        holder.wait()
    started = time.monotonic()
    async with asyncio.timeout(5):
        async with sandbox_slot(tmp_path, 1, poll_sec=0.05):
            pass
    assert time.monotonic() - started < 5
