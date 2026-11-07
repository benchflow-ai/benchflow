"""PERF-C × PERF-B (results/PERF-ALL.md §2): a discarded warm-pool trial
removes its folder only after its queued sandbox stop has ended.

With PERF-B's teardown queue, ``cleanup()`` returns while the sandbox stop
still runs in the background; ``discard_prestart`` used to remove the trial
folder right after it. Merge wave batch 31.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from benchflow.rollout import Rollout
from benchflow.rollout._teardown_queue import TeardownQueue


@pytest.mark.asyncio
async def test_the_folder_goes_only_after_the_queued_stop(tmp_path: Path) -> None:
    folder = tmp_path / "trial"
    folder.mkdir()
    queue = TeardownQueue(2)
    release = asyncio.Event()
    seen: list[bool] = []

    async def stop() -> None:
        await release.wait()
        # The stop still sees the trial folder it may write to.
        seen.append(folder.exists())

    rollout = object.__new__(Rollout)

    async def cleanup() -> None:
        rollout._queued_stop = await queue.submit(stop, what="trial")
        rollout._phase = "cleaned"

    async def prestart() -> None:
        return None

    rollout._prestart = asyncio.create_task(prestart())
    rollout._phase = "started"
    rollout._rollout_dir = folder
    rollout.cleanup = cleanup  # type: ignore[method-assign]

    discard = asyncio.create_task(rollout.discard_prestart())
    for _ in range(20):
        await asyncio.sleep(0)
    assert not discard.done(), "the folder went before the queued stop ended"
    assert folder.exists()
    release.set()
    await discard
    assert seen == [True]
    assert not folder.exists()
    await queue.drain()
