"""Regressions for the command channel on the base session path.

Adopted from REVIEW-4's Appendix A. Each test states what the session path or
the lifecycle contract requires and expects the channel to keep it.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from benchflow.diagnostics import TransientSandboxTransportError
from benchflow.sandbox._base import KIT_PROBE_ROUTE
from benchflow.sandbox.protocol import ExecTimeoutError
from tests.daytona_channel_fakes import _lifecycle_sandbox
from tests.test_daytona_channel import output
from tests.test_daytona_channel_contract import _exec_box
from tests.test_daytona_channel_lifecycle import _channel

pytestmark = pytest.mark.timeout(30)


def _start_then_lose(made):
    """Fake PTY: the step starts (S), then the websocket closes."""

    async def send(data):
        head = json.loads(data[4:])
        if "i" in head:
            await made.on_data(output(b"S", head["i"]))
            made.die("websocket closed")

    made.pty.send_input.side_effect = send


async def test_review4_public_exec_never_reruns_a_started_command(monkeypatch):
    """exec() keeps the channel's exactly-once rule.
    The step started and then the channel was lost: the caller gets the
    transient transport error, as a lost session exec does, and no session is
    created to run the step a second time."""
    box, _pool, made = _exec_box(monkeypatch)
    _start_then_lose(made)
    with pytest.raises(TransientSandboxTransportError):
        await box.exec("side-effect", user="root", timeout_sec=5)
    assert box._sandbox.process.created == []
    counts = box._exec_counts()
    assert counts.exec_calls == counts.channel_exec_calls == 1
    assert counts.channel_fallbacks == 0


async def test_review4_failed_channel_command_still_counts_its_time(monkeypatch):
    """A channel command that ends in an
    error adds its wall time to exec_seconds, as the session path's finally
    does for every exec."""
    box, _pool, made = _exec_box(monkeypatch)
    _start_then_lose(made)
    with pytest.raises(TransientSandboxTransportError):
        await box.exec("side-effect", user="root", timeout_sec=5)
    counts = box._exec_counts()
    assert counts.exec_seconds >= counts.channel_exec_seconds > 0


async def test_review4_no_channel_writer_starts_after_stop_returns(monkeypatch):
    """When stop() returns, the released channel has
    no task pending and starts none later. Here a timed-out cleanup_session
    command's termination frame waits for the writer lock behind another
    command whose send is stalled when stop() releases the provider. Its
    caller still gets the command timeout."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)
    stalled = asyncio.Event()
    second = []

    async def send(data):
        head = json.loads(data[4:])
        if head.get("i") == 1:
            await made.on_data(output(b"S", 1))
        elif head.get("q") == 1:
            # Another root step asks for the writer while this query holds it.
            second.append(asyncio.create_task(pool.run("true", KIT_PROBE_ROUTE, 30)))
            await made.on_data(output(b"T", 1))
        elif head.get("i") == 2:
            stalled.set()
            await asyncio.Event().wait()  # this send never completes

    made.pty.send_input.side_effect = send
    original_write = made._write
    writes, writes_after_stop = [], []
    stopped = False

    async def write(data):
        writes.append(json.loads(data[4:]))
        if stopped:
            writes_after_stop.append(writes[-1])
        return await original_write(data)

    made._write = write
    first = asyncio.create_task(
        pool.run("sleep 100", KIT_PROBE_ROUTE, 1, cleanup_session=True)
    )
    await stalled.wait()
    await asyncio.sleep(0.05)  # the termination frame now waits for the lock
    box._sandbox.release.set()
    await box.stop(True)
    stopped = True
    pending = [
        task.get_coro().__qualname__
        for task in asyncio.all_tasks()
        if task not in (asyncio.current_task(), first, *second) and not task.done()
    ]
    outcomes = await asyncio.gather(first, *second, return_exceptions=True)
    assert {"k": 1, "n": -1} in writes  # the queued termination frame was tried
    assert not pending, pending
    assert writes_after_stop == [], writes_after_stop
    assert isinstance(outcomes[0], ExecTimeoutError), repr(outcomes[0])
    assert isinstance(outcomes[1], TransientSandboxTransportError), repr(outcomes[1])
