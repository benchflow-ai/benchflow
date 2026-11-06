"""Expected-behaviour checks for the Daytona command channel.

Adopted from the reviewer's Appendix A. The helper-pipe difference is accepted,
as on Docker; the other checks require parity with the session path.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from benchflow.diagnostics import TransientSandboxTransportError
from benchflow.sandbox._base import KIT_PROBE_ROUTE
from tests.daytona_channel_fakes import _lifecycle_sandbox
from tests.test_daytona_channel import channel, local_pty, output  # noqa: F401
from tests.test_daytona_channel_lifecycle import _channel

pytestmark = pytest.mark.timeout(30)


@pytest.mark.parametrize("acknowledged", [False, True])
async def test_review_cancelling_one_sender_keeps_other_inflight_results(acknowledged):
    """Cancelling one caller mid-send must not fail
    another command already running on the channel; on the session path each
    command has its own session and is unaffected."""
    made = channel()
    first = asyncio.create_task(made.run("first", "/bin/sh", 10))
    await asyncio.sleep(0.01)
    await made.on_data(output(b"S", 1))  # the first command has started
    gate = asyncio.Event()

    async def stalled(data):
        if acknowledged and b'"i"' in data:
            await made.on_data(output(b"S", 2))
        await gate.wait()

    made.pty.send_input.side_effect = stalled
    second = asyncio.create_task(made.run("second", "/bin/sh", 10))
    await asyncio.sleep(0.01)
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    await made.on_data(output(b"X", 1, b"0"))
    assert (await first).return_code == 0
    gate.set()
    await asyncio.gather(*made.withdrawals)
    assert json.loads(made.pty.send_input.call_args.args[0][4:])["c"] == 2


async def test_review_command_queued_behind_a_stalled_frame_is_not_a_timeout(
    monkeypatch,
):
    """A command that waited for the write lock past
    its budget was never sent. The session path starts the budget at
    acceptance, so it would have run it; the channel holds proof of no start,
    so expect a fallback (None) or a transient transport error, never the
    command-timeout error."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)
    await made.write_lock.acquire()  # another frame's send is stalled
    try:
        try:
            outcome = await pool.run("side-effect", KIT_PROBE_ROUTE, 1)
        except Exception as exc:  # The outcome is the check.
            outcome = exc
    finally:
        made.write_lock.release()
    made.pty.send_input.assert_not_awaited()
    assert outcome is None or isinstance(outcome, TransientSandboxTransportError), (
        f"{type(outcome).__name__}: {outcome}"
    )


async def test_review_missing_status_response_is_a_transport_error(monkeypatch):
    """The step started, its deadline passed and the
    status query got no answer: expect the transient transport error, not a
    command timeout."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)

    async def send(data):
        if b'"i"' in data:
            await made.on_data(output(b"S", 1))  # started; no answer afterwards

    made.pty.send_input.side_effect = send
    with pytest.raises(TransientSandboxTransportError):
        await pool.run("sleep 100", KIT_PROBE_ROUTE, 1)


async def test_review_lost_started_command_is_never_sent_again(monkeypatch):
    """The channel dies after the step's S frame: expect
    no fallback (no session create) and a transient transport error."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)

    async def send(data):
        if b'"i"' in data:
            await made.on_data(output(b"S", 1))
            made.die("websocket closed")

    made.pty.send_input.side_effect = send
    with pytest.raises(TransientSandboxTransportError):
        await pool.run("side-effect", KIT_PROBE_ROUTE, 30)
    assert box._sandbox.process.created == []
    assert box._exec_counts().channel_fallbacks == 0


async def test_review_late_acknowledgement_within_the_session_request_bound():
    """The session path bounds the execute request by
    the budget plus 5 s (SDK http_timeout(timeout + 5)) and starts the budget
    at acceptance. An S frame 0.3 s after a 0.2 s budget, then the exit:
    expect the result, as the session path would return it."""
    made = channel()

    async def send(data):
        async def answer():
            await asyncio.sleep(0.3)
            await made.on_data(output(b"S", 1))
            await made.on_data(output(b"X", 1, b"0"))

        asyncio.get_running_loop().create_task(answer())

    made.pty.send_input.side_effect = send
    result = await made.run("true", "/bin/sh", 0.2)
    assert result.return_code == 0


async def test_review_no_channel_task_outlives_stop(monkeypatch):
    """When stop() returns, the provider is released and
    no task started by the channel is still pending."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    _pool, made = _channel(box)

    async def slow_disconnect():
        await asyncio.sleep(0.5)

    made.pty.disconnect = slow_disconnect
    box._sandbox.release.set()
    await box.stop(True)
    pending = [
        task.get_coro().__qualname__
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and not task.done()
    ]
    try:
        assert not pending, pending
    finally:
        if made.disconnecting is not None:
            await made.disconnecting


@pytest.mark.xfail(strict=True, reason="accepted helper-pipe lifetime")
async def test_review_timed_out_child_runs_on_after_the_helper_is_killed(
    local_pty,  # noqa: F811
    tmp_path,
):
    """A command the host stopped waiting for runs on,
    as a Daytona session command does, even if the helper is killed later
    (a channel restart's verdict)."""
    made, process = local_pty
    mark = tmp_path / "after"
    with pytest.raises(TimeoutError):
        await made.run(f"sleep 2; echo progress; echo done > {mark}", "/bin/sh", 1)
    process.kill()
    await process.wait()
    await asyncio.sleep(2.5)
    assert mark.exists() and mark.read_text() == "done\n"
