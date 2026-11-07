"""Channel retirement cannot split provider work from its bookkeeping."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import daytona as module
from benchflow.sandbox import daytona_strategies as strategies
from benchflow.sandbox._base import KIT_PROBE_ROUTE
from benchflow.sandbox._exec_channel import (
    ChannelLost,
    ChannelUnavailable,
    ChannelUnconfirmed,
)
from benchflow.sandbox.daytona_channel import DaytonaChannel, DaytonaChannels, eligible
from tests.daytona_channel_fakes import (
    _checkpoint,
    _lifecycle_sandbox,
    _Remote,
    _restore_setup,
)

pytestmark = pytest.mark.timeout(30)


def _channel(sandbox):
    provider = sandbox._sandbox
    made = DaytonaChannel(provider, sandbox._exec_counts())
    made.ready.set_result(None)
    made.synchronized = True
    made.pty = SimpleNamespace(
        send_input=AsyncMock(), kill=AsyncMock(), disconnect=AsyncMock()
    )
    pool = object.__new__(DaytonaChannels)
    pool.sandbox, pool.provider = sandbox, provider
    pool.generation, pool.route = sandbox._exec_generation, KIT_PROBE_ROUTE
    pool.closed, pool.channels, pool.folders = False, [made], []
    pool.cleanup_task = None
    pool.inflight = {}
    pool.starting = asyncio.get_running_loop().create_future()
    pool.starting.set_result(made)
    sandbox._command_channels = pool
    sandbox._command_channel_provider = provider
    sandbox._command_channels_closing = False
    return pool, made


@pytest.mark.parametrize("strategy", [module._DaytonaDirect, module._DaytonaDinD])
async def test_stop_retires_before_delete_without_awaiting_channel_startup(
    monkeypatch, strategy
):
    """Pending channel startup cannot bypass delete or the ledger."""
    box, ledger = _lifecycle_sandbox(monkeypatch, strategy)
    old = box._sandbox
    pool, made = _channel(box)
    release = asyncio.Event()
    pool.starting = asyncio.create_task(release.wait())
    work = asyncio.create_task(box.stop(True))
    try:
        await asyncio.wait_for(old.deleting.wait(), 1)
        assert pool.closed and made.dead
        assert box._command_channels is None
        assert pool.cleanup_task is None
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work
        assert ledger.entries == [old.id]
        assert box._sandbox is None
        assert pool.finalized
        assert pool.starting.done() or pool.starting.cancelling()
    finally:
        release.set()
        if not work.done():
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
        await box._close_command_channels()
        if made.disconnecting is not None:
            await made.disconnecting
    made.pty.kill.assert_not_awaited()
    made.pty.disconnect.assert_awaited_once()


async def test_stop_cleanup_cannot_clear_a_concurrently_restored_channel(monkeypatch):
    """Retired-channel cleanup owns no replacement handle or slot."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    old, new = box._sandbox, _Remote("replacement")
    old.release.set()
    _restore_setup(monkeypatch, box, new)
    pool, made = _channel(box)
    release = asyncio.Event()
    original_close = pool.close

    async def close():
        await release.wait()
        await original_close()

    pool.close = close
    replacement = None

    async def deliver(*args):
        nonlocal replacement
        replacement = _channel(box)

    monkeypatch.setattr(strategies, "_deliver_bfx", deliver)

    async def restore():
        await old.deleting.wait()
        await box.restore(_checkpoint())

    try:
        await asyncio.wait_for(asyncio.gather(box.stop(True), restore()), 1)
        assert box._sandbox is new
        assert replacement is not None
        assert box._command_channels is replacement[0]
        assert not box._command_channels_closing
        assert made.dead and not replacement[1].dead
        release.set()
        if pool.cleanup_task is not None:
            await asyncio.gather(pool.cleanup_task, return_exceptions=True)
        assert box._command_channels is replacement[0]
        assert not replacement[1].dead
    finally:
        release.set()
        await box._close_command_channels()


@pytest.mark.parametrize("recorded", [True, False], ids=["recorded", "missing"])
async def test_restore_commits_inspector_before_independent_channel_cleanup(
    monkeypatch, recorded
):
    """Inspector provenance commits before cleanup cancellation."""
    from benchflow.sandbox.lockdown import _path_inspector_prefix

    box, _ = _lifecycle_sandbox(monkeypatch)
    old, new = box._sandbox, _Remote("replacement")
    old.release.set()
    _restore_setup(monkeypatch, box, new)
    previous = "/etc/.runtime-bash-env-" + "a" * 32
    restored = "/etc/.runtime-bash-env-" + "b" * 32
    box._benchflow_path_inspectors = {"main": previous}
    pool, _ = _channel(box)
    release = asyncio.Event()
    pool.starting = asyncio.create_task(release.wait())

    async def cancel_after_provider_work(*args):
        asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)

    monkeypatch.setattr(strategies, "_deliver_bfx", cancel_after_provider_work)
    try:
        image = _checkpoint(
            {"benchflow.path-inspector.main": restored} if recorded else {}
        )
        work = asyncio.create_task(box.restore(image))
        with pytest.raises(asyncio.CancelledError):
            await work
        assert box._sandbox is new
        assert box._benchflow_path_inspectors.get("main") == (
            restored if recorded else None
        )
        if recorded:
            assert restored in _path_inspector_prefix(box)
        else:
            with pytest.raises(RuntimeError, match="missing pre-agent PATH inspector"):
                _path_inspector_prefix(box)
    finally:
        release.set()
        await box._close_command_channels()


async def test_rejected_restore_does_not_reopen_the_same_provider(monkeypatch):
    """Repeated publication cannot restart a retired channel generation."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    box._sandbox.process.create_pty_session = AsyncMock()
    pool, made = _channel(box)
    assert eligible(box, KIT_PROBE_ROUTE)
    box._strategy.restore = AsyncMock(side_effect=RuntimeError("rejected"))
    try:
        with pytest.raises(RuntimeError, match="rejected"):
            await box.restore(_checkpoint())
        box._on_sandbox_created()
        assert pool.closed and made.dead
        assert not eligible(box, KIT_PROBE_ROUTE)
    finally:
        await box._close_command_channels()


async def test_start_publishes_without_joining_retired_channel_startup(monkeypatch):
    """Optional startup cleanup cannot interrupt new-provider bookkeeping."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)
    release = asyncio.Event()
    pool.starting = asyncio.create_task(release.wait())
    new = _Remote("replacement")

    async def start(force_build):
        assert pool.closed and made.dead
        box._sandbox = new
        box._on_sandbox_created()
        asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)

    box._strategy.start = start
    try:
        work = asyncio.create_task(box.start(False))
        assert await asyncio.gather(work, return_exceptions=True) == [None]
        assert box._sandbox is new
        assert not box._command_channels_closing
    finally:
        release.set()
        await box._close_command_channels()


async def test_retirement_rejects_pending_and_future_frames(monkeypatch):
    """Retirement stops delivery synchronously, before optional teardown."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)
    work = asyncio.create_task(made.run("side-effect", "/bin/sh", 5))
    await asyncio.sleep(0.01)
    pool.retire()
    with pytest.raises(ChannelLost):
        await work
    sent = made.pty.send_input.await_count
    with pytest.raises(ChannelUnavailable):
        await made.run("next", "/bin/sh", 5)
    assert made.pty.send_input.await_count == sent
    await box._close_command_channels()


@pytest.mark.parametrize("delete_fails", [False, True])
def test_channel_runner_shutdown_has_no_orphaned_work(
    monkeypatch, tmp_path, delete_fails
):
    """Provider release finalizes channel work before Runner shutdown."""
    observed = {}

    async def main():
        box, ledger = _lifecycle_sandbox(monkeypatch)
        provider = box._sandbox
        box.rollout_paths = SimpleNamespace(rollout_dir=tmp_path)
        pool, made = _channel(box)
        entered = asyncio.Event()

        async def starting():
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0)
                raise

        pool.starting = asyncio.create_task(starting())
        await entered.wait()
        provider.fail_delete = delete_fails
        provider.release.set()
        await box.stop(True)
        # Runner.close must not hide orphaned channel work.
        assert not [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        ]
        assert provider.deleted or provider.id in ledger.entries
        assert pool.finalized and made.finalized
        assert not box._retired_command_channels
        observed.update(
            box=box,
            pool=pool,
            made=made,
            record=(tmp_path / module.EXEC_COUNTS_FILE).read_bytes(),
            counts=asdict(box._exec_counts()),
        )

    runner = asyncio.Runner()
    loop = runner.get_loop()
    try:
        runner.run(main())
    finally:
        runner.close()
    assert not asyncio.all_tasks(loop)
    assert observed["pool"].starting.done()
    observed["made"].pty.kill.assert_not_awaited()
    assert observed["counts"] == asdict(observed["box"]._exec_counts())
    assert observed["record"] == (tmp_path / module.EXEC_COUNTS_FILE).read_bytes()


async def test_released_channel_cannot_replay_or_rewrite_stop_counts(
    monkeypatch, tmp_path
):
    """An uncertain in-flight frame stays failed with final accounting."""
    from benchflow.diagnostics import TransientSandboxTransportError

    box, _ = _lifecycle_sandbox(monkeypatch)
    provider = box._sandbox
    box.rollout_paths = SimpleNamespace(rollout_dir=tmp_path)
    pool, made = _channel(box)
    entered, released = asyncio.Event(), asyncio.Event()

    async def uncertain(*args, **kwargs):
        entered.set()
        await released.wait()
        raise ChannelUnconfirmed("lost", 1)

    made.run = uncertain
    made.verdict = AsyncMock()
    work = asyncio.create_task(pool.run("side-effect", KIT_PROBE_ROUTE, 30))
    await entered.wait()
    provider.release.set()
    await box.stop(True)
    before = (tmp_path / module.EXEC_COUNTS_FILE).read_bytes()
    counts = asdict(box._exec_counts())
    assert json.loads(before)["channel_exec_pending_at_release"] == 1
    released.set()
    with pytest.raises(TransientSandboxTransportError):
        await work
    if made.disconnecting is not None:
        await made.disconnecting
    made.verdict.assert_not_awaited()
    assert counts == asdict(box._exec_counts())
    assert before == (tmp_path / module.EXEC_COUNTS_FILE).read_bytes()


async def test_confirmed_provider_release_drops_failed_channel_cleanup(monkeypatch):
    """Completed cleanup failures cannot retain deleted generations."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    for index in range(4):
        old, new = box._sandbox, _Remote(f"replacement-{index}")
        pool, made = _channel(box)
        pool.close = AsyncMock(side_effect=ConnectionError("cleanup failed"))
        box._retire_command_channels()
        box._queue_command_channel_cleanup()
        await pool.cleanup_task
        assert pool in box._retired_command_channels
        old.release.set()
        _restore_setup(monkeypatch, box, new)
        await box.restore(_checkpoint())
        assert old.deleted and pool.finalized
        assert not box._retired_command_channels
        if made.disconnecting is not None:
            await made.disconnecting


async def test_finalization_during_verdict_is_transport_loss_not_caller_cancel(
    monkeypatch,
):
    """Cancelling a released provider's journal cannot permit replay."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)
    entered = asyncio.Event()
    made.pid, made.journal = 123, "/root/journal"

    async def verdict(*args):
        entered.set()
        await asyncio.Event().wait()

    made.verdict = verdict
    work = asyncio.create_task(made.started_ids())
    await entered.wait()
    pool.finalize()
    assert await work is None
    if made.disconnecting is not None:
        await made.disconnecting


async def test_finalized_startup_disconnects_a_late_accepted_pty(monkeypatch):
    """The original startup owns a late handle after finalization."""
    from benchflow.sandbox import daytona_channel as channel_module

    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, _ = _channel(box)
    pool.channels = []
    entered, release = asyncio.Event(), asyncio.Event()
    pty = SimpleNamespace(send_input=AsyncMock(), disconnect=AsyncMock())

    async def create(**kwargs):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return pty

    box._sandbox.process.create_pty_session = create
    monkeypatch.setattr(channel_module, "helper_python_argv", lambda _: ["/python"])
    pool._control = AsyncMock(return_value=module.ExecResult(return_code=0))
    pool._stage_source = AsyncMock(return_value="/root/stage/payload")
    pool.starting = asyncio.create_task(pool._start())
    await entered.wait()
    pool.finalize()
    release.set()
    await asyncio.gather(pool.starting, return_exceptions=True)
    pty.send_input.assert_not_awaited()
    pty.disconnect.assert_awaited_once()


async def test_start_retry_finalizes_the_previous_channel(monkeypatch):
    """Startup retries release channels with the captured provider."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    provider = box._sandbox
    pool, made = _channel(box)
    provider.release.set()
    box._create_attempts = 0
    box._client_manager = SimpleNamespace(
        get_client=AsyncMock(side_effect=RuntimeError("stop after old release"))
    )
    with pytest.raises(RuntimeError, match="stop after old release"):
        await module.DaytonaSandbox._create_sandbox.__wrapped__(box, None)
    assert provider.deleted and pool.finalized
    assert not box._retired_command_channels
    if made.disconnecting is not None:
        await made.disconnecting


@pytest.mark.parametrize("refused", [False, True])
async def test_startup_admission_finalizes_removed_channels(monkeypatch, refused):
    """Sanitizer replacement and refusal finalize old channels."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    old, new = box._sandbox, _Remote("sanitized")
    providers = iter([old, new])
    channels = []

    async def create(**kwargs):
        box._sandbox = next(providers)
        box._sandbox.release.set()
        box._on_sandbox_created()
        channels.append(_channel(box))

    empty = dict.fromkeys(module.SYSTEM_SHELL_EMPTY_VARS, "")
    answers = (
        [RuntimeError("startup probe failed")]
        if refused
        else [{**empty, "BASH_ENV": "/startup"}, empty]
    )
    box._create_sandbox = create
    box._toolbox_startup_env = AsyncMock(side_effect=answers)
    params = SimpleNamespace(env_vars={}, model_copy=lambda **kwargs: None)
    if refused:
        with pytest.raises(RuntimeError, match="startup probe failed"):
            await box._create_sandbox_without_toolbox_bash_env(params)
    else:
        await box._create_sandbox_without_toolbox_bash_env(params)
    assert old.deleted and channels[0][0].finalized
    assert not box._retired_command_channels
    await box.stop(True)
    for _, made in channels:
        if made.disconnecting is not None:
            await made.disconnecting


async def test_late_pty_creation_does_not_launch_a_retired_helper():
    """An accepted late PTY is owned but receives no helper launch."""
    release = asyncio.Event()
    creating = asyncio.Event()
    pty = SimpleNamespace(
        send_input=AsyncMock(), kill=AsyncMock(), disconnect=AsyncMock()
    )

    async def create(**kwargs):
        creating.set()
        await release.wait()
        return pty

    provider = SimpleNamespace(process=SimpleNamespace(create_pty_session=create))
    made = DaytonaChannel(provider, module._ExecCounts())
    work = asyncio.create_task(made.connect("launch", KIT_PROBE_ROUTE))
    await creating.wait()
    made.die("retired")
    release.set()
    with pytest.raises(ChannelUnavailable):
        await work
    pty.send_input.assert_not_awaited()
    await made.close()
    pty.kill.assert_awaited_once()


async def test_helper_upload_stays_on_the_captured_provider(monkeypatch):
    """A late helper upload cannot retry against a replacement provider."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, _ = _channel(box)
    entered, release = asyncio.Event(), asyncio.Event()

    async def upload(*args):
        entered.set()
        await release.wait()
        raise ConnectionError("upload response lost")

    old = box._sandbox
    old.fs = SimpleNamespace(upload_file=AsyncMock(side_effect=upload))
    work = asyncio.create_task(pool._stage_source(b"protected helper"))
    await entered.wait()
    pool.retire()
    box._sandbox = SimpleNamespace(fs=SimpleNamespace(upload_file=AsyncMock()))
    release.set()
    with pytest.raises(ConnectionError, match="response lost"):
        await work
    old.fs.upload_file.assert_awaited_once()
    box._sandbox.fs.upload_file.assert_not_awaited()
    assert len(pool.folders) == 1
    pool._control = AsyncMock()
    await box._close_command_channels()
