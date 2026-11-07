"""Guards 9b0320f: deletion acceptance must not release sandbox ownership.

Exercise the installed Daytona SDK's delete, state polling and timeout code
with only its HTTP API replaced. No provider allocation or socket is allowed.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import daytona as native
from benchflow.sandbox import daytona_ledger as ledger_module
from benchflow.sandbox.daytona_ledger import Ledger
from benchflow.sandbox.daytona_strategies import _DaytonaDirect
from benchflow.sandbox.protocol import SandboxImage, SandboxStartupError

pytest.importorskip("daytona")
from daytona import AsyncSandbox, Sandbox
from daytona.common.errors import (
    DaytonaError,
    DaytonaNotFoundError,
    DaytonaTimeoutError,
)
from daytona_api_client.models.sandbox import Sandbox as SyncDto
from daytona_api_client_async.models.sandbox import Sandbox as AsyncDto

_ID = "00000000-0000-0000-0000-000000000002"
_DATA = dict(
    id=_ID,
    organization_id="offline",
    name="offline",
    user="root",
    labels={},
    public=False,
    target="us",
    cpu=1,
    gpu=0,
    memory=1,
    disk=1,
    toolbox_proxy_url="",
    state="started",
)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("socket access forbidden in offline deletion test")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setenv("BENCHFLOW_DAYTONA_LEFTOVERS", str(tmp_path / "leftovers"))
    ledger = Ledger()
    monkeypatch.setattr(native, "LEDGER", ledger)
    monkeypatch.setattr(ledger_module, "LEDGER", ledger)
    monkeypatch.setattr(ledger, "ensure_sweeper", lambda: None)
    yield ledger
    # No atexit sweep: failures were deliberately retained and asserted here.
    ledger.entries.clear()


class Provider:
    def __init__(self, *, sync=False, gone=False, error=None, pending=False):
        cls, dto_type = (Sandbox, SyncDto) if sync else (AsyncSandbox, AsyncDto)
        self.events = []
        self.gone = gone
        self.error = error
        self.pending = pending
        self.on_poll = lambda: None
        self.box = cls.model_construct(**_DATA)
        self.box._sub_id = None
        self.box._state_waiters = []
        self.box._subscription_manager = SimpleNamespace(
            subscribe=lambda *a, **k: None, unsubscribe=lambda *a: None
        )
        self.dto = dto_type.model_construct(
            **(dict.fromkeys(dto_type.model_fields) | _DATA | {"state": "destroying"})
        )
        if sync:
            self.box._sandbox_api = SimpleNamespace(
                delete_sandbox=self.delete, get_sandbox=self.get
            )
        else:
            self.box._sandbox_api = SimpleNamespace(
                delete_sandbox=self.adelete, get_sandbox=self.aget
            )

    def delete(self, *args, **kwargs):
        self.events.append("delete")
        if self.gone:
            raise DaytonaNotFoundError("already absent", status_code=404)
        return self.dto

    def get(self, *args, **kwargs):
        self.events.append("poll")
        self.on_poll()
        if self.error is not None:
            raise self.error
        if self.events.count("poll") > 1 and not self.pending:
            self.events.append("absent")
            raise DaytonaNotFoundError("absent", status_code=404)
        return self.dto

    async def adelete(self, *args, **kwargs):
        return self.delete(*args, **kwargs)

    async def aget(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def sync_client(self):
        return SimpleNamespace(
            get=lambda _, **kwargs: self.box,
            list=lambda _, **kwargs: [self.box],
            delete=lambda box, **kwargs: box.delete(**kwargs),
        )

    def async_client(self):
        async def listed(_):
            yield self.box

        return SimpleNamespace(get=AsyncMock(return_value=self.box), list=listed)


def environment(box):
    env = native.DaytonaSandbox.__new__(native.DaytonaSandbox)
    env._sandbox = box
    env._create_attempts = 0
    env._create_outcome_unknown = False
    env._client_manager = SimpleNamespace(get_client=AsyncMock())
    env.task_env_config = SimpleNamespace(build_timeout_sec=60)
    env.logger = logging.getLogger("test.confirmed-delete")
    env.rollout_paths = None
    return env


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["stop", "ledger", "attempt", "labelled", "failed-start"]
)
@pytest.mark.parametrize("gone", [False, True])
async def test_async_paths_confirm_deletion_before_return(path, gone, isolated):
    """Guards 9b0320f: ACK in DESTROYING cannot release the old allocation."""
    provider = Provider(gone=gone)
    box = provider.box
    env = environment(box)
    client = provider.async_client()
    if path in {"stop", "ledger"}:
        entry = isolated.open_sandbox(None, box.id, "test", managed="1", handle=box)
        provider.on_poll = lambda: _assert_open(isolated, entry)
    elif path == "attempt":
        entry = isolated.open_attempt(
            client, {"benchflow.create-attempt": "test"}, "test", managed="1"
        )
        provider.on_poll = lambda: _assert_open(isolated, entry)
    if path == "stop":
        await env._stop_sandbox(box)  # deletion owns its captured provider.
    elif path in {"ledger", "attempt"}:
        assert await isolated._sweep_one(entry)
    elif path == "labelled":
        assert await env._delete_sandboxes_labelled(client, {}, require_match=True)
    else:
        await env._delete_sandbox_that_failed_to_start(
            client,
            DaytonaError(
                f"Sandbox {_ID} failed to start with state: SandboxState.BUILD_FAILED"
            ),
        )
    assert not isolated.entries
    if gone:
        assert provider.events == ["delete"]
    else:
        assert box.state == "destroyed"
        assert provider.events == ["delete", "poll", "poll", "absent"]


def _assert_open(ledger, entry):
    assert entry.key in ledger.entries


@pytest.mark.parametrize("path", ["ledger", "attempt", "cleanup", "reaper"])
@pytest.mark.parametrize("gone", [False, True])
def test_sync_paths_confirm_deletion_before_return(path, gone, isolated):
    """Guards 9b0320f: exit/CLI/reaper success needs confirmed remote absence."""
    provider = Provider(sync=True, gone=gone)
    box = provider.box
    client = provider.sync_client()
    entry = isolated.open_sandbox(None, box.id, "test", managed="1")
    if path == "attempt":
        isolated.entries.clear()
        entry = isolated.open_attempt(
            None, {"benchflow.create-attempt": "test"}, "test", managed="1"
        )
    if path in {"ledger", "attempt"}:
        provider.on_poll = lambda: _assert_open(isolated, entry)
        isolated._sweep_one_sync(client, entry)
        assert entry.key not in isolated.entries
    elif path == "cleanup":
        counts = ledger_module.cleanup_leftovers(client, "1")
        assert counts["deleted"] + counts["closed"] == 1
        assert ledger_module.read_leftovers() == []
    else:
        from benchflow.sandbox.daytona_reaper import _delete_counted

        counts = {"deleted": 0, "failed": 0}
        _delete_counted(client, box, counts, False)
        assert counts == {"deleted": 1, "failed": 0}
    if gone:
        assert provider.events == ["delete"]
    else:
        assert box.state == "destroyed"
        assert provider.events == ["delete", "poll", "poll", "absent"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transient", "timeout", "cancel"])
async def test_async_failed_confirmation_keeps_ownership(failure, isolated):
    """Guards 9b0320f: polling failure/cancellation cannot close the ledger."""
    error = {
        "transient": DaytonaError("unavailable", status_code=503),
        "timeout": DaytonaTimeoutError("poll deadline"),
        "cancel": asyncio.CancelledError(),
    }[failure]
    provider = Provider(error=error)
    entry = isolated.open_sandbox(None, _ID, "test", managed="1", handle=provider.box)
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await isolated._sweep_one(entry)
    else:
        assert not await isolated._sweep_one(entry)
    assert entry.key in isolated.entries
    assert not entry.busy
    assert provider.box.state == "destroying"
    assert ledger_module.read_leftovers()[0]["sandbox_id"] == _ID


@pytest.mark.parametrize("failure", ["transient", "timeout"])
def test_sync_failed_confirmation_keeps_ownership(failure, isolated):
    """Guards 9b0320f: sync delete ACK followed by failed polling stays open."""
    error = (
        DaytonaError("unavailable", status_code=503)
        if failure == "transient"
        else DaytonaTimeoutError("poll deadline")
    )
    provider = Provider(sync=True, error=error)
    entry = isolated.open_sandbox(None, _ID, "test", managed="1")
    isolated._sweep_one_sync(provider.sync_client(), entry)
    assert entry.key in isolated.entries
    assert provider.box.state == "destroying"
    assert entry.errors


@pytest.mark.asyncio
async def test_actual_async_wait_has_an_explicit_bound(monkeypatch, isolated):
    """Guards 9b0320f: perpetual DESTROYING times out without releasing ownership."""
    from benchflow.sandbox import daytona_delete

    monkeypatch.setattr(daytona_delete, "DELETE_TIMEOUT_SEC", 0.02)
    provider = Provider(pending=True)
    entry = isolated.open_sandbox(None, _ID, "test", managed="1", handle=provider.box)
    assert not await asyncio.wait_for(isolated._sweep_one(entry), timeout=1)
    assert entry.key in isolated.entries
    assert "timeout" in entry.errors[-1].lower()
    assert provider.box.state == "destroying"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["retry", "restore"])
async def test_failed_replacement_keeps_handle_and_never_creates(
    path, monkeypatch, isolated
):
    """Guards 9b0320f: a replacement cannot overlap unconfirmed prior deletion."""
    provider = Provider(error=DaytonaTimeoutError("poll deadline"))
    env = environment(provider.box)
    manager = env._client_manager
    create = AsyncMock()
    env._create_sandbox_without_toolbox_bash_env = create
    monkeypatch.setattr(native, "_checkpoint_startup_env", lambda _: {})
    if path == "retry":
        with pytest.raises(SandboxStartupError):
            await env._create_sandbox(params=object())
    else:
        with pytest.raises(DaytonaTimeoutError):
            await _DaytonaDirect(env).restore(
                SandboxImage(provider="daytona", ref="checkpoint")
            )
    assert env._sandbox is provider.box
    assert f"sandbox:{_ID}" in isolated.entries
    manager.get_client.assert_not_awaited()
    create.assert_not_awaited()


def test_actual_sync_wait_has_an_explicit_bound(monkeypatch, isolated):
    """Guards 9b0320f: sync exit cleanup bounds a permanently pending delete."""
    from benchflow.sandbox import daytona_delete

    monkeypatch.setattr(daytona_delete, "DELETE_TIMEOUT_SEC", 0.05)
    provider = Provider(sync=True, pending=True)
    entry = isolated.open_sandbox(None, _ID, "test", managed="1")
    isolated._sweep_one_sync(provider.sync_client(), entry)
    assert entry.key in isolated.entries
    assert "timeout" in entry.errors[-1].lower()
    assert provider.box.state == "destroying"


@pytest.mark.asyncio
async def test_external_cancellation_during_poll_keeps_ownership(isolated):
    """Guards 9b0320f: interrupting the real SDK wait keeps durable ownership."""
    provider = Provider(pending=True)
    polled = asyncio.Event()
    provider.on_poll = polled.set
    entry = isolated.open_sandbox(None, _ID, "test", managed="1", handle=provider.box)
    work = asyncio.create_task(isolated._sweep_one(entry))
    await asyncio.wait_for(polled.wait(), timeout=1)
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await work
    assert entry.key in isolated.entries
    assert not entry.busy
    assert ledger_module.read_leftovers()[0]["sandbox_id"] == _ID


@pytest.mark.asyncio
@pytest.mark.parametrize("desired", [None, "destroyed"])
async def test_verifier_guard_rejects_destroying(desired):
    """Guards 9b0320f: desired destruction and in-progress deletion are not release."""
    from benchflow.rollout._separate_verifier import agent_sandbox_left_running

    provider = Provider(pending=True)
    provider.dto.desired_state = desired
    await provider.box.delete(timeout=1)  # actual SDK accepts, leaves DESTROYING
    error = await agent_sandbox_left_running(environment(provider.box), provider.box)
    assert error is not None
    assert "destroying" in error
    assert provider.events == ["delete", "poll"]


@pytest.mark.asyncio
@pytest.mark.parametrize("actual", ["started", "destroyed", "404", "timeout"])
async def test_verifier_guard_requires_actual_release(actual):
    """Guards 9b0320f: only actual destruction/404 releases a requested delete."""
    from benchflow.rollout._separate_verifier import agent_sandbox_left_running

    provider = Provider(pending=True)
    provider.box.state = actual if actual in {"started", "destroyed"} else "destroying"
    provider.box.desired_state = "destroyed"
    provider.error = {
        "404": DaytonaNotFoundError("absent", status_code=404),
        "timeout": DaytonaTimeoutError("poll deadline"),
    }.get(actual)
    error = await agent_sandbox_left_running(environment(provider.box), provider.box)
    if actual in {"destroyed", "404"}:
        assert error is None
    else:
        assert error is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [False, True])
async def test_after_stop_verifier_waits_for_real_sdk_confirmation(
    tmp_path, monkeypatch, isolated, pending
):
    """Guards 9b0320f: public stop can swallow failure, but verification must not overlap."""
    from benchflow.sandbox import daytona_delete
    from tests.test_sv_overlap import _scenario, _verify

    monkeypatch.setattr(daytona_delete, "DELETE_TIMEOUT_SEC", 0.02 if pending else 1)
    monkeypatch.setattr(native.DaytonaSandbox._stop_sandbox.retry, "sleep", AsyncMock())
    provider = Provider(pending=pending)
    provider.dto.desired_state = "destroyed"
    env = environment(provider.box)
    env._strategy = _DaytonaDirect(env)
    sc = await _scenario(
        tmp_path, monkeypatch, environment="daytona", trajectory=[], agent_env=env
    )
    create = sc.rollout._planes.create_environment
    allocations = []

    def create_after_release(*args, **kwargs):
        allocations.append(provider.box.state)
        return create(*args, **kwargs)

    sc.rollout._planes.create_environment = create_after_release
    try:
        rewards = await _verify(sc)
        if pending:
            assert allocations == []
            assert rewards is None
            assert "agent sandbox release" in sc.rollout._verifier_error
            assert sc.record()["status"] == "abandoned"
            assert env._sandbox is provider.box
            assert f"sandbox:{_ID}" in isolated.entries
        else:
            assert allocations == ["destroyed"]
            assert rewards == {"reward": 1.0}
            assert sc.record()["status"] == "complete"
    finally:
        provider.pending = False
        monkeypatch.setattr(daytona_delete, "DELETE_TIMEOUT_SEC", 1)
        await sc.rollout.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("handle", [None, SimpleNamespace(id=_ID, state="destroying")])
async def test_verifier_guard_without_refreshable_handle_is_uncertain(handle):
    """Guards 9b0320f: a missing native Daytona handle cannot establish release."""
    from benchflow.rollout._separate_verifier import agent_sandbox_left_running

    assert await agent_sandbox_left_running(environment(handle), handle) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_create_keeps_ownership_until_reconciled(
    monkeypatch, isolated, named, cancel
):
    """Guards 9b0320f: a failed create's pending delete blocks every new allocation."""
    from benchflow.sandbox import daytona_delete

    monkeypatch.setattr(daytona_delete, "DELETE_TIMEOUT_SEC", 0.02 if not cancel else 1)
    provider = Provider(pending=True)
    polled = asyncio.Event()
    provider.on_poll = polled.set
    client = provider.async_client()
    allocations = []
    create_error = DaytonaError(
        f"Sandbox {_ID} failed to start with state: SandboxState.BUILD_FAILED"
        if named
        else "gateway failed before returning sandbox",
        status_code=None if named else 502,
    )
    fresh = Provider().box

    async def create(params, timeout):
        allocations.append(provider.box.state)
        if len(allocations) == 1:
            raise create_error
        return fresh

    client.create = create
    env = environment(None)
    env._client_manager.get_client.return_value = client
    env._ensure_lease_renewal = lambda: None
    params = SimpleNamespace(labels={})
    work = asyncio.create_task(env._create_sandbox(params))
    if cancel:
        await asyncio.wait_for(polled.wait(), timeout=1)
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work
    else:
        with pytest.raises(SandboxStartupError) as raised:
            await work
        assert raised.value.__cause__ is create_error
    assert len(allocations) == 1
    assert env._sandbox is None
    assert env._create_outcome_unknown
    assert len(isolated.entries) == len(ledger_module.read_leftovers()) == 1
    with pytest.raises(SandboxStartupError):
        await env._create_sandbox(params)
    assert len(allocations) == 1
    provider.pending = False
    monkeypatch.setattr(daytona_delete, "DELETE_TIMEOUT_SEC", 1)
    await env._sweep_create_attempts()
    assert not isolated.entries
    assert not env._create_outcome_unknown
    await env._create_sandbox(params)
    assert len(allocations) == 2
    assert allocations[-1] == "destroyed"
    assert env._sandbox is fresh


def test_exit_deadline_is_shared_across_real_sdk_deletes(monkeypatch, isolated):
    """Guards 9b0320f: each pending allocation cannot receive a fresh exit budget."""
    from benchflow.sandbox import daytona_delete

    monkeypatch.setattr(daytona_delete, "DELETE_TIMEOUT_SEC", 0.05)
    providers = [Provider(sync=True, pending=True) for _ in range(3)]
    boxes = {}
    for i, provider in enumerate(providers):
        provider.box.id = f"box-{i}"
        boxes[provider.box.id] = provider.box
        isolated.open_sandbox(None, provider.box.id, "test", managed="1")
    client = SimpleNamespace(
        get=lambda name, request_timeout=None: boxes[name],
        delete=lambda box, **kwargs: box.delete(**kwargs),
    )
    left = isolated.exit_sweep(lambda: client, budget_sec=0.02)
    assert len(left) == len(ledger_module.read_leftovers()) == 3
    assert sum(provider.events.count("delete") for provider in providers) == 1


def test_expired_exit_budget_never_sends_unbounded_zero_timeout(isolated):
    """Guards 9b0320f: timeout zero disables the SDK bound, so do no I/O instead."""
    from unittest.mock import Mock

    isolated.open_sandbox(None, _ID, "test", managed="1")
    client = Mock()
    assert len(isolated.exit_sweep(lambda: client, budget_sec=0)) == 1
    client.get.assert_not_called()
    client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_timeout_reconciliation_cancel_keeps_the_late_allocation(
    monkeypatch, isolated
):
    """Guards 9b0320f: cancellation inside timeout recovery cannot lose the late box."""
    monkeypatch.setattr(native, "_STARTUP_HARD_TIMEOUT_BUFFER_SEC", 0.01)
    provider = Provider(pending=True)
    polled = asyncio.Event()
    provider.on_poll = polled.set
    client = provider.async_client()

    async def create(params, timeout):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return provider.box

    client.create = create
    env = environment(None)
    env.task_env_config.build_timeout_sec = 0
    env._client_manager.get_client.return_value = client
    work = asyncio.create_task(env._create_sandbox(SimpleNamespace(labels={})))
    await asyncio.wait_for(polled.wait(), timeout=1)
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await work
    assert env._sandbox is None
    assert env._create_outcome_unknown
    assert len(isolated.entries) == len(ledger_module.read_leftovers()) == 1
    assert not next(iter(isolated.entries.values())).busy
    provider.pending = False
    await env._sweep_create_attempts()
    assert not env._create_outcome_unknown
    assert not isolated.entries


def test_exit_deadline_bounds_the_whole_paginated_lookup(isolated):
    """Guards 9b0320f: the SDK's per-page request timeout cannot renew the exit budget."""
    import time

    pages = []

    def listed(query, request_timeout=None):
        assert 0 < request_timeout <= 0.05
        for i in range(10):
            pages.append(i)
            time.sleep(0.3)
            yield SimpleNamespace(id=f"box-{i}")

    client = SimpleNamespace(list=listed)
    isolated.open_attempt(
        None, {"benchflow.create-attempt": "pending"}, "test", managed="1"
    )
    assert len(isolated.exit_sweep(lambda: client, budget_sec=0.05)) == 1
    assert len(pages) < 10
    assert ledger_module.read_leftovers()
