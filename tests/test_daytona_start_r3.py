"""The Daytona start path, round 3 (DAYTONA-START "Round 3").

Each reproducer from both round-2 reviews is a case here. The file runs
unchanged on 1441921a8, where the cases fail, and on the fix. It goes through
entry points that exist at both commits: the eval-start auto-reap,
``bench sandbox cleanup``, ``DaytonaSandbox._create_sandbox`` and ``stop()``.
Only the job-exit cases import the new ledger module, inside the test.

The decision: liveness is judged by Daytona, never by a clock
outside it.

A. Automatic reaping never deletes a STARTED sandbox. It deletes only one
   that Daytona has stopped, archived or errored, whose lease lapsed, and
   that is this owner's.
B. Daytona stops an abandoned sandbox for us. Every sandbox is created with
   Daytona's inactivity stop, and a living controller sends an activity
   heartbeat.
C. Explicit cleanup never deletes a sandbox with a live lease unless it is
   named with ``--id``.
D. The attempt ledger lives until the job exits; leftovers go to a file that
   cleanup reads.
E. A sandbox that fails while starting is deleted and replaced by a fresh
   create.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import daytona as daytona_module
from benchflow.sandbox import daytona_start
from benchflow.sandbox.daytona import DaytonaSandbox
from benchflow.sandbox.daytona_strategies import _DaytonaDirect

# ── Fakes shared with round 2 ───────────────────────────────────────────


def _err(name: str, message: str, status: int | None = None) -> Exception:
    try:
        from daytona.common import errors

        cls = getattr(errors, name)
    except (ImportError, AttributeError):
        cls = type(name, (Exception,), {"__module__": "daytona.common.errors"})
        exc = cls(message)
        exc.status_code = status  # type: ignore[attr-defined]
        exc.headers = {}  # type: ignore[attr-defined]
        return exc
    return cls(message, status_code=status, headers={})


class _Client:
    """``create`` raises the scripted errors in order, then returns a
    sandbox; ``remote`` holds what the server made, by attempt label."""

    def __init__(self, errors: list[Exception] | None = None) -> None:
        self.errors = list(errors or [])
        self.creates = 0
        self.lookups = 0
        self.list_error: Exception | None = None
        self.remote: list[SimpleNamespace] = []
        self.made = SimpleNamespace(
            id="sb-ok",
            labels={},
            delete=AsyncMock(),
            refresh_activity=AsyncMock(),
            set_labels=AsyncMock(),
        )
        self.get = AsyncMock()
        self.params: list = []

    async def create(self, params, timeout):
        self.creates += 1
        self.params.append(dict(params.labels))
        if self.errors:
            raise self.errors.pop(0)
        return self.made

    def land(self, labels: dict, sandbox_id: str) -> SimpleNamespace:
        sandbox = SimpleNamespace(id=sandbox_id, labels=dict(labels))

        async def delete(*_a, **_kw):
            if sandbox in self.remote:
                self.remote.remove(sandbox)

        sandbox.delete = AsyncMock(side_effect=delete)
        self.remote.append(sandbox)
        return sandbox

    async def list(self, query=None, request_timeout=None):
        self.lookups += 1
        if self.list_error is not None:
            raise self.list_error
        wanted = dict(getattr(query, "labels", None) or {})
        for sandbox in list(self.remote):
            if all(sandbox.labels.get(k) == v for k, v in wanted.items()):
                yield sandbox


def _sandbox(client) -> DaytonaSandbox:
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox._sandbox = None
    sandbox._create_attempts = 0
    sandbox.rollout_paths = None
    sandbox._client_manager = SimpleNamespace(get_client=AsyncMock(return_value=client))
    sandbox.task_env_config = SimpleNamespace(
        build_timeout_sec=60,
        cpus=1,
        memory_mb=1024,
        storage_mb=1024,
        docker_image=None,
        env={},
    )
    sandbox.logger = logging.getLogger("test.daytona.start_r3")
    sandbox._runtime_kit_state = None
    sandbox._service_kits = {}
    sandbox._exec_generation = 0
    sandbox._kept_identity = None
    sandbox._kwargs = {}
    sandbox._persistent_env = {}
    sandbox.environment_name = "task"
    sandbox.environment_dir = Path("environment")
    sandbox._strategy = _DaytonaDirect(sandbox)
    return sandbox


def _params() -> SimpleNamespace:
    return SimpleNamespace(labels={"benchflow.managed": "1"})


def _ledger():
    """The round-3 ledger module, or None before it existed."""
    try:
        return importlib.import_module("benchflow.sandbox.daytona_ledger")
    except ImportError:
        return None


class _Clock:
    """``time`` for the Daytona modules, with a monotonic clock that can be
    moved forward without touching asyncio's own clock."""

    def __init__(self) -> None:
        self.offset = 0.0

    def __getattr__(self, name):
        return getattr(time, name)

    def monotonic(self) -> float:
        return time.monotonic() + self.offset


@pytest.fixture
def fast(monkeypatch):
    """No create backoff, short watch and settle times, a movable clock."""
    monkeypatch.setattr(daytona_start, "wait_sec", lambda *_: 0.0)
    monkeypatch.setattr(daytona_module, "_CREATE_WATCH_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(daytona_module, "_CREATE_SETTLE_SEC", 0.05)
    clock = _Clock()
    monkeypatch.setattr(daytona_module, "time", clock)
    ledger = _ledger()
    if ledger is not None:
        monkeypatch.setattr(ledger, "SWEEP_INTERVAL_SEC", 0.01)
        monkeypatch.setattr(ledger, "SETTLE_SEC", 0.05)
        monkeypatch.setattr(ledger, "time", clock)
    monkeypatch.setattr(
        daytona_module, "DaytonaTimeoutError", type(_err("DaytonaTimeoutError", "x"))
    )
    return clock


async def _until(check, *, seconds: float = 2.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return True
        await asyncio.sleep(0.01)
    return check()


# ── H1 / rule A: the auto-reap never deletes a STARTED sandbox ─────────


class _Server:
    """A fake Daytona list applying the label and creation filters."""

    def __init__(self, sandboxes) -> None:
        self.sandboxes = list(sandboxes)
        self.deleted: list[str] = []

    def list(self, query=None, request_timeout=None):
        wanted = dict(getattr(query, "labels", None) or {})
        before = getattr(query, "created_at_before", None)
        for sb in self.sandboxes:
            if any(sb.labels.get(k) != v for k, v in wanted.items()):
                continue
            if before is not None and sb.created_at >= before:
                continue
            yield sb

    def delete(self, sb, timeout=60, wait=False):
        self.deleted.append(sb.id)


def _resource(sandbox_id: str, *, lease: int, owner: str, state: str = "STARTED"):
    now = datetime.now(UTC)
    managed = f"1:{owner}" if owner else "1"
    labels = {
        "benchflow.managed": managed,
        "benchflow.run": "other-live-run",
        "benchflow.lease": str(lease),
    }
    if owner:
        labels["benchflow.owner"] = owner
    return SimpleNamespace(
        id=sandbox_id,
        state=state,
        labels=labels,
        created_at=now - timedelta(hours=30),
        last_activity_at=now,
    )


def _run_eval_start_reap(monkeypatch, server: _Server) -> None:
    monkeypatch.setattr(daytona_module, "build_sync_client", lambda *a, **k: server)

    class _Inline:
        def __init__(self, target, **_kw):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr("benchflow.evaluation.threading.Thread", _Inline)
    monkeypatch.delenv("BENCHFLOW_DAYTONA_AUTO_REAP", raising=False)
    from benchflow.evaluation import Evaluation

    evaluation = Evaluation.__new__(Evaluation)
    evaluation._config = SimpleNamespace(environment="daytona")
    evaluation._maybe_start_daytona_reap()


@pytest.mark.parametrize("owner", ["", "same-owner"])
@pytest.mark.parametrize(
    "scenario", ["controller-paused", "renewer-network-loss", "reaper-clock-ahead"]
)
def test_a_started_sandbox_with_a_lapsed_lease_is_never_auto_reaped(
    monkeypatch, owner, scenario
):
    """Review B H1 / review A H1, with the unpatched production constants: a
    30 h STARTED sandbox whose controller paused or lost its network 601 s
    past the grace, or whose fresh renewal was written by a clock 4,201 s
    behind the reaper's. Its remote command may still run."""
    if owner:
        monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", owner)
    else:
        monkeypatch.delenv("BENCHFLOW_DAYTONA_OWNER", raising=False)
    now = int(time.time())
    lease = now - 601 - 600
    if scenario == "reaper-clock-ahead":
        lease = now - 4201 + 3600
    server = _Server([_resource("still-running", lease=lease, owner=owner)])

    _run_eval_start_reap(monkeypatch, server)

    assert server.deleted == []


@pytest.mark.parametrize("state", ["STOPPED", "ARCHIVED", "ERROR", "BUILD_FAILED"])
def test_a_sandbox_daytona_stopped_with_a_lapsed_lease_is_reaped(monkeypatch, state):
    """The positive control: once Daytona has taken it out of service."""
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "same-owner")
    lapsed = int(time.time()) - 2000
    server = _Server(
        [
            _resource("stopped", lease=lapsed, owner="same-owner", state=state),
            _resource("live", lease=int(time.time()) + 3600, owner="same-owner"),
            _resource(
                "stopped-live-lease",
                lease=int(time.time()) + 3600,
                owner="same-owner",
                state=state,
            ),
            _resource("other-owner", lease=lapsed, owner="someone", state=state),
        ]
    )

    _run_eval_start_reap(monkeypatch, server)

    assert server.deleted == ["stopped"]


# ── Rule C: explicit cleanup and live leases ────────────────────────────


def _cleanup(monkeypatch, server, **kwargs) -> None:
    from benchflow.cli import main as cli_main

    monkeypatch.setattr(cli_main, "_daytona_client_or_exit", lambda **_: server)
    monkeypatch.setattr(cli_main, "_cleanup_daytona_snapshots", lambda *a, **k: None)
    cli_main._cleanup_daytona_sandboxes(dry_run=False, max_age_minutes=0, **kwargs)


def test_cleanup_all_never_deletes_a_live_lease(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "me")
    live = _resource("live", lease=int(time.time()) + 3600, owner="me")
    old = _resource("old-unleased", lease=0, owner="me", state="STOPPED")
    del old.labels["benchflow.lease"]
    server = _Server([live, old])

    _cleanup(monkeypatch, server, all_mine=True)

    assert server.deleted == ["old-unleased"]


def test_cleanup_deletes_a_live_lease_only_when_named(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "me")
    server = _Server(
        [
            _resource("live-a", lease=int(time.time()) + 3600, owner="me"),
            _resource("live-b", lease=int(time.time()) + 3600, owner="me"),
        ]
    )

    _cleanup(monkeypatch, server)
    assert server.deleted == []

    _cleanup(monkeypatch, server, names=frozenset({"live-b"}))
    assert server.deleted == ["live-b"]


# ── Rule B: Daytona's inactivity stop and the activity heartbeat ───────


def test_every_sandbox_is_created_with_the_inactivity_stop(tmp_path):
    """Round 2 left Daytona's auto-stop at 24 h, so an abandoned STARTED
    sandbox stayed up for a day; it is 15 min now."""
    pytest.importorskip("daytona")
    from benchflow.task.config import SandboxConfig

    (tmp_path / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    sandbox = DaytonaSandbox(
        environment_dir=tmp_path,
        environment_name="t",
        session_id="s",
        rollout_paths=None,
        task_env_config=SandboxConfig(),
    )
    assert sandbox._auto_stop_interval == 15


@pytest.mark.asyncio
async def test_a_living_controller_sends_the_activity_heartbeat(fast, monkeypatch):
    """set_labels is not activity at Daytona (measured, R3.3); the heartbeat
    is refresh_activity(), and it stops at stop()."""
    monkeypatch.setattr(daytona_module, "_HEARTBEAT_SEC", 0.01, raising=False)
    monkeypatch.setattr(daytona_module, "_LEASE_RENEW_SEC", 3600, raising=False)
    client = _Client()
    sandbox = _sandbox(client)
    await sandbox._create_sandbox(params=_params())

    assert await _until(lambda: client.made.refresh_activity.await_count >= 2)
    await sandbox.stop(delete=True)
    count = client.made.refresh_activity.await_count
    await asyncio.sleep(0.05)
    assert client.made.refresh_activity.await_count == count


# ── M1 / rule D: the attempt ledger lives until the job exits ──────────


async def _create_after_503(fast) -> tuple[_Client, DaytonaSandbox]:
    client = _Client([_err("DaytonaError", "Service Unavailable", 503)])
    sandbox = _sandbox(client)
    # Guards 9b0320f: an unanswered create must reconcile before replacement.
    from benchflow.sandbox.protocol import SandboxStartupError

    with pytest.raises(SandboxStartupError, match="Service Unavailable"):
        await sandbox._create_sandbox(params=_params())
    assert client.creates == 1 and sandbox._sandbox is None
    assert sandbox._create_outcome_unknown
    return client, sandbox


@pytest.mark.asyncio
async def test_a_landing_after_the_90_s_window_is_deleted_while_the_job_lives(fast):
    """Review B M1 / review A M2 (a): the watch dropped the attempt at its
    settle deadline; its sandbox then landed and stayed."""
    client, sandbox = await _create_after_503(fast)
    fast.offset += 100  # past the 90 s window, asyncio's clock untouched
    await asyncio.sleep(0.1)  # sweeps run past the deadline
    late = client.land(client.params[0], "landing-after-90-seconds")

    assert await _until(lambda: late.delete.await_count == 1)
    assert client.remote == []
    await sandbox.stop(delete=True)


@pytest.mark.asyncio
async def test_a_failed_delete_at_the_window_is_retried(fast):
    """Review B M1 / review A M2 (b): the delete failed at the deadline and
    the attempt was dropped although connectivity came back."""
    client, sandbox = await _create_after_503(fast)
    late = client.land(client.params[0], "delete-temporarily-unavailable")
    remove = late.delete.side_effect
    late.delete.side_effect = ConnectionResetError("network loss")
    fast.offset += 100
    assert await _until(lambda: late.delete.await_count >= 1)
    await asyncio.sleep(0.05)
    late.delete.side_effect = remove  # connectivity recovered

    await sandbox.stop(delete=True)
    assert await _until(lambda: client.remote == [])


@pytest.mark.asyncio
async def test_a_failed_lookup_at_the_window_is_retried(fast):
    """Review A M2 (c): the final lookup failed and the attempt was dropped."""
    client, sandbox = await _create_after_503(fast)
    client.list_error = ConnectionResetError("temporary list error")
    fast.offset += 100
    await asyncio.sleep(0.1)
    client.list_error = None
    late = client.land(client.params[0], "landed-after-list-error")

    await sandbox.stop(delete=True)
    assert await _until(lambda: client.remote == [])
    assert late.delete.await_count == 1


@pytest.mark.asyncio
async def test_job_exit_deletes_or_records_every_open_attempt(fast, monkeypatch):
    """At exit the ledger retries with the sync client; what it still cannot
    confirm is in the leftovers file, and `bench sandbox cleanup` deletes it."""
    ledger = _ledger()
    assert ledger is not None, "no job-exit ledger at this commit"
    monkeypatch.delenv("BENCHFLOW_DAYTONA_OWNER", raising=False)
    client, sandbox = await _create_after_503(fast)
    await sandbox.stop(delete=True)  # nothing landed yet: still open
    attempt = client.params[0]["benchflow.create-attempt"]
    assert [r["labels"] for r in ledger.read_leftovers()] == [
        {"benchflow.create-attempt": attempt}
    ]

    class Sync:
        def __init__(self, fail: bool) -> None:
            self.fail = fail
            self.deleted: list[str] = []
            self.remote = [
                SimpleNamespace(
                    id="landed-at-exit",
                    labels={"benchflow.create-attempt": attempt},
                )
            ]

        def list(self, query=None, request_timeout=None):
            wanted = dict(getattr(query, "labels", None) or {})
            return [
                s
                for s in self.remote
                if all(s.labels.get(k) == v for k, v in wanted.items())
            ]

        def delete(self, sb, timeout=60, wait=False):
            if self.fail:
                raise ConnectionResetError("still offline")
            self.deleted.append(sb.id)
            self.remote.remove(sb)

        def get(self, sandbox_id, request_timeout=None):
            return SimpleNamespace(id=sandbox_id)

    offline = Sync(fail=True)
    left = ledger.LEDGER.exit_sweep(
        lambda: offline, budget_sec=0.05, sleep=lambda _s: None
    )
    assert [e.key for e in left] == [f"attempt:{attempt}"]
    assert len(ledger.read_leftovers()) == 1

    online = Sync(fail=False)
    counts = ledger.cleanup_leftovers(online, "1")
    assert online.deleted == ["landed-at-exit"]
    assert counts["deleted"] == 1
    assert ledger.read_leftovers() == []


@pytest.mark.asyncio
async def test_a_failed_delete_at_stop_is_retried_and_recorded(fast):
    """The trial's own sandbox: stop()'s delete fails, the ledger retries it
    while the job lives, and the leftovers file names it until then."""
    ledger = _ledger()
    assert ledger is not None, "no job-exit ledger at this commit"
    client = _Client()
    # Not a gateway or transport error, so stop() itself does not repeat it.
    client.made.delete.side_effect = _err("DaytonaError", "delete refused", 409)
    sandbox = _sandbox(client)
    await sandbox._create_sandbox(params=_params())
    await sandbox.stop(delete=True)
    assert [r["sandbox_id"] for r in ledger.read_leftovers()] == ["sb-ok"]

    client.made.delete.side_effect = None
    assert await _until(lambda: ledger.read_leftovers() == [])


# ── M2 / rule E: a failed start gets a fresh sandbox ────────────────────

_ACCEPTANCE_START_FAILURES = [
    "Failed to create sandbox: Failure during waiting for sandbox to start: "
    "Sandbox f79f59cf-6b2c-4654-bc34-fc632a454cb9 failed to start with state: "
    'SandboxState.BUILD_FAILED, error reason: process "/bin/sh -c curl -LsSf '
    'https://astral.sh/uv/0.7.13/install.sh / sh" did not complete '
    "successfully: exit code: 1",
    "Failed to create sandbox: Failure during waiting for sandbox to start: "
    "Sandbox c35dba59-8d81-48a9-9f85-37760adc7daa failed to start with state: "
    "SandboxState.ERROR, error reason: Error response from daemon: failed to "
    "create task for container: failed to create shim task: OCI runtime create "
    "failed: failed to pre-register with sysbox-fs: failed to pre-register "
    "container with sysbox-fs: rpc error: code = DeadlineExceeded desc = "
    "context deadline exceeded",
]


@pytest.mark.parametrize(
    "message", _ACCEPTANCE_START_FAILURES, ids=["sympy-11618", "pypi-server"]
)
@pytest.mark.asyncio
async def test_a_failed_start_is_replaced_by_a_fresh_sandbox(fast, message):
    """Review B M2 / review A M1: ACCEPT-DAYTONA run 2's exact errors."""
    client = _Client([_err("DaytonaError", message)])
    failed = SimpleNamespace(id="failed-start", delete=AsyncMock())
    client.get.return_value = failed

    await sandbox_create(client)

    assert client.creates == 2
    assert failed.delete.await_count == 1
    assert len({p["benchflow.create-attempt"] for p in client.params}) == 2


async def sandbox_create(client) -> DaytonaSandbox:
    sandbox = _sandbox(client)
    await sandbox._create_sandbox(params=_params())
    assert sandbox._sandbox is client.made
    await sandbox.stop(delete=True)
    return sandbox


@pytest.mark.asyncio
async def test_a_failed_start_stops_after_three_creates(fast):
    message = _ACCEPTANCE_START_FAILURES[0]
    client = _Client([_err("DaytonaError", message) for _ in range(5)])
    client.get.return_value = SimpleNamespace(id="x", delete=AsyncMock())
    sandbox = _sandbox(client)
    with pytest.raises(Exception, match="BUILD_FAILED"):
        await sandbox._create_sandbox(params=_params())
    assert client.creates == 3


@pytest.mark.asyncio
async def test_a_rejected_key_still_refuses_after_one_create(fast):
    client = _Client([_err("DaytonaAuthenticationError", "Unauthorized", 401)] * 3)
    sandbox = _sandbox(client)
    with pytest.raises(Exception, match="rejected DAYTONA_API_KEY"):
        await sandbox._create_sandbox(params=_params())
    assert client.creates == 1


# ── Review A M3: a timeout reconciled later lets the start go on ───────


@pytest.mark.asyncio
async def test_a_timeout_reconciled_by_a_later_sweep_lets_the_create_retry(fast):
    client = _Client([_err("DaytonaTimeoutError", "create request timed out")])
    sandbox = _sandbox(client)
    with pytest.raises(Exception):  # noqa: B017 - the terminal startup error
        await sandbox._create_sandbox(params=_params())
    late = client.land(client.params[0], "late-timeout-then-confirmed-deleted")
    await sandbox._drain_create_attempts()
    assert late.delete.await_count == 1

    await sandbox._create_sandbox(params=_params())

    assert sandbox._sandbox is client.made and client.creates == 2
    await sandbox.stop(delete=True)


# ── Review B: the cancellation control (passes at both commits) ────────


@pytest.mark.asyncio
async def test_a_cancelled_create_that_lands_is_deleted_by_stop(fast):
    client = _Client()
    entered = asyncio.Event()
    result = asyncio.get_running_loop().create_future()

    async def create(params, timeout):
        client.creates += 1
        client.params.append(dict(params.labels))
        entered.set()
        return await result

    client.create = create
    sandbox = _sandbox(client)
    task = asyncio.create_task(sandbox._create_sandbox(params=_params()))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    result.set_exception(ConnectionError("request interrupted"))
    with pytest.raises(asyncio.CancelledError):
        await task
    late = client.land(client.params[0], "cancelled-create-landing")

    await sandbox.stop(delete=True)

    assert late.delete.await_count == 1 and client.remote == []
