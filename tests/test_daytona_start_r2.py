"""The Daytona start path, round 2 (DAYTONA-START "Round 2").

Each reproducer from both round-1 reviews is a case here, written against
entry points that exist before and after the fix (the run preflight, the
create, ``stop()``, the eval-start auto-reap and ``bench sandbox cleanup``),
so this file runs unchanged on 8022f8f9a, where they fail, and on the fix.

The rules:

1. Automatic deletion needs positive proof of abandonment: the eval-start
   auto-reap deletes only a sandbox whose run lease lapsed. Age and activity
   are never proof.
2. Only a rejected key refuses: 401 (by status, before any text), or 403
   without a quota message.
3. Capacity patience is a time budget, and Retry-After (seconds or an HTTP
   date) is honoured within it.
4. Every create is accounted for: an attempt whose outcome is unknown is
   watched by its label, and ``stop()`` returns only once it is found and
   deleted or settled.

SDK errors are the real ``daytona.common.errors`` classes when the SDK is
installed, else fakes with the same module, class name and status.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.doctor import DoctorProbes, check_daytona
from benchflow.sandbox import daytona as daytona_module
from benchflow.sandbox import daytona_ledger, daytona_start
from benchflow.sandbox.daytona import DaytonaSandbox
from benchflow.sandbox.daytona_strategies import _DaytonaDirect
from benchflow.sandbox.protocol import SandboxStartupError


def _err(
    name: str,
    message: str,
    status: int | None = None,
    headers: Mapping[str, str] | None = None,
) -> Exception:
    """A Daytona SDK error: the real class when installed."""
    try:
        from daytona.common import errors

        cls = getattr(errors, name)
    except (ImportError, AttributeError):
        cls = type(name, (Exception,), {"__module__": "daytona.common.errors"})
        exc = cls(message)
        exc.status_code = status  # type: ignore[attr-defined]
        exc.headers = dict(headers or {})  # type: ignore[attr-defined]
        return exc
    return cls(message, status_code=status, headers=dict(headers or {}))


def _rate_limited(retry_after: str | None = None) -> Exception:
    headers = {"Retry-After": retry_after} if retry_after is not None else None
    return _err("DaytonaRateLimitError", "Too Many Requests", 429, headers)


# ── Rule 2: only a rejected key refuses ─────────────────────────────────

KEY = "dtn_SECRETround2key_0123456789"


def _probes(tmp_path: Path, error: Exception | None) -> DoctorProbes:
    def daytona_check(environ: Mapping[str, str], timeout: float) -> None:
        if error is not None:
            raise error

    return DoctorProbes(
        environ={"DAYTONA_API_KEY": KEY},
        origins={"DAYTONA_API_KEY": "env"},
        home=tmp_path,
        dist_version=lambda dist: "0.207.0",
        daytona_check=daytona_check,
        sleep=lambda _s: None,
    )


@pytest.mark.parametrize(
    "error",
    [
        # Review A M1: request failed with these statuses.
        *(
            _err("DaytonaError", "request failed", status)
            for status in (400, 404, 408, 409, 418)
        ),
        # Review B M1.
        _err("DaytonaNotFoundError", "temporary edge route missing", 404),
        _err("DaytonaError", "upstream returned an empty response"),
        # A 403 quota refusal is capacity, not the key.
        _err("DaytonaAuthorizationError", "Total CPU limit exceeded", 403),
    ],
    ids=lambda e: f"{type(e).__name__}-{getattr(e, 'status_code', None)}",
)
def test_every_outcome_but_a_rejected_key_lets_the_run_start(tmp_path, error):
    row = check_daytona(
        _probes(tmp_path, error), required=True, offline=False, for_run=True
    )
    assert row.status == "warn"
    assert KEY not in row.summary


@pytest.mark.parametrize(
    "error",
    [
        _err("DaytonaAuthenticationError", "Unauthorized", 401),
        _err("DaytonaError", "request failed", 401),
        _err("DaytonaError", "request failed", 403),
        _err("DaytonaAuthorizationError", "Forbidden", 403),
        # Review B M1 / review A M5: a 401 is the key whatever its text says.
        _err(
            "DaytonaAuthenticationError",
            "Unauthorized: quota API credential invalid",
            401,
        ),
    ],
    ids=lambda e: f"{type(e).__name__}-{getattr(e, 'status_code', None)}",
)
def test_a_rejected_key_refuses(tmp_path, error):
    row = check_daytona(
        _probes(tmp_path, error), required=True, offline=False, for_run=True
    )
    assert row.status == "fail"
    assert "rejected DAYTONA_API_KEY" in row.summary


def test_401_with_quota_text_is_the_key_not_capacity():
    error = _err(
        "DaytonaAuthenticationError",
        "Unauthorized: account quota lookup unavailable",
        401,
    )
    assert daytona_start.error_kind(error) == "config"
    assert daytona_start.key_rejected_status(error) == 401


# ── Create against a fake client ────────────────────────────────────────


class _Client:
    """``create`` raises the scripted errors in order, then returns a
    sandbox; ``remote`` holds what the server made, by attempt label."""

    def __init__(self, errors: list[Exception] | None = None) -> None:
        self.errors = list(errors or [])
        self.creates = 0
        self.lookups = 0
        self.remote: list[SimpleNamespace] = []
        self.made = SimpleNamespace(id="sb-ok", labels={}, delete=AsyncMock())
        self.get = AsyncMock()
        self.params: list = []

    async def create(self, params, timeout):
        self.creates += 1
        self.params.append(dict(params.labels))
        if self.errors:
            raise self.errors.pop(0)
        return self.made

    def land(self, labels: dict, sandbox_id: str) -> SimpleNamespace:
        """The server finishes an earlier create: its sandbox appears."""
        sandbox = SimpleNamespace(id=sandbox_id, labels=dict(labels))

        async def delete(*_a, **_kw):
            if sandbox in self.remote:
                self.remote.remove(sandbox)

        sandbox.delete = AsyncMock(side_effect=delete)
        self.remote.append(sandbox)
        return sandbox

    async def list(self, query=None):
        self.lookups += 1
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
    sandbox.logger = logging.getLogger("test.daytona.start_r2")
    # What stop() reads (DaytonaSandbox.stop -> the direct strategy).
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


@pytest.fixture
def no_waits(monkeypatch):
    monkeypatch.setattr(daytona_start, "wait_sec", lambda *_: 0.0)
    # Round 3: the in-place look after a timeout and the ledger's sweeps.
    monkeypatch.setattr(daytona_module, "_CREATE_SETTLE_SEC", 0.05)
    monkeypatch.setattr(daytona_module, "_CREATE_WATCH_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(daytona_ledger, "SETTLE_SEC", 0.05)
    monkeypatch.setattr(daytona_ledger, "SWEEP_INTERVAL_SEC", 0.01)


@pytest.fixture(autouse=True)
def _sdk_timeout_class(monkeypatch):
    """The create catches the SDK's own timeout class once the SDK is
    loaded; use the class these tests raise."""
    monkeypatch.setattr(
        daytona_module,
        "DaytonaTimeoutError",
        type(_err("DaytonaTimeoutError", "x")),
    )


# ── Rule 3: capacity patience is a time budget ──────────────────────────


def _state(exc, attempt, elapsed=1.0, upcoming=0.0):
    return SimpleNamespace(
        outcome=SimpleNamespace(exception=lambda: exc),
        attempt_number=attempt,
        seconds_since_start=elapsed,
        upcoming_sleep=upcoming,
    )


def test_capacity_keeps_trying_for_the_promised_ten_minutes():
    # Review B M3: attempt 20 at 400 s stopped.
    stop = daytona_start.create_retries_exhausted
    assert not stop(_state(_rate_limited(), 20, elapsed=400))
    assert not stop(_state(_rate_limited(), 200, elapsed=500, upcoming=30))
    # The budget, not a count, ends it; a wait past it stops now.
    assert stop(_state(_rate_limited(), 3, elapsed=590, upcoming=30))
    assert stop(_state(_rate_limited(), 1, elapsed=1, upcoming=3600))


@pytest.mark.asyncio
async def test_create_is_saved_by_attempt_21_inside_the_capacity_budget(no_waits):
    # Review A M2 / review B M3: 20 x 429 then success.
    client = _Client([_rate_limited("1")] * 20)
    sandbox = _sandbox(client)
    with contextlib.suppress(Exception):
        await sandbox._create_sandbox(params=_params())
    assert client.creates == 21
    assert sandbox._sandbox is client.made


def test_retry_after_is_not_shortened():
    # Review A M3 / review B M4: 120 s became 60 s.
    assert daytona_start.wait_sec(_rate_limited("120"), 1) >= 120


def test_http_date_retry_after_is_honoured():
    # Review A M3 / review B M4: a date 120 s ahead gave 1-2 s.
    future = format_datetime(datetime.now(UTC) + timedelta(seconds=120), usegmt=True)
    assert daytona_start.wait_sec(_rate_limited(future), 1) >= 118


def test_retry_after_in_the_past_falls_back_to_the_backoff():
    past = format_datetime(datetime.now(UTC) - timedelta(seconds=60), usegmt=True)
    assert 1.0 <= daytona_start.wait_sec(_rate_limited(past), 1) <= 2.0
    assert 1.0 <= daytona_start.wait_sec(_rate_limited("soon"), 1) <= 2.0


def test_doctor_never_holds_for_a_long_retry_after(tmp_path):
    """A read with no budget of its own (doctor's key check) stops instead
    of sleeping an hour, and never asks early."""
    slept: list[float] = []
    probes = _probes(tmp_path, _rate_limited("3600"))
    probes = DoctorProbes(
        environ=probes.environ,
        origins=probes.origins,
        home=probes.home,
        dist_version=probes.dist_version,
        daytona_check=probes.daytona_check,
        sleep=slept.append,
    )
    row = check_daytona(probes, required=True, offline=False)
    assert slept == []
    assert row.status == "fail"
    assert "rejected" not in row.summary


@pytest.mark.asyncio
async def test_toolbox_probe_never_holds_for_a_long_retry_after(monkeypatch):
    slept: list[float] = []

    class _Asyncio:
        """daytona.py's asyncio, with a sleep that only records."""

        def __getattr__(self, name):
            return getattr(asyncio, name)

        @staticmethod
        async def sleep(seconds):
            slept.append(seconds)

    monkeypatch.setattr(daytona_module, "asyncio", _Asyncio())
    sandbox = _sandbox(_Client())
    exec_ = AsyncMock(side_effect=_rate_limited("3600"))
    sandbox._sandbox = SimpleNamespace(id="sb", process=SimpleNamespace(exec=exec_))
    with pytest.raises(Exception, match="Too Many Requests"):
        await sandbox._toolbox_startup_env()
    assert exec_.await_count == 1
    assert all(s <= 60 for s in slept)


@pytest.mark.asyncio
async def test_401_with_quota_text_is_not_retried_at_create(no_waits):
    # Review A M5: 20 creates instead of one.
    client = _Client(
        [
            _err(
                "DaytonaAuthenticationError",
                "Unauthorized: account quota lookup unavailable",
                401,
            )
        ]
        * 25
    )
    sandbox = _sandbox(client)
    with pytest.raises(SandboxStartupError, match="rejected DAYTONA_API_KEY"):
        await sandbox._create_sandbox(params=_params())
    assert client.creates == 1


# ── Rule 4: every create is accounted for ───────────────────────────────


def _attempt(labels: dict) -> dict:
    return {"benchflow.create-attempt": labels["benchflow.create-attempt"]}


@pytest.mark.asyncio
async def test_timed_out_create_that_lands_after_the_lookup_is_deleted_by_stop(
    no_waits,
):
    # Review B M2 / review A M4: 1 create, 1 lookup, 0 deletes, sandbox left.
    client = _Client([_err("DaytonaTimeoutError", "request timed out")])
    sandbox = _sandbox(client)
    with pytest.raises(SandboxStartupError):
        await sandbox._create_sandbox(params=_params())
    assert client.creates == 1
    late = client.land(client.params[0], "late-server-sandbox")

    await sandbox.stop(delete=True)

    assert late.delete.await_count == 1
    assert client.remote == []


@pytest.mark.asyncio
async def test_post_that_lands_just_after_an_empty_lookup_is_deleted(no_waits):
    # Review A M4: the list answers empty, then the POST lands. Round 3 (r2
    # review A M3): the create keeps looking in place, deletes the landing
    # and retries, instead of losing the trial.
    client = _Client([_err("DaytonaTimeoutError", "create request timed out")])
    sandbox = _sandbox(client)
    original_list = client.list

    async def list_then_land(query=None):
        async for found in original_list(query):
            yield found
        if not client.remote and client.lookups == 1:
            asyncio.get_running_loop().call_soon(
                client.land, client.params[0], "created-server-side"
            )

    client.list = list_then_land
    await sandbox._create_sandbox(params=_params())
    assert sandbox._sandbox is client.made
    # Retried only after the landing was found and deleted, never blind.
    assert client.creates == 2
    assert client.remote == []
    assert not sandbox._create_outcome_unknown

    await sandbox.stop(delete=True)
    assert client.remote == []


@pytest.mark.asyncio
async def test_503_with_late_server_creation_is_deleted_before_stop_returns(
    no_waits,
):
    """Guards 9b0320f: no second allocation while the first 503 may still land."""
    client = _Client([_err("DaytonaError", "Service Unavailable", 503)])
    sandbox = _sandbox(client)
    with pytest.raises(SandboxStartupError):
        await sandbox._create_sandbox(params=_params())
    assert sandbox._sandbox is None and client.creates == 1
    first = client.land(client.params[0], "server-created-after-503")

    await sandbox.stop(delete=True)

    assert first.delete.await_count == 1
    assert client.made.delete.await_count == 0  # no overlapping trial was created
    assert client.remote == []


@pytest.mark.asyncio
async def test_a_late_duplicate_is_deleted_while_the_trial_runs(no_waits):
    """Guards 9b0320f: the late allocation stays watched without creating a duplicate."""
    client = _Client([_err("DaytonaError", "Bad Gateway", 502)])
    sandbox = _sandbox(client)
    with pytest.raises(SandboxStartupError):
        await sandbox._create_sandbox(params=_params())
    first = client.land(client.params[0], "landed-while-running")

    for _ in range(50):
        if first.delete.await_count:
            break
        await asyncio.sleep(0.01)

    assert first.delete.await_count == 1  # not left for the trial's length
    assert sandbox._sandbox is None and client.creates == 1
    assert not sandbox._create_outcome_unknown
    await sandbox.stop(delete=True)


@pytest.mark.asyncio
async def test_an_attempt_that_never_lands_settles_and_stop_returns(no_waits):
    """Guards 9b0320f: an unresolved create is retained and cannot allocate a replacement.

    stop() still returns after the settle window, with durable ownership.
    """
    client = _Client([_err("DaytonaError", "Service Unavailable", 503)])
    sandbox = _sandbox(client)
    with pytest.raises(SandboxStartupError):
        await sandbox._create_sandbox(params=_params())
    assert client.creates == 1

    start = time.monotonic()
    await sandbox.stop(delete=True)

    assert time.monotonic() - start < 5
    assert client.lookups >= 2  # at the failure, and again until it settled
    attempt = client.params[0]["benchflow.create-attempt"]
    assert f"attempt:{attempt}" in daytona_ledger.LEDGER.entries
    [record] = daytona_ledger.read_leftovers()
    assert record["labels"] == {"benchflow.create-attempt": attempt}


@pytest.mark.asyncio
async def test_a_refused_create_is_settled_by_one_empty_lookup(no_waits):
    """A 4xx means the API refused the create: nothing to watch."""
    client = _Client([_rate_limited()])
    sandbox = _sandbox(client)
    await sandbox._create_sandbox(params=_params())
    assert client.lookups == 1
    assert not daytona_ledger.LEDGER.entries
    await sandbox.stop(delete=True)
    assert client.lookups == 1


@pytest.mark.asyncio
async def test_found_attempts_are_deleted_and_retried(no_waits):
    """Controls kept from round 1: a found timeout is deleted, then retried."""
    client = _Client([_err("DaytonaTimeoutError", "start timed out")])
    original_create = client.create

    async def create(params, timeout):
        if client.creates == 0:
            client.land(params.labels, "timed-out-but-found")
        return await original_create(params, timeout)

    client.create = create
    sandbox = _sandbox(client)
    await sandbox._create_sandbox(params=_params())
    assert client.creates == 2
    assert sandbox._sandbox is client.made
    assert client.remote == []


# ── Rule 1: the run lease ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_every_create_carries_the_run_id_and_a_lease(no_waits):
    client = _Client()
    sandbox = _sandbox(client)
    await sandbox._create_sandbox(params=_params())
    [labels] = client.params
    assert labels["benchflow.managed"] == "1"
    assert labels["benchflow.run"]
    hard_timeout = 60 + daytona_module._STARTUP_HARD_TIMEOUT_BUFFER_SEC
    assert int(labels["benchflow.lease"]) >= time.time() + 3600 + hard_timeout - 5
    await sandbox.stop(delete=True)


@pytest.mark.asyncio
async def test_the_live_run_renews_its_lease_until_stop(no_waits, monkeypatch):
    monkeypatch.setattr(daytona_module, "_LEASE_RENEW_SEC", 0.01, raising=False)
    monkeypatch.setattr(daytona_module, "_HEARTBEAT_SEC", 0.01, raising=False)
    client = _Client()
    client.made.labels = {
        "benchflow.managed": "1",
        "benchflow.run": "r",
        "benchflow.lease": "1",
        "code-toolbox-language": "python",
    }
    client.made.set_labels = AsyncMock()
    sandbox = _sandbox(client)
    await sandbox._create_sandbox(params=_params())
    for _ in range(50):
        if client.made.set_labels.await_count >= 2:
            break
        await asyncio.sleep(0.01)
    renewed = client.made.set_labels.await_args.args[0]
    assert int(renewed["benchflow.lease"]) > time.time() + 3000
    # set_labels replaces every label: the rest are kept.
    assert renewed["benchflow.managed"] == "1"
    assert renewed["code-toolbox-language"] == "python"

    await sandbox.stop(delete=True)
    count = client.made.set_labels.await_count
    await asyncio.sleep(0.05)
    assert client.made.set_labels.await_count == count


@pytest.mark.asyncio
async def test_renewal_keeps_ownership_when_the_handle_has_no_labels(
    no_waits, monkeypatch
):
    """A handle without labels must not stop the renewal (the lease would
    lapse under a live run) nor drop the ownership label."""
    monkeypatch.setattr(daytona_module, "_LEASE_RENEW_SEC", 0.01, raising=False)
    monkeypatch.setattr(daytona_module, "_HEARTBEAT_SEC", 0.01, raising=False)
    client = _Client()
    client.made.labels = None
    client.made.set_labels = AsyncMock()
    sandbox = _sandbox(client)
    await sandbox._create_sandbox(params=_params())
    for _ in range(50):
        if client.made.set_labels.await_count:
            break
        await asyncio.sleep(0.01)
    renewed = client.made.set_labels.await_args.args[0]
    assert renewed["benchflow.managed"] == "1"
    attempt = client.params[0]["benchflow.create-attempt"]
    assert renewed["benchflow.create-attempt"] == attempt
    assert int(renewed["benchflow.lease"]) > time.time() + 3000
    await sandbox.stop(delete=True)


# ── Rule 1: the eval-start auto-reap, against a fake server ─────────────


class _Server:
    """A fake Daytona list that applies the query's label and creation
    filters, as the API does, and records deletes."""

    def __init__(self, sandboxes: Sequence[SimpleNamespace]) -> None:
        self.sandboxes = list(sandboxes)
        self.deleted: list[str] = []
        self.queries: list = []

    def list(self, query=None):
        self.queries.append(query)
        wanted = dict(getattr(query, "labels", None) or {})
        before = getattr(query, "created_at_before", None)
        for sb in self.sandboxes:
            if any(sb.labels.get(k) != v for k, v in wanted.items()):
                continue
            created = datetime.fromisoformat(sb.created_at)
            if before is not None and created >= before:
                continue
            yield sb

    def delete(self, sb, timeout=60, wait=False):
        self.deleted.append(sb.id)


def _sb(
    sandbox_id: str,
    managed: str | None,
    *,
    age_min: float,
    idle_min: float | None,
    lease_in_min: float | None,
    state: str = "STARTED",
) -> SimpleNamespace:
    now = datetime.now(UTC)
    labels: dict[str, str] = {}
    if managed is not None:
        labels["benchflow.managed"] = managed
    if lease_in_min is not None:
        labels["benchflow.run"] = f"run-{sandbox_id}"
        labels["benchflow.lease"] = str(int(time.time() + lease_in_min * 60))
    return SimpleNamespace(
        id=sandbox_id,
        labels=labels,
        state=state,
        created_at=(now - timedelta(minutes=age_min)).isoformat(),
        last_activity_at=(
            None
            if idle_min is None
            else (now - timedelta(minutes=idle_min)).isoformat()
        ),
        updated_at=(now - timedelta(minutes=age_min)).isoformat(),
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


def test_a_second_eval_never_reaps_live_runs_with_no_owner_set(monkeypatch):
    """Two live runs, one longer than 24 h, and another user's run, all with
    the owner unset (the shared ``benchflow.managed=1``): none is deleted.
    Only the run whose lease lapsed and that Daytona stopped is (round 3,
    rule A); a STARTED one with a lapsed lease is left."""
    monkeypatch.delenv("BENCHFLOW_DAYTONA_OWNER", raising=False)
    server = _Server(
        [
            # Run A: 30 h long, quiet for 40 min, lease renewed 5 min ago.
            _sb("run-a-30h", "1", age_min=1800, idle_min=40, lease_in_min=55),
            # Run A again: no activity timestamp at all.
            _sb("run-a-no-activity", "1", age_min=1500, idle_min=None, lease_in_min=60),
            # Run B: started 5 minutes ago.
            _sb("run-b-fresh", "1", age_min=5, idle_min=1, lease_in_min=60),
            # Another user, no owner set either: 25 h, never active.
            _sb("other-user-25h", "1", age_min=1500, idle_min=None, lease_in_min=50),
            # A crashed run: its lease lapsed two hours ago, Daytona stopped it.
            _sb(
                "abandoned",
                "1",
                age_min=200,
                idle_min=190,
                lease_in_min=-120,
                state="STOPPED",
            ),
            # Lapsed but STARTED: a paused controller's command may still run.
            _sb("started-lapsed", "1", age_min=200, idle_min=0, lease_in_min=-120),
            # Made before leases: no proof either way, only reported.
            _sb("legacy-no-lease", "1", age_min=3000, idle_min=2000, lease_in_min=None),
            _sb("foreign", None, age_min=3000, idle_min=None, lease_in_min=-500),
        ]
    )

    _run_eval_start_reap(monkeypatch, server)

    assert server.deleted == ["abandoned"]


def test_a_second_eval_never_reaps_a_shared_owners_live_runs(monkeypatch):
    """Review A's reproducer: one shared explicit owner, a 1,500-minute
    STARTED sandbox 31 min idle and one with no activity timestamp."""
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "review-shared-owner")
    owner = "1:review-shared-owner"
    server = _Server(
        [
            _sb("live-long-trial", owner, age_min=1500, idle_min=31, lease_in_min=58),
            _sb(
                "live-missing-activity",
                owner,
                age_min=1500,
                idle_min=None,
                lease_in_min=58,
            ),
            _sb("recently-active", owner, age_min=1500, idle_min=1, lease_in_min=58),
            _sb("started-5-min-ago", owner, age_min=5, idle_min=1, lease_in_min=60),
            _sb(
                "other-owner",
                "1:someone-else",
                age_min=1500,
                idle_min=45,
                lease_in_min=-200,
            ),
            _sb(
                "lapsed",
                owner,
                age_min=100,
                idle_min=90,
                lease_in_min=-11,
                state="STOPPED",
            ),
            _sb("lapsed-started", owner, age_min=100, idle_min=0, lease_in_min=-11),
        ]
    )

    _run_eval_start_reap(monkeypatch, server)

    assert server.deleted == ["lapsed"]


def test_a_lease_inside_the_grace_is_not_lapsed(monkeypatch):
    monkeypatch.delenv("BENCHFLOW_DAYTONA_OWNER", raising=False)
    server = _Server(
        [
            _sb(
                "skewed-clock",
                "1",
                age_min=200,
                idle_min=100,
                lease_in_min=-5,
                state="STOPPED",
            )
        ]
    )
    _run_eval_start_reap(monkeypatch, server)
    assert server.deleted == []


# ── Rule 1: bench sandbox cleanup (explicit, operator-invoked) ──────────


def test_cleanup_keeps_a_live_lease_and_deletes_a_lapsed_one(monkeypatch, capsys):
    monkeypatch.delenv("BENCHFLOW_DAYTONA_OWNER", raising=False)
    server = _Server(
        [
            _sb("live-30h", "1", age_min=1800, idle_min=300, lease_in_min=50),
            _sb(
                "lapsed-young",
                "1",
                age_min=90,
                idle_min=80,
                lease_in_min=-30,
                state="STOPPED",
            ),
            # Lapsed but STARTED and young: the age rule keeps it.
            _sb("lapsed-started-young", "1", age_min=90, idle_min=0, lease_in_min=-30),
            # No lease: the age and activity rules still apply here.
            _sb("legacy-old-idle", "1", age_min=3000, idle_min=2000, lease_in_min=None),
            _sb("legacy-old-active", "1", age_min=3000, idle_min=1, lease_in_min=None),
        ]
    )
    from benchflow.cli import main as cli_main

    monkeypatch.setattr(cli_main, "_daytona_client_or_exit", lambda **_: server)
    monkeypatch.setattr(cli_main, "_cleanup_daytona_snapshots", lambda *a, **k: None)

    cli_main._cleanup_daytona_sandboxes(dry_run=False, max_age_minutes=1440)

    assert sorted(server.deleted) == ["lapsed-young", "legacy-old-idle"]
    out = capsys.readouterr().out
    assert "Warning" in out and "age" in out
    assert "live run's lease" in out

    server.deleted.clear()
    cli_main._cleanup_daytona_sandboxes(
        dry_run=False, max_age_minutes=1440, all_mine=True
    )
    # --all: every sandbox of this operator except the live lease (round 3)
    # and a STARTED one (round 4): only the stopped, lapsed one goes.
    assert server.deleted == ["lapsed-young"]

    server.deleted.clear()
    cli_main._cleanup_daytona_sandboxes(
        dry_run=False, max_age_minutes=1440, names=frozenset({"live-30h"})
    )
    assert "live-30h" in server.deleted  # named with --id
