"""The Daytona start path refuses only configuration errors (DAYTONA-START).

In the c210416e4 acceptance dry run, 33 PY8 trials were lost before they
started: the run preflight counted the Daytona SDK's import (1.2 s alone,
12-21 s with 192 processes starting at once on 32 vCPUs) inside a 15 s
network timeout and refused the job. The rule tested here: a rejected key
(401, or 403 without a quota message) fails fast and is not retried; a
timeout, 429, 5xx or quota refusal is retried with jittered backoff within a
budget and never refuses a run before it starts.

The SDK is faked at the client boundary, as in
``tests/test_daytona_failed_start_cleanup.py``: fake errors carry the SDK's
module name, class name, ``status_code`` and ``headers``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow import doctor as doctor_mod
from benchflow._utils.scoring import is_unrecoverable_startup_error
from benchflow.doctor import DoctorProbes, check_daytona
from benchflow.sandbox import daytona as daytona_module
from benchflow.sandbox import daytona_start
from benchflow.sandbox.daytona import DaytonaSandbox
from benchflow.sandbox.daytona_reaper import (
    reap_abandoned_sandboxes,
    reap_stale_sandboxes,
)
from benchflow.sandbox.protocol import SandboxStartupError


def _sdk_error(
    name: str,
    message: str,
    *,
    status: int | None = None,
    headers: Mapping[str, str] | None = None,
) -> Exception:
    """An error shaped like the Daytona SDK's (``daytona.common.errors``)."""
    cls = type(name, (Exception,), {"__module__": "daytona.common.errors"})
    exc = cls(message)
    exc.status_code = status  # type: ignore[attr-defined]
    exc.headers = dict(headers or {})  # type: ignore[attr-defined]
    return exc


def _rate_limited(retry_after: str | None = None) -> Exception:
    headers = {"Retry-After": retry_after} if retry_after is not None else None
    return _sdk_error(
        "DaytonaRateLimitError",
        "Failed to create sandbox: Too Many Requests",
        status=429,
        headers=headers,
    )


def _unavailable() -> Exception:
    return _sdk_error(
        "DaytonaError", "Failed to create sandbox: Service Unavailable", status=503
    )


def _rejected() -> Exception:
    return _sdk_error(
        "DaytonaAuthenticationError",
        "Failed to create sandbox: Unauthorized",
        status=401,
    )


# ── The classifier ──────────────────────────────────────────────────────


class _Urllib3Error(Exception):
    pass


_Urllib3Error.__module__ = "urllib3.exceptions"


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (TimeoutError("read"), "timeout"),
        (_sdk_error("DaytonaTimeoutError", "timed out"), "timeout"),
        (ConnectionResetError("reset"), "transient"),
        (_sdk_error("DaytonaConnectionError", "refused"), "transient"),
        (_Urllib3Error("pool"), "transient"),
        (_rate_limited(), "capacity"),
        (
            _sdk_error(
                "DaytonaAuthorizationError", "Total CPU limit exceeded", status=403
            ),
            "capacity",
        ),
        (
            _sdk_error(
                "DaytonaValidationError", "Total disk quota exceeded", status=400
            ),
            "capacity",
        ),
        (_sdk_error("DaytonaError", "no available runners"), "capacity"),
        (_rejected(), "config"),
        # A 401 is the key whatever its text says (round 2, rule 2).
        (
            _sdk_error("DaytonaAuthenticationError", "quota lookup failed", status=401),
            "config",
        ),
        (_sdk_error("DaytonaAuthorizationError", "Forbidden", status=403), "config"),
        (_unavailable(), "transient"),
        (_sdk_error("DaytonaError", "boom", status=500), "transient"),
        (
            _sdk_error("DaytonaError", "<html><title>502 Bad Gateway</title></html>"),
            "transient",
        ),
        (
            _sdk_error(
                "DaytonaError",
                "Failed to list sandboxes: HTTPSConnectionPool(host='app.daytona.io', "
                "port=443): Max retries exceeded with url: /api/sandbox",
            ),
            "transient",
        ),
        (
            _sdk_error(
                "DaytonaError",
                "Sandbox x failed to start with state: SandboxState.BUILD_FAILED, "
                "error reason: no capacity",
            ),
            # Round 3, rule E: a failed start is replaced by a fresh create.
            "start_failed",
        ),
        (
            _sdk_error("DaytonaError", "Snapshot build failed: failed to build image"),
            "other",
        ),
        (_sdk_error("DaytonaValidationError", "cpu must be >= 1", status=400), "other"),
        (RuntimeError("401 Unauthorized"), "other"),
        (asyncio.CancelledError(), "other"),
        (None, "other"),
    ],
)
def test_error_kind(exc, kind):
    assert daytona_start.error_kind(exc) == kind


def test_backoff_is_jittered_and_capped():
    for attempt in range(1, 10):
        step = min(30.0, 2.0 * 2 ** (attempt - 1))
        waits = {daytona_start.backoff_sec(attempt) for _ in range(50)}
        assert all(step / 2 <= w <= step for w in waits)
        assert len(waits) > 1  # trials that failed together spread out


def test_retry_after_is_honoured_for_capacity():
    assert daytona_start.wait_sec(_rate_limited("7"), 1) == 7.0
    # Not capped: the create budget decides whether it is worth waiting.
    assert daytona_start.wait_sec(_rate_limited("3600"), 1) == 3600.0
    # Never shorter than the backoff.
    assert 8.0 <= daytona_start.wait_sec(_rate_limited("1"), 4) <= 16.0
    # No header, or a date already past: the jittered backoff.
    assert 1.0 <= daytona_start.wait_sec(_rate_limited(), 1) <= 2.0
    date = "Wed, 21 Oct 2020 07:28:00 GMT"
    assert 1.0 <= daytona_start.wait_sec(_rate_limited(date), 1) <= 2.0
    # A 5xx's Retry-After is not read (only capacity waits are the API's).
    unavailable = _unavailable()
    unavailable.headers = {"Retry-After": "500"}  # type: ignore[attr-defined]
    assert daytona_start.wait_sec(unavailable, 1) <= 2.0


def _state(exc, attempt, elapsed=1.0, upcoming=0.0):
    return SimpleNamespace(
        outcome=SimpleNamespace(exception=lambda: exc),
        attempt_number=attempt,
        seconds_since_start=elapsed,
        upcoming_sleep=upcoming,
    )


def test_create_budgets_per_kind():
    stop = daytona_start.create_retries_exhausted
    # Capacity: a time budget, no attempt count.
    assert not stop(_state(_rate_limited(), 100))
    assert not stop(_state(_rate_limited(), 2, elapsed=570.0, upcoming=30.0))
    assert stop(_state(_rate_limited(), 2, elapsed=571.0, upcoming=30.0))
    assert not stop(_state(_unavailable(), 5))
    assert stop(_state(_unavailable(), 6))
    assert stop(_state(_unavailable(), 2, elapsed=290.0, upcoming=20.0))
    assert not stop(_state(TimeoutError(), 2))
    assert stop(_state(TimeoutError(), 3))
    should = daytona_start.create_should_retry
    assert should(_state(_rate_limited(), 1))
    assert should(_state(_unavailable(), 1))
    assert should(_state(TimeoutError(), 1))
    assert not should(_state(_rejected(), 1))
    assert not should(_state(asyncio.CancelledError(), 1))


# ── Create, against a fake client ───────────────────────────────────────


class _ScriptedClient:
    """``create`` raises the scripted errors in order, then returns a sandbox."""

    def __init__(self, errors: list[Exception], *, landed: bool = False) -> None:
        self.errors = list(errors)
        self.creates = 0
        self.made = SimpleNamespace(id="sb-ok", delete=AsyncMock())
        self.get = AsyncMock()
        self.queries: list = []
        self.landed = landed
        self.remote: list = []
        self.deleted: list[str] = []

    async def create(self, params, timeout):
        self.creates += 1
        if self.errors:
            if self.landed:
                box = SimpleNamespace(
                    id=f"failed-{self.creates}", labels=dict(params.labels)
                )

                async def delete(timeout=60, wait=False):
                    assert wait is True and 0 < timeout <= 60
                    self.deleted.append(box.id)
                    self.remote.remove(box)

                box.delete = delete
                self.remote.append(box)
            raise self.errors.pop(0)
        return self.made

    async def list(self, query=None):
        self.queries.append(query)
        wanted = dict(getattr(query, "labels", None) or {})
        for box in list(self.remote):
            if all(box.labels.get(k) == v for k, v in wanted.items()):
                yield box


def _sandbox(client) -> DaytonaSandbox:
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox._sandbox = None
    sandbox._create_attempts = 0
    sandbox.rollout_paths = None
    sandbox._client_manager = SimpleNamespace(get_client=AsyncMock(return_value=client))
    sandbox.task_env_config = SimpleNamespace(
        build_timeout_sec=60, cpus=1, memory_mb=1024, storage_mb=1024, docker_image=None
    )
    sandbox.logger = logging.getLogger("test.daytona.start_path")
    return sandbox


@pytest.fixture
def waits(monkeypatch) -> list[tuple[str, int, float]]:
    """Record (kind, attempt, wait it would take) for every wait the start
    path asks for, and wait 0 s."""
    asked: list[tuple[str, int, float]] = []
    real = daytona_start.wait_sec

    def fake(exc, attempt):
        asked.append((daytona_start.error_kind(exc), attempt, real(exc, attempt)))
        return 0.0

    monkeypatch.setattr(daytona_start, "wait_sec", fake)
    return asked


@pytest.mark.asyncio
async def test_429_storm_is_waited_out_with_retry_after(waits):
    client = _ScriptedClient([_rate_limited("40")] * 4)
    sandbox = _sandbox(client)

    await sandbox._create_sandbox(params=SimpleNamespace(labels={}))

    assert sandbox._sandbox is client.made
    assert client.creates == 5  # the old policy lost the trial after 3
    assert [w[2] for w in waits] == [40.0] * 4


@pytest.mark.asyncio
async def test_5xx_is_retried_with_jitter_then_succeeds(waits):
    """Guards PR #13: preserve retry policy after confirmed 5xx cleanup."""
    client = _ScriptedClient([_unavailable()] * 5, landed=True)
    sandbox = _sandbox(client)

    await sandbox._create_sandbox(params=SimpleNamespace(labels={}))

    assert sandbox._sandbox is client.made
    assert client.creates == 6  # the old policy did not retry a 5xx at all
    assert [kind for kind, _, _ in waits] == ["transient"] * 5
    # Each retry first looked for the attempt's own sandbox by its label.
    assert len(client.queries) == 5
    assert client.deleted == [f"failed-{i}" for i in range(1, 6)]
    assert client.remote == []


@pytest.mark.asyncio
async def test_5xx_budget_is_bounded(waits):
    """Guards PR #13: confirmed cleanup does not reset the retry budget."""
    client = _ScriptedClient([_unavailable()] * 50, landed=True)
    sandbox = _sandbox(client)

    with pytest.raises(Exception, match="Service Unavailable"):
        await sandbox._create_sandbox(params=SimpleNamespace(labels={}))
    assert client.creates == daytona_start.CREATE_TRANSIENT_MAX_ATTEMPTS
    assert len(client.deleted) == client.creates and client.remote == []


@pytest.mark.asyncio
async def test_capacity_budget_is_bounded_by_time(monkeypatch):
    """A Retry-After past the budget stops at once, with no early retry."""
    client = _ScriptedClient([_rate_limited("3600")] * 5)
    sandbox = _sandbox(client)

    with pytest.raises(Exception, match="Too Many Requests"):
        await sandbox._create_sandbox(params=SimpleNamespace(labels={}))
    assert client.creates == 1


@pytest.mark.asyncio
async def test_rejected_key_fails_fast_and_is_not_retried_by_the_trial(waits):
    client = _ScriptedClient([_rejected()] * 5)
    sandbox = _sandbox(client)

    with pytest.raises(SandboxStartupError) as raised:
        await sandbox._create_sandbox(params=SimpleNamespace(labels={}))

    assert client.creates == 1
    assert waits == []
    message = str(raised.value)
    assert "Daytona rejected DAYTONA_API_KEY (HTTP 401)" in message
    assert "dashboard/keys" in message
    # The trial-level retry (retry_on_infra) does not repeat it either.
    assert is_unrecoverable_startup_error(message)
    assert not is_unrecoverable_startup_error(
        "Sandbox creation failed after 6 attempts: Service Unavailable"
    )


@pytest.mark.asyncio
async def test_quota_refusal_sent_as_403_is_waited_out(waits):
    quota = _sdk_error(
        "DaytonaAuthorizationError",
        "Failed to create sandbox: Total CPU limit exceeded. Maximum allowed: 100",
        status=403,
    )
    client = _ScriptedClient([quota] * 3)
    sandbox = _sandbox(client)

    await sandbox._create_sandbox(params=SimpleNamespace(labels={}))
    assert client.creates == 4
    assert [kind for kind, _, _ in waits] == ["capacity"] * 3


# ── The toolbox start-up probe ──────────────────────────────────────────


def _probe_sandbox(results: list) -> tuple[DaytonaSandbox, AsyncMock]:
    async def exec_(*_args, **_kwargs):
        value = results.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    exec_mock = AsyncMock(side_effect=exec_)
    sandbox = _sandbox(_ScriptedClient([]))
    sandbox._sandbox = SimpleNamespace(id="sb", process=SimpleNamespace(exec=exec_mock))
    return sandbox, exec_mock


def _probe_answer() -> SimpleNamespace:
    """The probe's answer in its framing: an ``E`` record (byte length, a
    colon, ``name=value``) for each start-up name that is set, then ``Z``;
    only BASH_ENV is set."""
    record = "BASH_ENV=/etc/bashrc"
    return SimpleNamespace(exit_code=0, result=f"E{len(record.encode())}:{record}Z")


@pytest.mark.asyncio
async def test_toolbox_probe_asks_again_after_a_transient_error(waits):
    sandbox, exec_mock = _probe_sandbox([_unavailable(), _probe_answer()])

    values = await sandbox._toolbox_startup_env()

    assert values["BASH_ENV"] == "/etc/bashrc"
    assert exec_mock.await_count == 2


@pytest.mark.asyncio
async def test_toolbox_probe_does_not_retry_a_rejected_key(waits):
    sandbox, exec_mock = _probe_sandbox([_rejected(), _probe_answer()])

    with pytest.raises(Exception, match="Unauthorized"):
        await sandbox._toolbox_startup_env()
    assert exec_mock.await_count == 1


@pytest.mark.asyncio
async def test_toolbox_probe_gives_up_after_three_tries(waits):
    sandbox, exec_mock = _probe_sandbox([_unavailable()] * 5)

    with pytest.raises(Exception, match="Service Unavailable"):
        await sandbox._toolbox_startup_env()
    assert exec_mock.await_count == 3


# ── The run preflight and bench doctor ──────────────────────────────────


KEY = "dtn_SECRETstartpathkey_0123456789"


def _probes(tmp_path: Path, errors: list, slept: list) -> DoctorProbes:
    calls = iter(errors)

    def daytona_check(environ: Mapping[str, str], timeout: float) -> None:
        error = next(calls, None)
        if error is not None:
            raise error

    return DoctorProbes(
        environ={"DAYTONA_API_KEY": KEY},
        origins={"DAYTONA_API_KEY": "env"},
        home=tmp_path,
        dist_version=lambda dist: "0.207.0",
        daytona_check=daytona_check,
        sleep=slept.append,
    )


@pytest.mark.parametrize(
    "error",
    [TimeoutError(), _rate_limited(), _unavailable(), ConnectionResetError("reset")],
)
def test_run_preflight_does_not_refuse_an_unanswered_key_check(tmp_path, error):
    slept: list = []
    probes = _probes(tmp_path, [error], slept)

    row = check_daytona(probes, required=True, offline=False, for_run=True)

    assert row.status == "warn"
    assert "sandbox creation retries" in row.summary
    assert slept == []  # one try; creation owns the retries
    assert KEY not in row.summary


def test_run_preflight_refuses_a_rejected_key(tmp_path):
    probes = _probes(tmp_path, [_rejected()], [])

    row = check_daytona(probes, required=True, offline=False, for_run=True)

    assert row.status == "fail"
    assert "rejected DAYTONA_API_KEY" in row.summary
    assert "dashboard/keys" in (row.fix or "")


def test_doctor_retries_and_names_an_unanswered_api_not_the_key(tmp_path):
    slept: list = []
    probes = _probes(tmp_path, [TimeoutError()] * 3, slept)

    row = check_daytona(probes, required=True, offline=False)

    assert row.status == "fail"
    assert "did not answer after 3 tries" in row.summary
    assert "the key was not checked" in row.summary
    assert "dashboard/keys" not in (row.fix or "")
    assert len(slept) == 2


def test_doctor_passes_when_a_retry_answers(tmp_path):
    slept: list = []
    probes = _probes(tmp_path, [_unavailable()], slept)

    row = check_daytona(probes, required=True, offline=False)

    assert row.status == "pass"
    assert len(slept) == 1


def test_key_check_timeout_does_not_count_the_sdk_import(monkeypatch):
    """The 33 lost trials: a slow import (CPU contention) used the timeout."""
    listed = threading.Event()

    class _Client:
        def list(self, query=None):
            listed.set()
            return iter(())

    def slow_client(api_key=None):
        time.sleep(0.5)  # the import of the SDK on a busy machine
        return _Client()

    monkeypatch.setattr(daytona_module, "build_sync_client", slow_client)

    doctor_mod._daytona_list_one({"DAYTONA_API_KEY": KEY}, 0.2)

    assert listed.is_set()


# ── The eval-start auto-reap ────────────────────────────────────────────
# Which sandboxes it deletes is tested in test_daytona_start_r2.py; here,
# what it asks Daytona for.


class _ReapClient:
    def __init__(self) -> None:
        self.queries: list = []

    def list(self, query=None):
        self.queries.append(query)
        return iter(())


def test_auto_reap_lists_only_our_sandboxes_old_enough_to_have_lapsed(monkeypatch):
    monkeypatch.delenv("BENCHFLOW_DAYTONA_OWNER", raising=False)
    client = _ReapClient()

    reap_abandoned_sandboxes(client)

    [query] = client.queries
    assert query.labels == {"benchflow.managed": "1"}
    # A lease is at least an hour from creation: nothing younger can lapse.
    cutoff = datetime.now(UTC) - timedelta(minutes=60)
    assert abs((query.created_at_before - cutoff).total_seconds()) < 60


def test_auto_reap_scopes_to_the_owner(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "team-a")
    client = _ReapClient()

    reap_abandoned_sandboxes(client)

    [query] = client.queries
    assert query.labels == {"benchflow.managed": "1:team-a"}


def test_cleanup_command_keeps_the_whole_list():
    client = _ReapClient()

    reap_stale_sandboxes(client)

    assert client.queries == [None]


def test_eval_start_uses_the_lease_reap_and_leaves_snapshots(tmp_path, monkeypatch):
    seen: list = []
    snapshots: list = []
    monkeypatch.setattr(
        daytona_module,
        "reap_abandoned_sandboxes",
        lambda **kw: seen.append(kw) or {"deleted": 0, "failed": 0, "unleased": 0},
    )
    monkeypatch.setattr(
        daytona_module, "reap_stale_snapshots", lambda **kw: snapshots.append(kw)
    )

    class _Inline:
        def __init__(self, target, **_kw):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr("benchflow.evaluation.threading.Thread", _Inline)
    from benchflow.evaluation import Evaluation

    evaluation = Evaluation.__new__(Evaluation)
    evaluation._config = SimpleNamespace(environment="daytona")
    monkeypatch.delenv("BENCHFLOW_DAYTONA_AUTO_REAP", raising=False)
    evaluation._maybe_start_daytona_reap()

    assert seen == [{}]
    # Snapshots carry no lease, so nothing proves one abandoned.
    assert snapshots == []
