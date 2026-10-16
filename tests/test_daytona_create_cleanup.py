"""Daytona sandbox creation: gateway errors retry, and failed attempts leave no sandbox behind.

Found in a FrontierPhysics oracle sweep: a 502 while the SDK waited for a new
sandbox to start failed the attempt without a retry, and the sandbox it had
already created kept running with nothing able to delete it.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
from tenacity import wait_none

import benchflow.sandbox.daytona as daytona_mod
from benchflow.sandbox.daytona import DaytonaSandbox
from benchflow.sandbox.daytona_pty import _is_daytona_transient_retry_error

LABEL = daytona_mod._CREATE_ATTEMPT_LABEL


def _daytona_error(name: str, **attrs: object) -> Exception:
    cls = type(name, (Exception,), {"__module__": "daytona.common.errors"})
    error = cls(f"{name} from the Daytona SDK")
    for key, value in attrs.items():
        setattr(error, key, value)
    return error


@pytest.mark.parametrize(
    "error",
    [
        _daytona_error("DaytonaBadGatewayError", status_code=502),
        _daytona_error("DaytonaServiceUnavailableError", status_code=503),
        _daytona_error("DaytonaConnectionTimeoutError"),
        # SDKs before the typed gateway classes raise the base class with the status.
        _daytona_error("DaytonaError", status_code=502),
        _daytona_error("DaytonaError", status_code=503),
        _daytona_error("DaytonaError", status_code=504),
    ],
)
def test_gateway_errors_are_transient(error: Exception) -> None:
    assert _is_daytona_transient_retry_error(error)


@pytest.mark.parametrize(
    "error",
    [
        _daytona_error("DaytonaError", status_code=400),
        _daytona_error("DaytonaError", status_code=500),
        _daytona_error("DaytonaError"),
        # A typed class goes by its name, whatever status it carries.
        _daytona_error("DaytonaAuthenticationError", status_code=502),
        RuntimeError("502 in a message from elsewhere"),
        type("Other", (Exception,), {"status_code": 502})("not from the SDK"),
    ],
)
def test_other_errors_stay_permanent(error: Exception) -> None:
    assert not _is_daytona_transient_retry_error(error)


@pytest.mark.parametrize(
    ("status", "transient"),
    [(502, True), (503, True), (504, True), (401, False), (500, False)],
)
def test_installed_sdk_status_mapping(status: int, transient: bool) -> None:
    """The installed SDK's own error for each status, not a forged class."""
    errors = pytest.importorskip("daytona.common.errors")
    error = errors.create_daytona_error("gateway answer", status_code=status)
    assert _is_daytona_transient_retry_error(error) is transient


class _FakeSandbox:
    def __init__(self, sandbox_id: str, labels: dict[str, str], registry: dict) -> None:
        self.id = sandbox_id
        self.labels = dict(labels)
        self._registry = registry

    async def delete(self) -> None:
        self._registry.pop(self.id, None)


class _FakeDaytona:
    """Daytona as seen through the SDK: create() may make the sandbox and still raise.

    ``plan`` holds one entry per create() call: "ok", an exception raised after
    the sandbox exists, ("late", exc) to raise first and let the sandbox appear
    after the next list(), or "hang" to make it and never return.
    """

    def __init__(self, plan: list, registry: dict) -> None:
        self.plan = list(plan)
        self.registry = registry
        self.creates: list[dict[str, str]] = []
        self.list_queries: list[dict[str, str]] = []
        self.list_failures: list[Exception] = []
        self.honour_filter = True
        self._late: list[_FakeSandbox] = []

    async def create(self, params, timeout):
        del timeout
        step = self.plan.pop(0)
        self.creates.append(dict(params.labels))
        sandbox = _FakeSandbox(f"sb-{len(self.creates)}", params.labels, self.registry)
        if isinstance(step, tuple) and step[0] == "late":
            self._late.append(sandbox)
            raise step[1]
        self.registry[sandbox.id] = sandbox
        if step == "hang":
            await asyncio.sleep(3600)
        if isinstance(step, Exception):
            raise step
        return sandbox

    async def list(self, query):
        self.list_queries.append(dict(query.labels))
        if self.list_failures:
            raise self.list_failures.pop(0)
        for sandbox in list(self.registry.values()):
            # The server honours the label filter (unless a test says otherwise).
            if not self.honour_filter or all(
                sandbox.labels.get(k) == v for k, v in query.labels.items()
            ):
                yield sandbox
        for sandbox in self._late:
            self.registry[sandbox.id] = sandbox
        self._late.clear()


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr(
        daytona_mod, "ListSandboxesQuery", lambda labels: SimpleNamespace(labels=labels)
    )
    for fn in (
        DaytonaSandbox._create_sandbox,
        DaytonaSandbox._list_create_leftovers,
        DaytonaSandbox._delete_leftover,
    ):
        monkeypatch.setattr(fn.retry, "wait", wait_none())


def _sandbox_under_test(daytona: _FakeDaytona) -> DaytonaSandbox:
    env = DaytonaSandbox.__new__(DaytonaSandbox)

    class _Manager:
        async def get_client(self):
            return daytona

    env._client_manager = _Manager()
    env._sandbox = None
    env.task_env_config = SimpleNamespace(build_timeout_sec=60)
    env.logger = logging.getLogger("test.daytona.create")
    env._on_sandbox_created = lambda: None
    return env


def _params() -> SimpleNamespace:
    return SimpleNamespace(labels={"benchflow.managed": "1"})


@pytest.mark.asyncio
async def test_failed_creation_deletes_only_its_own_leftover(no_backoff) -> None:
    registry: dict = {}
    registry["other"] = _FakeSandbox("other", {LABEL: "another-run"}, registry)
    daytona = _FakeDaytona(
        [RuntimeError("Failure during waiting for sandbox to start")], registry
    )
    env = _sandbox_under_test(daytona)

    with pytest.raises(RuntimeError, match="Failure during waiting"):
        await env._create_sandbox(_params())

    assert list(registry) == ["other"]
    assert daytona.list_queries == [{LABEL: daytona.creates[0][LABEL]}]


@pytest.mark.asyncio
async def test_a_server_ignoring_the_filter_still_spares_other_sandboxes(
    no_backoff,
) -> None:
    registry: dict = {}
    registry["unlabelled"] = _FakeSandbox(
        "unlabelled", {"benchflow.managed": "1"}, registry
    )
    registry["other"] = _FakeSandbox("other", {LABEL: "another-run"}, registry)
    daytona = _FakeDaytona(
        [RuntimeError("Failure during waiting for sandbox to start")], registry
    )
    daytona.honour_filter = False
    env = _sandbox_under_test(daytona)

    with pytest.raises(RuntimeError, match="Failure during waiting"):
        await env._create_sandbox(_params())

    assert sorted(registry) == ["other", "unlabelled"]


@pytest.mark.asyncio
async def test_gateway_error_is_retried_after_the_leftover_is_deleted(
    no_backoff,
) -> None:
    registry: dict = {}
    daytona = _FakeDaytona(
        [_daytona_error("DaytonaError", status_code=502), "ok"], registry
    )
    env = _sandbox_under_test(daytona)

    await env._create_sandbox(_params())

    assert len(daytona.creates) == 2
    assert daytona.creates[0][LABEL] == daytona.creates[1][LABEL]
    assert list(registry) == ["sb-2"]
    assert env._sandbox is registry["sb-2"]


@pytest.mark.asyncio
async def test_the_lookup_retries_gateway_errors_too(no_backoff) -> None:
    registry: dict = {}
    daytona = _FakeDaytona([RuntimeError("start-wait failed")], registry)
    daytona.list_failures = [_daytona_error("DaytonaBadGatewayError", status_code=502)]
    env = _sandbox_under_test(daytona)

    with pytest.raises(RuntimeError, match="start-wait failed"):
        await env._create_sandbox(_params())

    assert registry == {}
    assert len(daytona.list_queries) == 2


@pytest.mark.asyncio
async def test_a_sandbox_that_appears_late_is_deleted_after_the_retry(
    no_backoff,
) -> None:
    registry: dict = {}
    late = ("late", _daytona_error("DaytonaError", status_code=504))
    daytona = _FakeDaytona([late, "ok"], registry)
    env = _sandbox_under_test(daytona)

    await env._create_sandbox(_params())

    assert list(registry) == ["sb-2"]
    assert env._sandbox is registry["sb-2"]


@pytest.mark.asyncio
async def test_a_timed_out_creation_leaves_nothing(no_backoff, monkeypatch) -> None:
    monkeypatch.setattr(daytona_mod, "_STARTUP_HARD_TIMEOUT_BUFFER_SEC", 0.05)
    registry: dict = {}
    daytona = _FakeDaytona(["hang", "hang", "hang"], registry)
    env = _sandbox_under_test(daytona)
    env.task_env_config = SimpleNamespace(build_timeout_sec=0)

    with pytest.raises(TimeoutError):
        await env._create_sandbox(_params())

    assert registry == {}


@pytest.mark.asyncio
async def test_stop_sweeps_what_a_failed_lookup_missed(no_backoff) -> None:
    registry: dict = {}
    daytona = _FakeDaytona([RuntimeError("start-wait failed")], registry)
    daytona.list_failures = [RuntimeError("lookup refused")]
    env = _sandbox_under_test(daytona)
    stopped = []

    async def strategy_stop(delete):
        stopped.append(delete)

    env._strategy = SimpleNamespace(stop=strategy_stop)

    with pytest.raises(RuntimeError, match="start-wait failed"):
        await env._create_sandbox(_params())
    assert list(registry) == ["sb-1"]

    await env.stop(delete=True)

    assert registry == {}
    assert stopped == [True]


@pytest.mark.asyncio
async def test_successful_creation_looks_nothing_up(no_backoff) -> None:
    registry: dict = {}
    daytona = _FakeDaytona(["ok"], registry)
    env = _sandbox_under_test(daytona)

    await env._create_sandbox(_params())

    assert env._sandbox is registry["sb-1"]
    assert env._sandbox.labels["benchflow.managed"] == "1"
    assert env._sandbox.labels[LABEL]
    assert daytona.list_queries == []
