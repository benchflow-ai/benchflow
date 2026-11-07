"""A Daytona sandbox whose image is still building is never deleted.

Deleting a sandbox while Daytona builds its image cancels the build, and
Daytona keeps that build as failed under the image's build key: every later
create of the image fails within a second with "BUILD_FAILED, error reason:
context canceled", and Daytona has no way to clear it. 0.9 could delete one
in the cleanup of a create that timed out mid-build (in process and through
the ledger), in ``bench sandbox cleanup`` (``--all``, ``--id``, by age, and
the recorded leftovers) and in teardown.

Guards the fix for the gap a benchmark's own infrastructure works around as
a patch. The SDK's own ``AsyncSandbox``/``Sandbox`` and their states are
used where the state's spelling matters; no socket is opened.
"""

from __future__ import annotations

import logging
import socket
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import daytona as daytona_module
from benchflow.sandbox import daytona_delete, daytona_start
from benchflow.sandbox import daytona_ledger as ledger_module
from benchflow.sandbox.daytona import DaytonaSandbox
from benchflow.sandbox.daytona_ledger import Ledger
from benchflow.sandbox.daytona_reaper import reap_stale_sandboxes
from benchflow.sandbox.daytona_strategies import _DaytonaDirect
from benchflow.sandbox.protocol import SandboxStartupError

_ATTEMPT = "benchflow.create-attempt"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("socket access forbidden in this test")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setenv("BENCHFLOW_DAYTONA_LEFTOVERS", str(tmp_path / "leftovers"))
    monkeypatch.delenv("BENCHFLOW_DAYTONA_OWNER", raising=False)
    monkeypatch.setattr(daytona_start, "wait_sec", lambda *_: 0.0)
    ledger = Ledger()
    monkeypatch.setattr(daytona_module, "LEDGER", ledger)
    monkeypatch.setattr(ledger_module, "LEDGER", ledger)
    monkeypatch.setattr(ledger, "ensure_sweeper", lambda: None)
    yield ledger
    ledger.entries.clear()


class _Box(SimpleNamespace):
    """A sandbox as a lookup or a list returns it; ``delete`` records."""

    def __init__(self, sandbox_id: str, state: str, labels: dict | None = None):
        super().__init__(
            id=sandbox_id,
            state=state,
            labels=dict(labels or {}),
            created_at=(datetime.now(UTC) - timedelta(hours=30)).isoformat(),
            delete=AsyncMock(),
            refresh_activity=AsyncMock(),
            set_labels=AsyncMock(),
        )


def test_build_states_are_the_ones_a_delete_would_cancel():
    """Every state of the installed SDK is classified; BUILD_FAILED is a
    finished build, so its sandbox can go."""
    pytest.importorskip("daytona")
    from daytona import SandboxState

    building = {s for s in SandboxState if daytona_delete.still_building(_Box("x", s))}
    assert building == {
        SandboxState.PENDING_BUILD,
        SandboxState.BUILDING_SNAPSHOT,
        SandboxState.PULLING_SNAPSHOT,
    }
    for spelled in ("SandboxState.PENDING_BUILD", "pending_build", "BUILDING_SNAPSHOT"):
        assert daytona_delete.still_building(_Box("x", spelled))
    assert daytona_delete.still_building(_Box("x", "building_image")), "a new one"
    for done in ("SandboxState.BUILD_FAILED", "started", "error", None):
        assert not daytona_delete.still_building(_Box("x", done))


@pytest.mark.asyncio
async def test_the_sdks_own_sandbox_in_a_build_state_is_not_deleted():
    """The SDK's AsyncSandbox and Sandbox carry the state as an enum; neither
    delete request is sent."""
    pytest.importorskip("daytona")
    from daytona import AsyncSandbox, Sandbox

    sent: list[str] = []
    data = dict(id="sb-sdk", organization_id="o", state="building_snapshot")

    async def adelete(*_a, **_kw):
        sent.append("async")

    box = AsyncSandbox.model_construct(**data)
    box._sandbox_api = SimpleNamespace(delete_sandbox=adelete)
    with pytest.raises(daytona_delete.SandboxStillBuilding, match="sb-sdk"):
        await daytona_delete.delete_sandbox(box)

    sync_box = Sandbox.model_construct(**data)
    client = SimpleNamespace(delete=lambda sb, **_kw: sent.append("sync"))
    with pytest.raises(daytona_delete.SandboxStillBuilding, match="BUILDING_SNAPSHOT"):
        daytona_delete.delete_sandbox_sync(client, sync_box)
    assert sent == []


def _env(box: _Box | None) -> DaytonaSandbox:
    env = DaytonaSandbox.__new__(DaytonaSandbox)
    env._sandbox = box
    env._create_attempts = 0
    env._create_outcome_unknown = False
    env._client_manager = SimpleNamespace(get_client=AsyncMock())
    env.task_env_config = SimpleNamespace(build_timeout_sec=60)
    env.logger = logging.getLogger("test.daytona.keep_building")
    env.rollout_paths = None
    return env


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "deleted"),
    [("BUILDING_SNAPSHOT", 0), ("PENDING_BUILD", 0), ("STARTED", 1)],
)
async def test_teardown_leaves_a_sandbox_that_still_builds(
    state, deleted, isolated, caplog
):
    """stop() warns with the id and keeps the sandbox in the ledger, which
    deletes it once it has started; a STARTED one is deleted as before."""
    box = _Box("sb-1", state)
    env = _env(box)
    with caplog.at_level(logging.WARNING):
        await _DaytonaDirect(env).stop(delete=True)
    assert box.delete.await_count == deleted
    assert env._sandbox is None
    if not deleted:
        assert "sandbox:sb-1" in isolated.entries
        assert any(
            r.levelno == logging.WARNING and "sb-1" in r.getMessage()
            for r in caplog.records
        )
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


# The SDK's own timeout (no HTTP status): a create that did not finish.
_DaytonaTimeoutError = type(
    "DaytonaTimeoutError", (Exception,), {"__module__": "daytona.common.errors"}
)


class _Client:
    """``create`` lands a sandbox that is still building, then the create
    times out; ``list`` finds sandboxes by label."""

    def __init__(self) -> None:
        self.remote: list[_Box] = []
        self.creates = 0
        self.get = AsyncMock()

    async def create(self, params, timeout):
        self.creates += 1
        self.remote.append(
            _Box(f"sb-{self.creates}", "BUILDING_SNAPSHOT", params.labels)
        )
        raise _DaytonaTimeoutError(
            "Sandbox failed to become ready within the timeout period"
        )

    async def list(self, query=None, request_timeout=None):
        wanted = dict(getattr(query, "labels", None) or {})
        for sandbox in list(self.remote):
            if all(sandbox.labels.get(k) == v for k, v in wanted.items()):
                yield sandbox


@pytest.mark.asyncio
async def test_a_create_that_timed_out_mid_build_is_deleted_only_once_started(
    monkeypatch, isolated
):
    """0.9 looks the timed-out attempt's sandbox up by its label and deletes
    it before any retry, while Daytona may still be building it. It stays
    now: the create fails as an unreconciled one (no retry while its outcome
    is unknown), and the ledger deletes the sandbox once it has started."""
    monkeypatch.setattr(daytona_module, "DaytonaTimeoutError", _DaytonaTimeoutError)
    monkeypatch.setattr(daytona_module, "_CREATE_WATCH_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(daytona_module, "_CREATE_SETTLE_SEC", 0.05)
    monkeypatch.setattr(ledger_module, "SETTLE_SEC", 0.05)
    monkeypatch.setattr(ledger_module, "SWEEP_INTERVAL_SEC", 0.01)
    client = _Client()
    env = _env(None)
    env._client_manager = SimpleNamespace(get_client=AsyncMock(return_value=client))

    with pytest.raises(SandboxStartupError):
        await env._create_sandbox(params=SimpleNamespace(labels={}))

    (building,) = client.remote
    assert building.delete.await_count == 0
    assert client.creates == 1
    assert env._create_outcome_unknown

    building.state = "STARTED"
    await env._drain_create_attempts()
    assert building.delete.await_count == 1
    assert not env._create_outcome_unknown
    assert not isolated.entries


@pytest.mark.asyncio
async def test_a_failed_create_whose_lookup_finds_a_building_sandbox_keeps_it(
    isolated,
):
    """The labelled cleanup after a create failed without naming its sandbox
    (a 5xx) does not delete one that is building, and says it is not done."""
    building = _Box("sb-9", "PENDING_BUILD", {_ATTEMPT: "a1"})
    client = _Client()
    client.remote.append(building)
    env = _env(None)

    assert not await env._delete_sandboxes_labelled(
        client, {_ATTEMPT: "a1"}, require_match=True
    )
    assert building.delete.await_count == 0


@pytest.mark.asyncio
async def test_the_ledger_reports_a_building_sandbox_once_and_deletes_it_later(
    isolated, caplog
):
    """Each sweep looks again; the warning is written once per sandbox."""
    building = _Box("sb-7", "BUILDING_SNAPSHOT", {_ATTEMPT: "a7"})
    client = _Client()
    client.remote.append(building)
    entry = isolated.open_attempt(client, {_ATTEMPT: "a7"}, "test", managed="1")

    with caplog.at_level(logging.WARNING):
        assert not await isolated._sweep_one(entry)
        assert not await isolated._sweep_one(entry)
    assert building.delete.await_count == 0
    assert sum("sb-7" in r.getMessage() for r in caplog.records) == 1

    building.state = "STARTED"
    assert await isolated._sweep_one(entry)
    assert building.delete.await_count == 1


class _Server:
    """The sync client of ``bench sandbox cleanup`` and the exit sweep."""

    def __init__(self, sandboxes: list[_Box]) -> None:
        self.sandboxes = sandboxes
        self.deleted: list[str] = []

    def list(self, query=None, request_timeout=None):
        wanted = dict(getattr(query, "labels", None) or {})
        return [
            sb
            for sb in self.sandboxes
            if all(sb.labels.get(k) == v for k, v in wanted.items())
        ]

    def get(self, sandbox_id, request_timeout=None):
        return next(sb for sb in self.sandboxes if sb.id == sandbox_id)

    def delete(self, sb, timeout=60, wait=False):
        self.deleted.append(sb.id)


_OWNED = {"benchflow.managed": "1"}


@pytest.mark.parametrize(
    "mode", ["all", "id", "age", "dry-run-all", "dry-run-id"], ids=lambda m: m
)
def test_bench_sandbox_cleanup_never_deletes_a_sandbox_that_still_builds(mode):
    """``--all``, ``--id``, the age rule and their dry runs skip it with the
    reason ``building``; the stopped one beside it goes as before."""
    building = _Box("building", "PENDING_BUILD", _OWNED)
    stopped = _Box("stopped", "STOPPED", _OWNED)
    server = _Server([building, stopped])
    skips: list[tuple[str, str]] = []
    kwargs = {
        "dry_run": mode.startswith("dry-run"),
        "on_skip": lambda sb, why: skips.append((sb.id, why)),
        "ignore_age": mode.endswith("all"),
        "names": frozenset({"building", "stopped"} if mode.endswith("id") else ()),
        "max_age_minutes": 0,
        "failed_max_age_minutes": 0,
        "min_idle_minutes": 0,
    }

    counts = reap_stale_sandboxes(server, **kwargs)

    assert ("building", "building") in skips
    assert counts["deleted"] == 1
    assert server.deleted == ([] if kwargs["dry_run"] else ["stopped"])


def test_recorded_leftovers_keep_a_sandbox_that_still_builds(isolated):
    """The exit sweep and ``bench sandbox cleanup`` keep the attempt open
    while its sandbox builds, count it apart, and delete it once started."""
    building = _Box("sb-5", "BUILDING_SNAPSHOT", {_ATTEMPT: "a5"})
    server = _Server([building])
    entry = isolated.open_attempt(None, {_ATTEMPT: "a5"}, "test", managed="1")

    isolated._sweep_one_sync(server, entry)
    assert entry.key in isolated.entries and server.deleted == []

    counts = ledger_module.cleanup_leftovers(server, "1")
    assert counts["building"] == 1 and counts["deleted"] == 0
    assert server.deleted == []
    assert len(ledger_module.read_leftovers()) == 1

    building.state = "STARTED"
    counts = ledger_module.cleanup_leftovers(server, "1")
    assert server.deleted == ["sb-5"] and counts["deleted"] == 1
    assert ledger_module.read_leftovers() == []


def test_a_stopped_or_failed_build_is_still_deleted():
    """The control: BUILD_FAILED and STOPPED sandboxes are not held back."""
    for state in ("BUILD_FAILED", "STOPPED", "ERROR"):
        server = _Server([_Box("x", state)])
        daytona_delete.delete_sandbox_sync(server, server.sandboxes[0])
        assert server.deleted == ["x"]
