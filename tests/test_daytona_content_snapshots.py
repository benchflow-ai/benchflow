"""Daytona snapshots named by content (opt-in), against a fake Daytona API (PY8 U13).

The real direct strategy's ``start()`` runs against a fake client whose
snapshot service keeps snapshots by name the way Daytona does (one per name:
a second create of a name gets a conflict). The SDK's own ``Image``,
``Resources``, ``CreateSnapshotParams`` and sandbox params build the
requests.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("daytona")

from benchflow.sandbox import daytona as daytona_mod
from tests.test_daytona_install_layers import (
    AGENT,
    AGENT_ENV,
    MODEL,
    OTHER_IMAGE,
    TASK_IMAGE,
    _configure,
    _sandbox,
)

daytona_mod._load_daytona_sdk()
import daytona as sdk  # noqa: E402


def _optional(module: str) -> Any:
    """``module``, or None on a BenchFlow without it (the fails-before base),
    so these tests fail there on what BenchFlow does, not on an import."""
    try:
        return importlib.import_module(module)
    except ImportError:
        return None


content = _optional("benchflow.sandbox.daytona_content_snapshots")
SNAPSHOTS_ENV = "BENCHFLOW_DAYTONA_SNAPSHOTS"

OWNER = "py8-lock"
PREFIX = f"bf-img2-{hashlib.sha256(OWNER.encode()).hexdigest()[:32]}-"
PINNED_BASE = "python@sha256:" + "3" * 64


class Server:
    """Daytona's side: snapshots by name, shared by every client."""

    def __init__(self) -> None:
        self.snaps: dict[str, SimpleNamespace] = {}
        self.create_calls: list[Any] = []
        self.successes = 0
        self.deleted: list[str] = []
        self.activated: list[str] = []
        self.build_fail = False
        self.create_gate: asyncio.Event | None = None
        self.get_script: dict[str, list[str]] = {}
        self.get_barrier: list[asyncio.Event] | None = None
        self._ids = 0

    def put(self, name: str, state: str) -> SimpleNamespace:
        self._ids += 1
        snap = SimpleNamespace(
            id=f"snap-{self._ids}",
            name=name,
            state=daytona_mod.SnapshotState(state),
            error_reason=None,
        )
        self.snaps[name] = snap
        return snap


class Snapshots:
    def __init__(self, server: Server) -> None:
        self.server = server

    async def get(self, name: str) -> Any:
        server = self.server
        if server.get_barrier is not None:
            # Both instances read the name (missing) before either creates.
            missing = name not in server.snaps
            mine = asyncio.Event()
            server.get_barrier.append(mine)
            if len(server.get_barrier) >= 2:
                for event in server.get_barrier:
                    event.set()
                server.get_barrier = None
            await mine.wait()
            if missing:
                raise sdk.DaytonaNotFoundError(f"snapshot {name} not found")
        script = server.get_script.get(name)
        if script:
            state = script.pop(0)
            if state == "gone":
                server.snaps.pop(name, None)
            elif state.startswith("replace:"):
                server.put(name, state.removeprefix("replace:"))
            else:
                server.snaps[name].state = daytona_mod.SnapshotState(state)
        if name not in server.snaps:
            raise sdk.DaytonaNotFoundError(f"snapshot {name} not found")
        return server.snaps[name]

    async def create(self, params: Any, timeout: Any = None) -> Any:
        server = self.server
        server.create_calls.append(params)
        if params.name in server.snaps:
            raise sdk.DaytonaConflictError(f"snapshot {params.name} already exists")
        snap = server.put(params.name, "pending")
        if server.create_gate is not None:
            await server.create_gate.wait()
        if server.build_fail:
            snap.state = daytona_mod.SnapshotState("build_failed")
            raise sdk.DaytonaError(
                f"Failed to create snapshot {params.name}, reason: x"
            )
        snap.state = daytona_mod.SnapshotState("active")
        server.successes += 1
        return snap

    async def delete(self, snap: Any) -> None:
        self.server.deleted.append(snap.id)
        if self.server.snaps.get(snap.name) is snap:
            del self.server.snaps[snap.name]

    async def activate(self, snap: Any) -> Any:
        self.server.activated.append(snap.id)
        snap.state = daytona_mod.SnapshotState("active")
        return snap


class Client:
    """One process's client: creates sandboxes, from an image or a snapshot."""

    def __init__(self, server: Server) -> None:
        self.server = server
        self.snapshot = Snapshots(server)
        self.requests: list[Any] = []

    async def create(self, params, timeout=None):
        self.requests.append(params)
        if isinstance(params, sdk.CreateSandboxFromSnapshotParams):
            assert params.snapshot in self.server.snaps, "no such snapshot"
        return SimpleNamespace(
            id=f"sb-{len(self.requests)}",
            created_at="2026-10-16T00:00:00Z",
            build_info=None,
            runner_id="runner-a",
            delete=AsyncMock(),
        )

    async def get(self, sandbox_id):
        return SimpleNamespace(id=sandbox_id, delete=AsyncMock())


@pytest.fixture
def server(monkeypatch) -> Server:
    monkeypatch.setenv(SNAPSHOTS_ENV, "content")
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", OWNER)
    monkeypatch.delenv("BENCHFLOW_FRESH_INSTALL", raising=False)
    if content is not None:
        monkeypatch.setattr(content, "_POLL_START_SEC", 0.01)
        monkeypatch.setattr(content, "_POLL_MAX_SEC", 0.02)
    return Server()


def _use(monkeypatch, client: Client) -> None:
    manager = SimpleNamespace(get_client=AsyncMock(return_value=client))
    monkeypatch.setattr(
        daytona_mod,
        "DaytonaClientManager",
        SimpleNamespace(get_instance=AsyncMock(return_value=manager)),
    )


async def _start(
    monkeypatch,
    client: Client,
    rollout_dir: Path,
    *,
    image: str | None = TASK_IMAGE,
    dockerfile: Path | None = None,
    configure: bool = True,
    force_build: bool = False,
    cpus: int = 1,
    memory_mb: int = 1024,
    storage_mb: int = 1024,
    build_timeout_sec: float = 600,
    **options: Any,
):
    _use(monkeypatch, client)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    sandbox = _sandbox(client, rollout_dir, image=image, dockerfile=dockerfile)  # type: ignore[arg-type]
    sandbox.task_env_config.cpus = cpus
    sandbox.task_env_config.memory_mb = memory_mb
    sandbox.task_env_config.storage_mb = storage_mb
    sandbox.task_env_config.build_timeout_sec = build_timeout_sec
    for name, value in options.items():
        if name.startswith("_"):
            setattr(sandbox, name, value)
    if configure:
        _configure(
            sandbox,
            agent=options.get("agent_name", AGENT),
            model=MODEL,
            agent_env=AGENT_ENV,
            fresh_install=options.get("fresh", False),
        )
    await sandbox._strategy.start(force_build=force_build)
    return sandbox


def _record(sandbox) -> dict:
    return sandbox._install_layers_record or {}


def _is_own(name: str) -> bool:
    import re

    return re.fullmatch(re.escape(PREFIX) + "[0-9a-f]{16}", name or "") is not None


def _task(root: Path, dockerfile: str, files: dict[str, str] | None = None) -> Path:
    env = root / "environment"
    env.mkdir(parents=True)
    (env / "Dockerfile").write_text(dockerfile)
    for name, text in (files or {}).items():
        path = env / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return env / "Dockerfile"


DOCKERFILE = f"FROM {PINNED_BASE}\nCOPY app /app\nCOPY run.sh /run.sh\n"
FILES = {"app/main.py": "print('hi')\n", "run.sh": "#!/bin/sh\n"}


# -- creates and starts from it --------------------------------------------


@pytest.mark.parametrize(
    ("agent", "model", "install_agent", "setup_timeout", "allowance"),
    [
        (AGENT, MODEL, True, 120, 1500),
        (AGENT, MODEL, True, 1200, 2100),
        (AGENT, None, True, 120, 900),
        (AGENT, MODEL, False, 120, 600),
        ("nop", None, True, 120, 0),
    ],
    ids=["both", "larger-setup-budget", "agent-only", "proxy-only", "no-layers"],
)
async def test_content_snapshot_preserves_the_recipe_install_budgets(
    monkeypatch, server, tmp_path, agent, model, install_agent, setup_timeout, allowance
):
    """Guards the U3b merge of b89ae706 against a fixed snapshot build budget."""
    assert content is not None
    ensure = AsyncMock(wraps=content.ensure_snapshot)
    monkeypatch.setattr(content, "ensure_snapshot", ensure)
    client = Client(server)
    _use(monkeypatch, client)
    sandbox = _sandbox(client, tmp_path)  # type: ignore[arg-type]
    sandbox.task_env_config.build_timeout_sec = 61.6
    # The integration base predates this option. Reach its snapshot timeout
    # too, so the fails-before assertion measures the budget, not its API.
    setup_options = (
        {"sandbox_setup_timeout": setup_timeout}
        if "sandbox_setup_timeout"
        in inspect.signature(sandbox.configure_install_layers).parameters
        else {}
    )
    _configure(
        sandbox,
        agent=agent,
        model=model,
        agent_env=AGENT_ENV,
        install_agent=install_agent,
        **setup_options,
    )

    await sandbox._strategy.start(force_build=False)

    assert _record(sandbox)["snapshot"] == "created"
    assert len(server.create_calls) == 1
    assert isinstance(client.requests[0], sdk.CreateSandboxFromSnapshotParams)
    assert ensure.await_count == 1
    assert ensure.await_args.kwargs["timeout_sec"] == 62 + allowance
    assert sandbox._layer_build_allowance_sec == allowance


async def test_a_trial_starts_from_a_content_snapshot(monkeypatch, server, tmp_path):
    client = Client(server)
    sandbox = await _start(monkeypatch, client, tmp_path / "trial-1")
    (request,) = client.requests
    assert isinstance(request, sdk.CreateSandboxFromSnapshotParams)
    assert request.snapshot.startswith(PREFIX)
    assert _is_own(request.snapshot)
    record = _record(sandbox)
    assert record["snapshot"] == "created"
    assert record["snapshot_name"] == request.snapshot
    (made,) = server.create_calls
    # The snapshot takes the trial's resources; the sandbox its labels and network.
    assert made.resources.disk == 4 and made.resources.cpu == 1
    owned = daytona_mod._benchflow_owned_labels()
    assert {k: request.labels.get(k) for k in owned} == owned
    assert request.network_block_all is False
    # The layers it holds are the recipe's, keys and all; the trial still
    # checks each key file before it skips an install.
    assert set(sandbox._install_layer_keys) == {"proxy-litellm", f"agent-{AGENT}"}
    assert all(v["source"] == "snapshot" for v in record["layers"].values())


async def test_a_second_trial_reuses_it(monkeypatch, server, tmp_path):
    client = Client(server)
    first = await _start(monkeypatch, client, tmp_path / "trial-1")
    second = await _start(monkeypatch, client, tmp_path / "trial-2")
    assert _record(second)["snapshot"] == "reused"
    assert _record(second)["snapshot_name"] == _record(first)["snapshot_name"]
    assert len(server.create_calls) == 1
    assert second._install_layer_keys == first._install_layer_keys


async def test_two_copies_of_one_task_in_two_folders_share_the_name(
    monkeypatch, server, tmp_path
):
    names = []
    for copy in ("copy-a", "copy-b"):
        dockerfile = _task(tmp_path / copy, DOCKERFILE, FILES)
        os.chmod(dockerfile.parent / "run.sh", 0o755)
        sandbox = await _start(
            monkeypatch,
            Client(server),
            tmp_path / copy / "trial",
            image=None,
            dockerfile=dockerfile,
        )
        names.append(_record(sandbox)["snapshot_name"])
    assert names[0] == names[1] and names[0].startswith(PREFIX)


@pytest.mark.parametrize(
    "change",
    [
        "dockerfile-byte",
        "context-file",
        "file-mode",
        "layer-key",
        "cpu",
        "memory",
        "disk",
    ],
)
async def test_the_name_changes_with_each_input(monkeypatch, server, tmp_path, change):
    def build(root: Path, alter: bool) -> dict:
        text = DOCKERFILE + ("# x\n" if alter and change == "dockerfile-byte" else "")
        files = dict(FILES)
        if alter and change == "context-file":
            files["app/main.py"] = "print('hj')\n"
        dockerfile = _task(root, text, files)
        mode = 0o700 if alter and change == "file-mode" else 0o755
        os.chmod(dockerfile.parent / "run.sh", mode)
        kwargs: dict[str, Any] = {}
        if alter and change == "layer-key":
            kwargs["agent_name"] = "codex-acp"
        if alter and change == "cpu":
            kwargs["cpus"] = 2
        if alter and change == "memory":
            kwargs["memory_mb"] = 2048
        if alter and change == "disk":
            kwargs["storage_mb"] = 4096
        return {"dockerfile": dockerfile, **kwargs}

    names = []
    for alter in (False, True):
        root = tmp_path / ("b" if alter else "a")
        sandbox = await _start(
            monkeypatch,
            Client(server),
            root / "trial",
            image=None,
            **build(root, alter),
        )
        names.append(_record(sandbox)["snapshot_name"])
    assert names[0] and names[1] and names[0] != names[1]


# -- no content snapshot, and why --------------------------------------------


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("unpinned-image", "not pinned by digest"),
        ("unpinned-from", "not pinned by digest"),
        ("add", "ADD"),
        ("symlink", "symlink"),
        ("glob", "glob"),
        ("no-owner", "BENCHFLOW_DAYTONA_OWNER"),
        ("fresh", "--fresh-install"),
        ("force-build", "force_build"),
    ],
)
async def test_what_the_hash_cannot_see_gets_no_snapshot(
    monkeypatch, server, tmp_path, case, reason
):
    kwargs: dict[str, Any] = {}
    if case == "unpinned-image":
        kwargs["image"] = "python:3.13-slim"
    elif case in ("unpinned-from", "add", "symlink", "glob"):
        text = {
            "unpinned-from": "FROM python:3.13-slim\nCOPY run.sh /run.sh\n",
            "add": f"FROM {PINNED_BASE}\nADD run.sh /run.sh\n",
            "symlink": f"FROM {PINNED_BASE}\nCOPY app /app\n",
            "glob": f"FROM {PINNED_BASE}\nCOPY *.sh /\n",
        }[case]
        dockerfile = _task(tmp_path, text, FILES)
        if case == "symlink":
            (dockerfile.parent / "app" / "link").symlink_to("/tmp/victim-a")
        kwargs |= {"image": None, "dockerfile": dockerfile}
    elif case == "no-owner":
        monkeypatch.delenv("BENCHFLOW_DAYTONA_OWNER")
    elif case == "fresh":
        kwargs["fresh"] = True
    elif case == "force-build":
        kwargs["force_build"] = True
        dockerfile = _task(tmp_path, DOCKERFILE, FILES)
        kwargs["dockerfile"] = dockerfile
    client = Client(server)
    sandbox = await _start(monkeypatch, client, tmp_path / "trial", **kwargs)
    assert server.create_calls == []
    (request,) = client.requests
    assert isinstance(request, sdk.CreateSandboxFromImageParams)
    record = _record(sandbox)
    assert record["snapshot"] == "off"
    assert reason in record["snapshot_reason"]


async def test_a_branch_checkpoint_keeps_its_precedence(monkeypatch, server, tmp_path):
    client = Client(server)
    server.put("bf-snap-py8-lock-abcdef-trial-1", "active")
    sandbox = await _start(
        monkeypatch,
        client,
        tmp_path / "trial",
        _start_snapshot_ref="bf-snap-py8-lock-abcdef-trial-1",
    )
    (request,) = client.requests
    assert request.snapshot == "bf-snap-py8-lock-abcdef-trial-1"
    assert server.create_calls == []
    assert _record(sandbox)["snapshot"] == "off"
    assert "branch checkpoint" in _record(sandbox)["snapshot_reason"]


async def test_an_explicit_template_keeps_its_precedence(monkeypatch, server, tmp_path):
    client = Client(server)
    server.put("team-template-reasoning-gym-hard", "active")
    sandbox = await _start(
        monkeypatch,
        client,
        tmp_path / "trial",
        _snapshot_template_name="team-template-{name}",
    )
    (request,) = client.requests
    assert request.snapshot == "team-template-reasoning-gym-hard"
    assert server.create_calls == []
    assert "template" in _record(sandbox)["snapshot_reason"]


async def test_an_unconfigured_sandbox_never_takes_a_content_snapshot(
    monkeypatch, server, tmp_path
):
    """A separate verifier's sandbox plans no install layers: no snapshot."""
    client = Client(server)
    sandbox = await _start(monkeypatch, client, tmp_path / "verifier", configure=False)
    assert server.create_calls == []
    assert isinstance(client.requests[0], sdk.CreateSandboxFromImageParams)
    assert sandbox._install_layers_record is None


# -- two processes, one name ---------------------------------------------------


async def test_two_instances_without_a_shared_lock_create_it_once(
    monkeypatch, server, tmp_path
):
    # Two processes: no in-process lock is shared between them.
    if content is not None:
        monkeypatch.setattr(content, "_name_lock", lambda _name: asyncio.Lock())
    server.get_barrier = []
    one, two = Client(server), Client(server)
    first, second = await asyncio.gather(
        _start(monkeypatch, one, tmp_path / "trial-1"),
        _start(monkeypatch, two, tmp_path / "trial-2"),
    )
    assert server.successes == 1
    assert len(server.create_calls) == 2  # one of them got a conflict
    outcomes = sorted(_record(s)["snapshot"] for s in (first, second))
    assert outcomes == ["created", "waited"]
    conflicts = sum(_record(s)["snapshot_events"]["conflicts"] for s in (first, second))
    assert conflicts == 1
    for client in (one, two):
        (request,) = client.requests
        assert isinstance(request, sdk.CreateSandboxFromSnapshotParams)
        assert request.snapshot == _record(first)["snapshot_name"]


# -- states --------------------------------------------------------------------


async def _name_for(monkeypatch, server, tmp_path) -> str:
    probe = Server()
    probe_client = Client(probe)
    sandbox = await _start(monkeypatch, probe_client, tmp_path / "probe")
    return _record(sandbox)["snapshot_name"]


@pytest.mark.parametrize(
    ("states", "outcome", "deleted", "activated"),
    [
        (["pending", "building", "active"], "waited", 0, 0),
        (["removing", "gone"], "created", 0, 0),
        (["inactive", "active"], "waited", 0, 1),
        (["error", "error"], "created", 1, 0),
    ],
    ids=["pending", "removing", "inactive", "error-replaced"],
)
async def test_each_state_is_handled(
    monkeypatch, server, tmp_path, states, outcome, deleted, activated
):
    name = await _name_for(monkeypatch, server, tmp_path)
    server.put(name, states[0])
    server.get_script[name] = list(states)
    client = Client(server)
    sandbox = await _start(monkeypatch, client, tmp_path / "trial")
    assert _record(sandbox)["snapshot"] == outcome
    assert len(server.deleted) == deleted
    assert len(server.activated) == activated
    assert isinstance(client.requests[0], sdk.CreateSandboxFromSnapshotParams)


async def test_an_error_snapshot_replaced_by_someone_else_is_not_deleted(
    monkeypatch, server, tmp_path
):
    name = await _name_for(monkeypatch, server, tmp_path)
    server.put(name, "error")
    # Between the two reads another creator replaced it; theirs is pending.
    server.get_script[name] = ["error", "replace:pending", "pending", "active"]
    sandbox = await _start(monkeypatch, Client(server), tmp_path / "trial")
    assert server.deleted == []
    assert server.create_calls == []
    assert _record(sandbox)["snapshot"] == "waited"


@pytest.mark.parametrize("state", ["unknown_default_open_api"])
async def test_an_unknown_state_falls_back_to_the_recipe(
    monkeypatch, server, tmp_path, state
):
    name = await _name_for(monkeypatch, server, tmp_path)
    server.put(name, state)
    client = Client(server)
    sandbox = await _start(monkeypatch, client, tmp_path / "trial")
    assert _record(sandbox)["snapshot"] == "fallback"
    assert isinstance(client.requests[0], sdk.CreateSandboxFromImageParams)
    assert server.deleted == []


async def test_a_snapshot_that_stays_pending_times_out_to_the_recipe(
    monkeypatch, server, tmp_path
):
    assert content is not None
    name = await _name_for(monkeypatch, server, tmp_path)
    server.put(name, "pending")
    # Keep the polling deadline short without replacing the recipe's budget.
    ensure = content.ensure_snapshot

    async def short_deadline(*args, **kwargs):
        kwargs["timeout_sec"] = 0.1
        return await ensure(*args, **kwargs)

    monkeypatch.setattr(content, "ensure_snapshot", short_deadline)
    client = Client(server)
    sandbox = await _start(monkeypatch, client, tmp_path / "trial", build_timeout_sec=1)
    record = _record(sandbox)
    assert record["snapshot"] == "fallback"
    assert "not active within" in record["snapshot_reason"]
    assert isinstance(client.requests[0], sdk.CreateSandboxFromImageParams)
    assert server.deleted == []


async def test_a_create_conflict_while_get_finds_nothing_backs_off(monkeypatch):
    """A name the service still holds (409 on create) while `get` answers 404
    spun get and create back to back until the deadline (U13-R1-A-5)."""
    assert content is not None, "no content snapshots in this BenchFlow"
    monkeypatch.setattr(content, "_POLL_START_SEC", 0.05)
    monkeypatch.setattr(content, "_POLL_MAX_SEC", 0.05)
    calls = {"get": 0, "create": 0}

    async def get(name):
        calls["get"] += 1
        raise sdk.DaytonaNotFoundError(f"snapshot {name} not found")

    async def create(params, timeout=None):
        calls["create"] += 1
        raise sdk.DaytonaConflictError("snapshot already exists")

    client = SimpleNamespace(snapshot=SimpleNamespace(get=get, create=create))
    with pytest.raises(content.SnapshotFallback):
        await content.ensure_snapshot(
            client, PREFIX + "0" * 16, lambda: object(), timeout_sec=0.5
        )
    assert 1 <= calls["create"] <= 20, calls


async def test_a_build_failure_falls_back_to_the_layered_recipe(
    monkeypatch, server, tmp_path
):
    server.build_fail = True
    client = Client(server)
    sandbox = await _start(monkeypatch, client, tmp_path / "trial")
    record = _record(sandbox)
    assert record["snapshot"] == "fallback"
    (request,) = client.requests
    assert isinstance(request, sdk.CreateSandboxFromImageParams)
    assert "RUN " in request.image.dockerfile()  # U3b's layers, as without snapshots
    assert set(sandbox._install_layer_keys) == {"proxy-litellm", f"agent-{AGENT}"}
    assert all(v["source"] == "image" for v in record["layers"].values())
    # The failed snapshot it made is removed, never an active one.
    assert record["snapshot_name"] not in server.snaps


async def test_cancellation_propagates_without_a_fallback_sandbox(
    monkeypatch, server, tmp_path
):
    server.create_gate = asyncio.Event()  # the create never finishes
    client = Client(server)
    task = asyncio.create_task(_start(monkeypatch, client, tmp_path / "trial"))
    for _ in range(500):
        if server.create_calls or task.done():
            break
        await asyncio.sleep(0.01)
    assert server.create_calls, "no snapshot create was started"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.requests == []


@pytest.mark.parametrize("layers", [False, True])
@pytest.mark.parametrize("timeout_error", [TimeoutError, sdk.DaytonaTimeoutError])
@pytest.mark.parametrize("lookup", ["empty", "error", "delete-failed", "deleted"])
@pytest.mark.parametrize("source", ["content", "template", "branch", "image"])
async def test_snapshot_create_requires_positive_reconciliation(
    monkeypatch, server, tmp_path, layers, timeout_error, lookup, source
):
    """Guards U3b-B9-C1 at d4e332b1, including the decorated retry path."""
    from tenacity import wait_none

    from benchflow.sandbox.protocol import SandboxStartupError

    # These fake-provider outcomes are fixed; exercise the settling loop without
    # spending the production 90-second window on every parameter combination.
    monkeypatch.setattr(daytona_mod, "_CREATE_SETTLE_SEC", 0.02)
    monkeypatch.setattr(daytona_mod, "_CREATE_WATCH_INTERVAL_SEC", 0.005)

    client = Client(server)
    original_create = client.create
    events = []
    deleted = AsyncMock(side_effect=RuntimeError("delete failed"))

    async def delete(timeout=60, wait=False):
        events.append("deleted")

    async def listed(_query):
        events.append("listed")
        if lookup == "error":
            raise RuntimeError("lookup failed")
        if lookup in {"delete-failed", "deleted"}:
            yield SimpleNamespace(
                id="late-snapshot-sandbox",
                delete=delete if lookup == "deleted" else deleted,
            )

    async def create(params, timeout=None):
        if not events:
            client.requests.append(params)
            events.append("timeout")
            raise timeout_error("snapshot create timed out")
        events.append("next create")
        return await original_create(params, timeout)

    client.create = create
    client.list = listed
    _use(monkeypatch, client)
    sandbox = _sandbox(client, tmp_path)
    _configure(
        sandbox, agent=AGENT if layers else "oracle", model=MODEL, agent_env=AGENT_ENV
    )
    if source in {"template", "branch"}:
        server.put("trusted-checkpoint", "active")
        if source == "template":
            sandbox._snapshot_template_name = "trusted-checkpoint"
        else:
            sandbox._start_snapshot_ref = "trusted-checkpoint"
    elif source == "image":
        monkeypatch.delenv(SNAPSHOTS_ENV)
    sandbox._create_sandbox = daytona_mod.DaytonaSandbox._create_sandbox.retry_with(
        wait=wait_none()
    ).__get__(sandbox)
    if lookup == "deleted":
        await sandbox._strategy.start(force_build=False)
        assert events == ["timeout", "listed", "deleted", "next create"]
        assert not sandbox._create_outcome_unknown
    else:
        with pytest.raises(SandboxStartupError):
            await sandbox._strategy.start(force_build=False)
        assert len(client.requests) == 1
        assert sandbox._create_outcome_unknown
        # Even an explicit second call must not create while the first is unknown.
        with pytest.raises(SandboxStartupError):
            await sandbox._create_sandbox(params=client.requests[0])
        assert len(client.requests) == 1


@pytest.mark.parametrize("layers", [False, True])
async def test_snapshot_terminal_startup_error_never_falls_back(
    monkeypatch, server, tmp_path, layers
):
    """Guards U3b-B9-C1 at d4e332b1: preserve the typed refusal itself."""
    from benchflow.sandbox.protocol import SandboxStartupError

    client = Client(server)
    _use(monkeypatch, client)
    sandbox = _sandbox(client, tmp_path)
    _configure(
        sandbox, agent=AGENT if layers else "oracle", model=MODEL, agent_env=AGENT_ENV
    )
    error = sandbox._startup_failure(TimeoutError("unknown create"))
    sandbox._create_sandbox = AsyncMock(side_effect=[error, None])
    with pytest.raises(SandboxStartupError) as raised:
        await sandbox._strategy.start(force_build=False)
    assert raised.value is error
    assert sandbox._create_sandbox.await_count == 1


@pytest.mark.parametrize("layers", [False, True])
async def test_snapshot_known_failure_still_falls_back_to_image(
    monkeypatch, server, tmp_path, layers
):
    """Guards d4e332b1 and 9b0320f: confirmed cleanup permits retry and fallback."""
    from tenacity import wait_none

    monkeypatch.setattr(
        daytona_mod.DaytonaSandbox._create_sandbox.retry, "wait", wait_none()
    )
    client = Client(server)
    original_create = client.create
    events = []

    async def delete(timeout=60, wait=False):
        assert wait is True and timeout > 0
        events.append("deleted")

    async def get(sandbox_id):
        return SimpleNamespace(id=sandbox_id, delete=delete)

    async def create(params, timeout=None):
        if isinstance(params, sdk.CreateSandboxFromSnapshotParams):
            client.requests.append(params)
            events.append("snapshot failed")
            # Guards 9b0320f: safe fallback requires a known allocation whose
            # deletion the fake client can confirm, not an unanswered create.
            raise sdk.DaytonaError(
                "Sandbox 00000000-0000-0000-0000-000000000002 failed to start "
                "with state: SandboxState.BUILD_FAILED"
            )
        events.append("image created")
        return await original_create(params, timeout)

    client.create = create
    client.get = get
    sandbox = await _start(
        monkeypatch, client, tmp_path, agent_name=AGENT if layers else "oracle"
    )
    # Layered builds fall back immediately; plain builds retain the existing
    # three-attempt policy. Every failed allocation is gone before either path.
    attempts = 1 if layers else 3
    assert events == ["snapshot failed", "deleted"] * attempts + ["image created"]
    assert len(client.requests) == attempts + 1
    assert isinstance(client.requests[-1], sdk.CreateSandboxFromImageParams)
    assert _record(sandbox)["snapshot"] == "fallback"


# -- names and helpers -----------------------------------------------------------


@pytest.mark.parametrize("base", ["ubuntu", "UBUNTU", "registry.example/base:latest"])
def test_current_stage_alias_cannot_pin_its_own_external_base(base):
    """Guard 1bf9bd73: a stage cannot resolve its own external base alias."""
    _, reason = content.check_recipe(f"FROM {base} AS {base}\n")
    assert reason is not None and "not pinned by digest" in reason


def test_a_previously_declared_stage_is_an_immutable_base():
    """Guard 1bf9bd73: earlier stage aliases remain valid bases."""
    refs, reason = content.check_recipe(
        f"FROM {PINNED_BASE} AS base\nFROM base AS next\nCOPY --from=base /a /b\n"
    )
    assert reason is None
    assert refs == [PINNED_BASE]


def test_own_names_match_the_whole_owner_prefix_and_a_content_hash(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", OWNER)
    prefix = content.owner_prefix()
    assert prefix == PREFIX
    assert content.is_own_name(PREFIX + "0123456789abcdef", prefix)
    for name in (
        PREFIX + "0123456789abcde",  # short
        PREFIX + "0123456789abcdefa",  # long
        PREFIX + "0123456789ABCDEF",  # not hex as written
        PREFIX[:-1] + "x0123456789abcdef",
        "bf-snap-" + PREFIX.removeprefix("bf-img2-") + "0123456789abcdef",
        f"bf-img-{OWNER}x-{PREFIX.split('-')[-2]}-0123456789abcdef",
    ):
        assert not content.is_own_name(name, prefix), name


def test_another_task_image_gives_another_name(monkeypatch, tmp_path):
    def name(image: str) -> str | None:
        identity = content.content_identity(
            recipe=f"FROM {image}\n",
            context=[],
            layer_lines=[],
            layer_keys={},
            cpu=1,
            memory=1,
            disk=1,
            api_target="api|default",
        )
        return identity.content_hash

    assert name(TASK_IMAGE) != name(OTHER_IMAGE)
    assert name(TASK_IMAGE) is not None


def test_the_api_target_is_in_the_hash(monkeypatch):
    def name() -> str | None:
        return content.content_identity(
            recipe=f"FROM {TASK_IMAGE}\n",
            context=[],
            layer_lines=[],
            layer_keys={},
            cpu=1,
            memory=1,
            disk=1,
            api_target=content.api_target(),
        ).content_hash

    monkeypatch.delenv("DAYTONA_TARGET", raising=False)
    us = name()
    monkeypatch.setenv("DAYTONA_TARGET", "eu")
    assert name() != us
