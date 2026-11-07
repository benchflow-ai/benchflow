"""A Daytona image whose build key cannot start is built again under a new key.

Daytona keeps a build under a key of its recipe's Dockerfile text and
contexts. A build that a delete cancelled stays failed under that key: every
later create fails within a second with "BUILD_FAILED, error reason: context
canceled". A built image whose runner accepts creates and never starts them
fails every create of that key after about 15 minutes. On 0.9 a fresh create
of the same recipe was all that followed (three creates of a cancelled build,
or the plain-image fallback of a layered one), so the task never started.

Guards the fix for the gap a benchmark's own infrastructure works around as
a patch: the same image is created again with one trailing Dockerfile
comment (salts 1 to 3), after the failed sandbox's removal is confirmed.
The SDK's own Image and create params build the request; the client is fake.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("daytona")

from daytona import (
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    Image,
)
from daytona.common.errors import DaytonaError, create_daytona_error

from benchflow.sandbox import daytona as daytona_mod
from benchflow.sandbox import daytona_ledger as ledger_module
from benchflow.sandbox import daytona_start
from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDirect
from benchflow.sandbox.daytona_ledger import Ledger
from benchflow.sandbox.protocol import SandboxStartupError

daytona_mod._load_daytona_sdk()

DOCKERFILE = "FROM python:3.12-slim\nRUN echo task\n"
_WAIT = "Failed to create sandbox: Failure during waiting for sandbox to start: "


def _salt(recipe: str, n: int) -> str:
    return f"{recipe}# benchflow: Daytona rebuild {n}\n"


def _cancelled(sandbox_id: str) -> Exception:
    return DaytonaError(
        f"{_WAIT}Sandbox {sandbox_id} entered error state: "
        "SandboxState.BUILD_FAILED, error reason: context canceled"
    )


def _stuck(ending: str, sandbox_id: str) -> Exception:
    """The four endings of a create its runner never started, as the SDK
    raised them on 2026-11-01 (the reason can still be unset; or a 502)."""
    if ending == "502":
        return create_daytona_error(
            f"{_WAIT}Failed to refresh sandbox data: <html><head><title>"
            "502 Bad Gateway</title></head></html>",
            status_code=502,
        )
    reason = {
        "timeout": "Timeout while creating sandbox",
        "job": "Job timed out - no update received for 30 minutes",
        "none": "None",
    }[ending]
    return DaytonaError(
        f"{_WAIT}Sandbox {sandbox_id} entered error state: SandboxState.ERROR, "
        f"error reason: {reason}"
    )


class FakeDaytona:
    """Fails a create whose recipe ``failures`` names, with the next error
    queued for it; any other create starts. Failed sandboxes can be looked
    up by id or by the create's labels, and deleted."""

    def __init__(self, failures: dict[str, list[Exception]] | None = None) -> None:
        self.failures = failures or {}
        self.recipes: list[str] = []
        self.deleted: list[str] = []
        self.left: dict[str, SimpleNamespace] = {}

    def _box(self, sandbox_id: str, labels: dict, state: str) -> SimpleNamespace:
        async def delete(*_a, **_kw):
            self.deleted.append(sandbox_id)

        return SimpleNamespace(
            id=sandbox_id, labels=dict(labels), state=state, delete=delete
        )

    async def create(self, params, timeout=None):
        image = getattr(params, "image", None)
        recipe = image.dockerfile() if image else f"snapshot {params.snapshot}"
        self.recipes.append(recipe)
        queued = self.failures.get(recipe)
        if queued:
            error = queued.pop(0) if len(queued) > 1 else queued[0]
            sandbox_id = f"sb-{len(self.recipes)}"
            # The failed sandbox Daytona holds; the SDK raised without it.
            self.left[sandbox_id] = self._box(sandbox_id, params.labels, "error")
            text = str(error).replace("sb-ID", sandbox_id)
            raise type(error)(text, status_code=getattr(error, "status_code", None))
        return SimpleNamespace(
            id=f"sb-{len(self.recipes)}",
            labels=dict(params.labels),
            state="started",
            delete=AsyncMock(),
            refresh_activity=AsyncMock(),
            set_labels=AsyncMock(),
        )

    async def get(self, sandbox_id):
        return self.left[sandbox_id]

    async def list(self, query=None, request_timeout=None):
        wanted = dict(getattr(query, "labels", None) or {})
        for box in list(self.left.values()):
            if box.id not in self.deleted and all(
                box.labels.get(k) == v for k, v in wanted.items()
            ):
                yield box


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHFLOW_DAYTONA_LEFTOVERS", str(tmp_path / "leftovers"))
    monkeypatch.setattr(daytona_start, "wait_sec", lambda *_: 0.0)
    # A runner that never starts a sandbox ends its create after ~15 minutes,
    # past every same-recipe retry budget; the fake fails at once.
    monkeypatch.setattr(daytona_start, "CREATE_CAPACITY_BUDGET_SEC", 0.0)
    monkeypatch.setattr(daytona_start, "CREATE_TRANSIENT_BUDGET_SEC", 0.0)
    ledger = Ledger()
    monkeypatch.setattr(daytona_mod, "LEDGER", ledger)
    monkeypatch.setattr(ledger_module, "LEDGER", ledger)
    monkeypatch.setattr(ledger, "ensure_sweeper", lambda: None)
    yield ledger
    ledger.entries.clear()


def _sandbox(client: FakeDaytona, rollout_dir: Path) -> DaytonaSandbox:
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "Dockerfile").write_text(DOCKERFILE)
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox.logger = logging.getLogger("test.daytona.rebuild")
    sandbox.environment_name = "task"
    sandbox.environment_dir = rollout_dir
    sandbox.task_env_config = SimpleNamespace(
        cpus=1,
        memory_mb=1024,
        storage_mb=1024,
        docker_image=None,
        build_timeout_sec=600,
    )
    sandbox.rollout_paths = SimpleNamespace(rollout_dir=rollout_dir)
    sandbox._compose_mode = False
    sandbox._auto_delete_interval = 0
    sandbox._auto_stop_interval = 0
    sandbox._network_block_all = False
    sandbox._snapshot_template_name = None
    sandbox._start_snapshot_ref = None
    sandbox._sandbox = None
    sandbox._create_attempts = 0
    sandbox._client_manager = SimpleNamespace(get_client=AsyncMock(return_value=client))
    sandbox._strategy = _DaytonaDirect(sandbox)
    sandbox._sandbox_exec = AsyncMock(
        return_value=SimpleNamespace(
            return_code=0, stdout="BF daemon-log root-only\n", stderr=""
        )
    )
    # The client models image builds, not the toolbox's process transport.
    sandbox._toolbox_startup_env = AsyncMock(
        return_value=dict.fromkeys(daytona_mod.SYSTEM_SHELL_EMPTY_VARS, "")
    )
    return sandbox


def _params(rollout_dir: Path) -> CreateSandboxFromImageParams:
    return CreateSandboxFromImageParams(
        image=Image.from_dockerfile(rollout_dir / "Dockerfile"),
        labels={"benchflow.managed": "1"},
    )


async def test_a_build_cached_as_cancelled_is_rebuilt_under_salt_1(tmp_path):
    client = FakeDaytona({DOCKERFILE: [_cancelled("sb-ID")]})
    sandbox = _sandbox(client, tmp_path)

    await sandbox._create_sandbox_without_toolbox_bash_env(_params(tmp_path))

    assert client.recipes == [DOCKERFILE, _salt(DOCKERFILE, 1)]
    assert client.deleted == ["sb-1"], "the failed build's sandbox goes first"
    assert sandbox._sandbox.id == "sb-2"
    assert sandbox._daytona_rebuild == {"reason": "cancelled", "salt": 1}
    record = json.loads((tmp_path / "sandbox.json").read_text())
    assert record["daytona_rebuild"] == {"reason": "cancelled", "salt": 1}


async def test_salts_go_in_order_and_the_first_error_stands_after_the_last(
    tmp_path,
):
    """Every salt is cached as cancelled too: salts 1, 2 and 3 in order, each
    failed sandbox deleted, then the plain recipe's own error."""
    salted = [_salt(DOCKERFILE, n) for n in (1, 2, 3)]
    client = FakeDaytona(
        {recipe: [_cancelled("sb-ID")] for recipe in [DOCKERFILE, *salted]}
    )
    sandbox = _sandbox(client, tmp_path)

    with pytest.raises(DaytonaError, match="sb-1 entered error state"):
        await sandbox._create_sandbox_without_toolbox_bash_env(_params(tmp_path))

    assert client.recipes == [DOCKERFILE, *salted]
    assert client.deleted == ["sb-1", "sb-2", "sb-3", "sb-4"]
    assert sandbox._daytona_rebuild is None


async def test_a_salted_build_that_fails_otherwise_raises_its_own_error(tmp_path):
    """A broken Dockerfile under the salt is the task's error: it gets 0.9's
    fresh creates of that recipe and is raised as it is."""
    broken = DaytonaError(
        f"{_WAIT}Sandbox sb-ID entered error state: SandboxState.BUILD_FAILED, "
        "error reason: exit code: 1"
    )
    client = FakeDaytona(
        {DOCKERFILE: [_cancelled("sb-ID")], _salt(DOCKERFILE, 1): [broken]}
    )
    sandbox = _sandbox(client, tmp_path)

    with pytest.raises(DaytonaError, match="exit code: 1"):
        await sandbox._create_sandbox_without_toolbox_bash_env(_params(tmp_path))

    assert client.recipes == [DOCKERFILE, _salt(DOCKERFILE, 1)]
    assert sandbox._daytona_rebuild is None


@pytest.mark.parametrize("ending", ["timeout", "job", "none", "502"])
async def test_a_create_its_runner_never_started_is_rebuilt(tmp_path, ending):
    """The stuck sandbox is deleted first (0.9 confirms a failed start's
    removal before any new create), then the salted recipe starts."""
    client = FakeDaytona({DOCKERFILE: [_stuck(ending, "sb-ID")]})
    sandbox = _sandbox(client, tmp_path)

    await sandbox._create_sandbox_without_toolbox_bash_env(_params(tmp_path))

    assert client.recipes == [DOCKERFILE, _salt(DOCKERFILE, 1)]
    assert client.deleted == ["sb-1"]
    assert sandbox._daytona_rebuild == {"reason": "stuck", "salt": 1}


async def test_a_failed_start_whose_sandbox_is_not_confirmed_gone_is_not_rebuilt(
    tmp_path, isolated
):
    """0.9's rule: no new create while an earlier one's sandbox may exist.
    The cancelled build's sandbox cannot be deleted, so the start fails."""
    client = FakeDaytona({DOCKERFILE: [_cancelled("sb-ID")]})

    async def refused(*_a, **_kw):
        raise DaytonaError("delete refused", status_code=409)

    original_create = client.create

    async def create(params, timeout=None):
        try:
            return await original_create(params, timeout)
        finally:
            for box in client.left.values():
                box.delete = refused

    client.create = create
    sandbox = _sandbox(client, tmp_path)

    with pytest.raises(SandboxStartupError, match="context canceled"):
        await sandbox._create_sandbox_without_toolbox_bash_env(_params(tmp_path))

    assert client.recipes == [DOCKERFILE]
    assert sandbox._create_outcome_unknown
    assert "sandbox:sb-1" in isolated.entries


async def test_a_snapshot_create_keeps_its_error_after_one_create(
    tmp_path, monkeypatch
):
    """A create with no recipe has no build key to change; a fresh create of
    a build cached as cancelled fails the same way, so none is made (0.9 made
    three within the start budget)."""
    monkeypatch.setattr(daytona_start, "CREATE_CAPACITY_BUDGET_SEC", 600.0)
    client = FakeDaytona({"snapshot snap": [_cancelled("sb-ID")]})
    sandbox = _sandbox(client, tmp_path)
    params = CreateSandboxFromSnapshotParams(
        snapshot="snap", labels={"benchflow.managed": "1"}
    )

    with pytest.raises(DaytonaError, match="context canceled"):
        await sandbox._create_sandbox_without_toolbox_bash_env(params)
    assert client.recipes == ["snapshot snap"]


def test_rebuild_reason_reads_only_the_sdks_own_errors():
    """BenchFlow's startup error wrapping one (its cleanup was not confirmed)
    is never rebuilt, nor are other ERROR reasons or a 502 on the create
    request itself."""
    assert daytona_start.rebuild_reason(_cancelled("a")) == "cancelled"
    for ending in ("timeout", "job", "none", "502"):
        assert daytona_start.rebuild_reason(_stuck(ending, "a")) == "stuck", ending
    wrapped = SandboxStartupError(f"Sandbox creation failed: {_cancelled('a')}")
    assert daytona_start.rebuild_reason(wrapped) is None
    for other in (
        DaytonaError(
            f"{_WAIT}Sandbox a entered error state: SandboxState.ERROR, "
            "error reason: context canceled"
        ),
        DaytonaError(
            f"{_WAIT}Sandbox a entered error state: SandboxState.ERROR, "
            "error reason: None of the runners has capacity"
        ),
        DaytonaError(
            f"{_WAIT}Sandbox a entered error state: SandboxState.BUILD_FAILED, "
            "error reason: Timeout while creating sandbox"
        ),
        create_daytona_error(
            "Failed to create sandbox: <html>502 Bad Gateway</html>", status_code=502
        ),
        RuntimeError("context canceled"),
    ):
        assert daytona_start.rebuild_reason(other) is None, str(other)


# ── With install layers ─────────────────────────────────────────────────


async def _layered_start(
    client: FakeDaytona, rollout_dir: Path, monkeypatch
) -> DaytonaSandbox:
    """The direct strategy's start of a prebuilt-image task with the proxy
    and agent installs as cached layers."""
    monkeypatch.delenv("BENCHFLOW_FRESH_INSTALL", raising=False)
    manager = SimpleNamespace(get_client=AsyncMock(return_value=client))
    monkeypatch.setattr(
        daytona_mod,
        "DaytonaClientManager",
        SimpleNamespace(get_instance=AsyncMock(return_value=manager)),
    )
    sandbox = _sandbox(client, rollout_dir)
    sandbox.task_env_config.docker_image = "python:3.12-slim"
    sandbox.configure_install_layers(
        agent="claude-agent-acp",
        model="anthropic/claude-sonnet-4-6",
        agent_env={"ANTHROPIC_API_KEY": "sk-test-not-a-key"},
    )
    await sandbox._strategy.start(force_build=False)
    return sandbox


async def test_a_layered_recipe_is_rebuilt_with_its_layers_before_any_fallback(
    monkeypatch, tmp_path
):
    """The salt goes after the layer lines (the text the SDK sends, which
    Daytona keys), so the trial keeps its cached installs; the layer keys the
    sandbox checks are the plain recipe's."""
    probe = FakeDaytona()
    first = await _layered_start(probe, tmp_path / "probe", monkeypatch)
    layered = probe.recipes[0]
    assert "RUN " in layered and first._install_layer_keys

    client = FakeDaytona({layered: [_cancelled("sb-ID")]})
    sandbox = await _layered_start(client, tmp_path / "trial", monkeypatch)

    assert client.recipes == [layered, _salt(layered, 1)]
    assert sandbox._install_layer_keys == first._install_layer_keys
    record = json.loads((tmp_path / "trial" / "sandbox.json").read_text())
    assert record["daytona_rebuild"] == {"reason": "cancelled", "salt": 1}
    assert "fallback" not in record["install_layers"]


async def test_a_layered_rebuild_that_gives_up_falls_back_to_the_plain_image(
    monkeypatch, tmp_path
):
    """Every salted layered recipe is cached as cancelled too: then the
    task image alone, as for a layer that does not build."""
    probe = FakeDaytona()
    await _layered_start(probe, tmp_path / "probe", monkeypatch)
    layered = probe.recipes[0]
    salted = [_salt(layered, n) for n in (1, 2, 3)]

    client = FakeDaytona(
        {recipe: [_cancelled("sb-ID")] for recipe in [layered, *salted]}
    )
    sandbox = await _layered_start(client, tmp_path / "trial", monkeypatch)

    assert client.recipes == [layered, *salted, "FROM python:3.12-slim\n"]
    assert sandbox._install_layers_record["fallback"] == "layer build failed"
    assert sandbox._daytona_rebuild is None
