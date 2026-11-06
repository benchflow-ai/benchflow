"""Cached install layers on Daytona (proxy and agent), and #1169's disk.

These drive the real direct strategy's ``start()`` and ``_create_sandbox``
against a fake Daytona client that records what a create request would send
(the image recipe, which Daytona caches, and the disk). The SDK's own
``Image``, ``Resources`` and ``CreateSandboxFromImageParams`` build the request.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("daytona")

from benchflow.sandbox import daytona as daytona_mod
from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDirect

daytona_mod._load_daytona_sdk()

TASK_IMAGE = (
    "ghcr.io/example/reasoning-gym-hard@sha256:"
    "1111111111111111111111111111111111111111111111111111111111111111"
)
OTHER_IMAGE = (
    "ghcr.io/example/other-task@sha256:"
    "2222222222222222222222222222222222222222222222222222222222222222"
)
AGENT = "claude-agent-acp"
MODEL = "anthropic/claude-sonnet-4-6"
AGENT_ENV = {"ANTHROPIC_API_KEY": "sk-test-not-a-key"}
_RUN_SCRIPT = re.compile(r"^RUN printf '%s' (\S+) (>{1,2}) ", re.MULTILINE)


class FakeDaytona:
    """Records create requests; one build record per recipe, as Daytona keeps.

    Times come from this fake server's clock (``datetime.now(UTC)``); the
    sandbox's ``created_at`` is ISO text, as the API returns it. ``runner``
    is the runner the next sandbox lands on.
    """

    def __init__(self, *, fail_layered_builds: bool = False) -> None:
        self.requests: list[object] = []
        self.timeouts: list[object] = []
        self.builds: dict[str, datetime] = {}
        self.fail_layered_builds = fail_layered_builds
        self.runner = "runner-a"

    async def create(self, params, timeout=None):
        self.requests.append(params)
        self.timeouts.append(timeout)
        recipe = params.image.dockerfile()
        if self.fail_layered_builds and "RUN " in recipe:
            # The error the SDK raises (DaytonaError), not a bare RuntimeError,
            # so the start path's own classification sees it.
            from daytona.common.errors import DaytonaError

            raise DaytonaError(
                "Failure during waiting for sandbox to start: Sandbox "
                "sb-failed failed to start with state: SandboxState.BUILD_FAILED, "
                "error reason: bash: not found"
            )
        now = datetime.now(UTC)
        built_at = self.builds.setdefault(recipe, now)
        return SimpleNamespace(
            id=f"sb-{len(self.requests)}",
            created_at=now.isoformat().replace("+00:00", "Z"),
            build_info=SimpleNamespace(
                created_at=built_at, snapshot_ref=f"ref-{abs(hash(recipe))}"
            ),
            runner_id=self.runner,
            delete=AsyncMock(),
        )

    async def get(self, sandbox_id):
        return SimpleNamespace(id=sandbox_id, delete=AsyncMock())

    async def list(self, query=None, request_timeout=None):
        # This fake creates no remote resource when a snapshot is rejected.
        # Guards PR #13: definitive 4xx still checks for owned leftovers.
        for _ in ():
            yield _


def _sandbox(
    client: FakeDaytona,
    rollout_dir: Path,
    *,
    image: str | None = TASK_IMAGE,
    storage_mb: int = 1024,
    dockerfile: Path | None = None,
) -> DaytonaSandbox:
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox.logger = daytona_mod.logger
    sandbox.environment_name = "reasoning-gym-hard"
    sandbox.environment_dir = dockerfile.parent if dockerfile else rollout_dir
    sandbox.task_env_config = SimpleNamespace(
        cpus=1,
        memory_mb=1024,
        storage_mb=storage_mb,
        docker_image=image,
        build_timeout_sec=600,
    )
    sandbox.rollout_paths = SimpleNamespace(rollout_dir=rollout_dir)
    sandbox._compose_mode = False
    sandbox._auto_delete_interval = 0
    sandbox._auto_stop_interval = 0
    sandbox._network_block_all = False
    sandbox._snapshot_template_name = None
    sandbox._start_snapshot_ref = None
    sandbox._start_snapshot_env = dict.fromkeys(daytona_mod.SYSTEM_SHELL_EMPTY_VARS, "")
    sandbox._sandbox = None
    sandbox._client_manager = None
    sandbox._create_attempts = 0
    sandbox._strategy = _DaytonaDirect(sandbox)
    # The merged K1 startup requires an explicit root-only log receipt.
    sandbox._sandbox_exec = AsyncMock(
        return_value=SimpleNamespace(
            return_code=0, stdout="BF daemon-log root-only\n", stderr=""
        )
    )
    # This client models image builds, not the toolbox's process transport.
    sandbox._toolbox_startup_env = AsyncMock(
        return_value=dict.fromkeys(daytona_mod.SYSTEM_SHELL_EMPTY_VARS, "")
    )
    return sandbox


def _configure(sandbox: DaytonaSandbox, **kwargs) -> None:
    # On a BenchFlow without install layers this is a no-op, so the
    # assertions below name what is missing rather than an AttributeError.
    configure = getattr(sandbox, "configure_install_layers", None)
    if configure is not None:
        configure(**kwargs)


@pytest.fixture
def client(monkeypatch) -> FakeDaytona:
    fake = FakeDaytona()
    manager = SimpleNamespace(get_client=AsyncMock(return_value=fake))
    monkeypatch.setattr(
        daytona_mod,
        "DaytonaClientManager",
        SimpleNamespace(get_instance=AsyncMock(return_value=manager)),
    )
    monkeypatch.delenv("BENCHFLOW_FRESH_INSTALL", raising=False)
    return fake


async def _start(client: FakeDaytona, rollout_dir: Path, **configure) -> DaytonaSandbox:
    rollout_dir.mkdir(parents=True, exist_ok=True)
    sandbox = _sandbox(client, rollout_dir, image=configure.pop("image", TASK_IMAGE))
    _configure(
        sandbox,
        agent=configure.pop("agent", AGENT),
        model=configure.pop("model", MODEL),
        agent_env=configure.pop("agent_env", AGENT_ENV),
        **configure,
    )
    await sandbox._strategy.start(force_build=False)
    return sandbox


def _scripts(recipe: str) -> list[str]:
    scripts = []
    for encoded, redirect in _RUN_SCRIPT.findall(recipe):
        if redirect == ">":
            scripts.append("")
        scripts[-1] += encoded
    return [base64.b64decode(encoded).decode("utf-8") for encoded in scripts]


async def test_recipe_carries_the_proxy_and_agent_layers_with_their_keys(
    client, tmp_path
):
    from benchflow.agents.registry import AGENT_INSTALLERS
    from benchflow.providers.proxy_requirements import proxy_install
    from benchflow.sandbox.daytona_layers import proxy_install_command

    sandbox = await _start(client, tmp_path / "trial-1")

    recipe = client.requests[0].image.dockerfile()
    assert recipe.startswith(f"FROM {TASK_IMAGE}\n")
    scripts = _scripts(recipe)
    assert len(scripts) == 2, recipe
    # Each layer runs exactly what the trial would otherwise run in the sandbox.
    assert scripts[0].endswith(proxy_install_command(proxy_install(None)) + "\n")
    assert scripts[1].endswith(AGENT_INSTALLERS[AGENT] + "\n")
    keys = sandbox._install_layer_keys
    assert set(keys) == {"proxy-litellm", f"agent-{AGENT}"}
    assert "> /opt/benchflow/daytona-proxy/proxy.key" in recipe
    assert f"printf '%s\\n' {keys['proxy-litellm']} >" in recipe
    assert f"printf '%s\\n' {keys[f'agent-{AGENT}']} >" in recipe
    # The create may also take the time the layers need to build.
    assert client.timeouts == [600 + sandbox._install_layers.build_allowance_sec]
    # Package caches never reach the image.
    assert "rm -rf /tmp/benchflow-layer.sh /tmp/benchflow-layer-cache" in recipe

    record = json.loads((tmp_path / "trial-1" / "sandbox.json").read_text())
    layers = record["install_layers"]
    assert layers["mode"] == "layers"
    assert layers["layers"]["proxy-litellm"]["key"] == keys["proxy-litellm"]
    assert layers["build_record"] == "new"
    assert layers["runner_id"] == "runner-a"
    assert layers["build_created_at"] and layers["sandbox_created_at"]


async def test_an_unchanged_recipe_is_sent_again_and_reused(client, tmp_path):
    first = await _start(client, tmp_path / "trial-1")
    # A later trial of the same task, agent and model on the same runner.
    client.builds = {
        recipe: built - timedelta(minutes=10) for recipe, built in client.builds.items()
    }
    second = await _start(client, tmp_path / "trial-2")

    recipes = [request.image.dockerfile() for request in client.requests]
    assert recipes[0] == recipes[1]
    assert first._install_layer_keys == second._install_layer_keys
    assert first._install_layers_record["build_record"] == "new"
    assert second._install_layers_record["build_record"] == "reused"
    assert (
        first._install_layers_record["build_ref"]
        == second._install_layers_record["build_ref"]
    )


async def test_a_reused_build_record_on_another_runner_is_not_called_a_hit(
    client, tmp_path
):
    """Daytona keeps one build record per recipe, shared by its runners.

    A trial that lands on a runner that never built the recipe sees the first
    build's record, so sandbox.json says only what is known: the record was
    reused, and which runner the trial ran on. Whether it started warm is read
    across trials (same build_ref on the same runner_id), never from a label.
    """
    await _start(client, tmp_path / "trial-1")
    client.builds = {
        recipe: built - timedelta(minutes=10) for recipe, built in client.builds.items()
    }
    client.runner = "runner-b"
    await _start(client, tmp_path / "trial-2")

    first = json.loads((tmp_path / "trial-1" / "sandbox.json").read_text())
    second = json.loads((tmp_path / "trial-2" / "sandbox.json").read_text())
    layers = second["install_layers"]
    assert "cache" not in layers
    assert layers["build_record"] == "reused"
    assert layers["runner_id"] == "runner-b"
    assert first["install_layers"]["runner_id"] == "runner-a"
    assert layers["build_ref"] == first["install_layers"]["build_ref"]


async def test_build_record_ignores_this_machines_clock(client, tmp_path, monkeypatch):
    """A host clock a minute ahead of Daytona's must not turn a build into a reuse."""
    from benchflow.sandbox import daytona_layers, daytona_strategies

    ahead = datetime.now(UTC) + timedelta(minutes=1)
    # Before the fix the label compared Daytona's time with the host's.
    monkeypatch.setattr(daytona_strategies, "now_utc", lambda: ahead, raising=False)
    monkeypatch.setattr(daytona_layers, "now_utc", lambda: ahead, raising=False)

    sandbox = await _start(client, tmp_path / "trial-1")

    record = sandbox._install_layers_record
    assert record.get("cache") != "hit"  # the label the skew used to produce
    assert record["build_record"] == "new"


def test_build_record_is_unknown_without_both_daytona_times():
    from benchflow.sandbox.daytona_layers import build_record_outcome

    built = datetime.now(UTC)
    no_sandbox_time = SimpleNamespace(
        build_info=SimpleNamespace(created_at=built, snapshot_ref="ref-1"),
        created_at=None,
        runner_id="runner-a",
    )
    outcome = build_record_outcome(no_sandbox_time)
    assert outcome["build_record"] == "unknown"
    assert outcome["build_created_at"] == built.isoformat()
    assert outcome["runner_id"] == "runner-a"
    bad_text = SimpleNamespace(
        build_info=SimpleNamespace(created_at=built, snapshot_ref="ref-1"),
        created_at="not a time",
    )
    assert build_record_outcome(bad_text)["build_record"] == "unknown"


async def test_another_task_image_never_shares_a_layer(client, tmp_path):
    first = await _start(client, tmp_path / "trial-1")
    other = await _start(client, tmp_path / "trial-2", image=OTHER_IMAGE)

    assert set(first._install_layer_keys) == set(other._install_layer_keys)
    for name, key in first._install_layer_keys.items():
        assert other._install_layer_keys[name] != key
    assert client.requests[1].image.dockerfile().startswith(f"FROM {OTHER_IMAGE}\n")


@pytest.mark.parametrize("how", ["flag", "env"])
async def test_fresh_install_leaves_the_layers_out(client, tmp_path, monkeypatch, how):
    if how == "env":
        monkeypatch.setenv("BENCHFLOW_FRESH_INSTALL", "1")
        from benchflow.sandbox.daytona_layers import fresh_install_requested

        fresh = fresh_install_requested(None)
    else:
        fresh = True
    sandbox = await _start(client, tmp_path / "trial-1", fresh_install=fresh)

    assert client.requests[0].image.dockerfile() == f"FROM {TASK_IMAGE}\n"
    assert sandbox._install_layer_keys == {}
    record = json.loads((tmp_path / "trial-1" / "sandbox.json").read_text())
    assert record["install_layers"]["mode"] == "fresh"
    # The installs still run in the sandbox, so the disk still holds them.
    assert client.requests[0].resources.disk == 4


async def test_1169_disk_is_storage_mb_plus_the_installs(client, tmp_path):
    """reasoning-gym-hard asks for 1024 MB; its image takes most of it."""
    await _start(client, tmp_path / "trial-1")

    # 1024 MB for the task, 1024 MB for the proxy venv, 1280 MB for the agent.
    assert client.requests[0].resources.disk == 4
    record = json.loads((tmp_path / "trial-1" / "sandbox.json").read_text())
    assert record["install_layers"]["storage_mb"] == 1024
    assert record["install_layers"]["footprint_mb"] == 2304
    assert record["install_layers"]["disk_gib"] == 4


async def test_oracle_trials_keep_the_task_disk_and_image(client, tmp_path):
    sandbox = await _start(client, tmp_path / "trial-1", agent="oracle", model=None)

    assert client.requests[0].image.dockerfile() == f"FROM {TASK_IMAGE}\n"
    assert client.requests[0].resources.disk == 1
    assert sandbox._install_layers_record["mode"] == "none"


async def test_a_layer_that_does_not_build_falls_back_to_the_task_image(
    monkeypatch, tmp_path
):
    fake = FakeDaytona(fail_layered_builds=True)
    manager = SimpleNamespace(get_client=AsyncMock(return_value=fake))
    monkeypatch.setattr(
        daytona_mod,
        "DaytonaClientManager",
        SimpleNamespace(get_instance=AsyncMock(return_value=manager)),
    )
    sandbox = await _start(fake, tmp_path / "trial-1")

    recipes = [request.image.dockerfile() for request in fake.requests]
    assert len(recipes) == 2
    assert "RUN " in recipes[0]
    assert recipes[1] == f"FROM {TASK_IMAGE}\n"
    assert fake.timeouts == [600 + sandbox._install_layers.build_allowance_sec, 600]
    assert sandbox._install_layer_keys == {}
    assert sandbox._install_layers_record["fallback"] == "layer build failed"
    # The trial installs in the sandbox, which still has room for it.
    assert fake.requests[1].resources.disk == 4


async def test_dockerfile_user_is_root_for_the_layers_and_restored(client, tmp_path):
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    dockerfile = env_dir / "Dockerfile"
    dockerfile.write_text("FROM python:3.13-slim\nUSER app")  # no final newline
    sandbox = _sandbox(client, tmp_path, image=None, dockerfile=dockerfile)
    _configure(sandbox, agent=AGENT, model=MODEL, agent_env=AGENT_ENV)

    await sandbox._strategy.start(force_build=False)

    lines = client.requests[0].image.dockerfile().splitlines()
    assert lines[:3] == ["FROM python:3.13-slim", "USER app", "USER root"]
    assert lines[-1] == "USER app"
    assert sum(line.startswith("RUN base64 -d ") for line in lines) == 2


@pytest.mark.parametrize(
    ("dockerfile_text", "final_user"),
    [
        # Only the builder stage sets USER: the final image runs as its own
        # (root) user, so the layers add no USER line at all.
        (
            "FROM node:22 AS build\nUSER node\nRUN npm ci\n"
            "FROM python:3.13-slim\nCOPY --from=build /app /app\n",
            None,
        ),
        # Both stages set USER: the final stage's user is restored.
        (
            "FROM node:22 AS build\nUSER node\nRUN npm ci\n"
            "FROM python:3.13-slim\nUSER app\nCOPY --from=build /app /app\n",
            "app",
        ),
    ],
)
async def test_multi_stage_dockerfile_restores_only_the_final_stage_user(
    client, tmp_path, dockerfile_text, final_user
):
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    dockerfile = env_dir / "Dockerfile"
    dockerfile.write_text(dockerfile_text)
    sandbox = _sandbox(client, tmp_path, image=None, dockerfile=dockerfile)
    _configure(sandbox, agent=AGENT, model=MODEL, agent_env=AGENT_ENV)

    await sandbox._strategy.start(force_build=False)

    lines = client.requests[0].image.dockerfile().splitlines()
    base = dockerfile_text.splitlines()
    assert lines[: len(base)] == base
    added = lines[len(base) :]
    assert sum(line.startswith("RUN base64 -d ") for line in added) == 2
    users = [line for line in added if line.startswith("USER ")]
    if final_user is None:
        assert users == []
    else:
        assert users == ["USER root", f"USER {final_user}"]
        assert added[-1] == f"USER {final_user}"


class _LayerSandbox:
    """A live sandbox whose image carries a layer with key ``image_key``."""

    def __init__(self, layer, key: str, image_key: str) -> None:
        self.layer = layer
        self.key = key
        self.image_key = image_key
        self.exec_calls: list[str] = []
        self.notes: list[tuple[str, str, str]] = []

    def cached_install_layer(self, kind, name):
        if (kind, name) == (self.layer.kind, self.layer.name):
            return self.layer, self.key
        return None

    def note_install_layer(self, kind, name, outcome):
        self.notes.append((kind, name, outcome))

    async def exec(self, command, timeout_sec=None, user=None, **kwargs):
        self.exec_calls.append(command)
        if (
            command.startswith("test -x")
            or command.startswith("cat /opt/")
            or command.startswith("(\nset -eu\nroot=")
        ):
            return SimpleNamespace(return_code=0, stdout=self.image_key + "\n")
        return SimpleNamespace(return_code=0, stdout="", stderr="")


def _agent_layer():
    from benchflow.sandbox.daytona_layers import plan_install_layers

    plan = plan_install_layers(agent=AGENT, model=MODEL, agent_env=AGENT_ENV)
    assert plan.agent is not None and plan.proxy is not None
    return plan


async def test_install_agent_skips_the_install_when_the_layer_is_there(tmp_path):
    from benchflow.agents.install import install_agent

    plan = _agent_layer()
    sandbox = _LayerSandbox(plan.agent, "k" * 32, "k" * 32)

    await install_agent(sandbox, AGENT, tmp_path)

    assert sandbox.exec_calls == [f"cat /opt/benchflow/layers/agent-{AGENT}.key"]
    assert sandbox.notes == [("agent", AGENT, "used")]
    log = (tmp_path / "agent" / "install-stdout.txt").read_text()
    assert "cached image layer" in log


async def test_install_agent_installs_when_the_key_does_not_match(tmp_path):
    from benchflow.agents.install import install_agent
    from benchflow.agents.registry import AGENT_INSTALLERS

    plan = _agent_layer()
    sandbox = _LayerSandbox(plan.agent, "k" * 32, "stale")

    await install_agent(sandbox, AGENT, tmp_path)

    assert AGENT_INSTALLERS[AGENT] in sandbox.exec_calls
    assert sandbox.notes == [("agent", AGENT, "not found, installed fresh")]


async def test_proxy_uses_the_layer_venv_without_installing(monkeypatch):
    from benchflow.providers import litellm_runtime as runtime_mod
    from benchflow.providers.litellm_config import resolve_litellm_route

    plan = _agent_layer()
    sandbox = _LayerSandbox(plan.proxy, "p" * 32, "p" * 32)
    uploaded: dict[str, str] = {}

    async def no_install(*_args, **_kwargs):
        pytest.fail("the proxy was installed although the layer holds it")

    async def upload_text(_sandbox, text, path, _suffix):
        uploaded[path] = text

    async def state(*_args, **_kwargs):
        return {"port": 45999}

    async def healthy(*_args, **_kwargs):
        return None

    monkeypatch.setattr(runtime_mod, "_ensure_sandbox_litellm", no_install)
    monkeypatch.setattr(runtime_mod, "_upload_text", upload_text)
    monkeypatch.setattr(runtime_mod, "_wait_for_sandbox_state", state)
    monkeypatch.setattr(runtime_mod, "_poll_sandbox_health", healthy)
    route = resolve_litellm_route(MODEL, dict(AGENT_ENV))

    await runtime_mod._start_sandbox_litellm(
        sandbox=sandbox,
        route=route,
        master_key="sk-master-test",
        agent_env=dict(AGENT_ENV),
        session_id="s",
        agent_name=AGENT,
    )

    launch = json.loads(
        next(
            text
            for path, text in uploaded.items()
            if path.endswith("launch_config.json")
        )
    )
    assert launch["python"] == "/opt/benchflow/daytona-proxy/venv/bin/python"
    assert launch["litellm"] == "/opt/benchflow/daytona-proxy/venv/bin/litellm"
    assert sandbox.notes == [("proxy", "litellm", "used")]


async def test_proxy_layer_is_not_used_for_a_route_with_other_packages():
    """Guards U3b-B9-C2 at d4e332b1 with the current Vertex locked set."""
    from benchflow.providers.proxy_requirements import proxy_install
    from benchflow.sandbox.daytona_layers import (
        cached_layer_in_sandbox,
        proxy_install_command,
    )

    plan = _agent_layer()
    sandbox = _LayerSandbox(plan.proxy, "p" * 32, "p" * 32)

    layer = await cached_layer_in_sandbox(
        sandbox,
        "proxy",
        "litellm",
        script=proxy_install_command(proxy_install("google-vertex")),
    )

    assert layer is None
    assert sandbox.exec_calls == []


def test_disk_request_rounds_up_and_stays_within_the_account_limit():
    from benchflow.sandbox.daytona_layers import disk_gib

    assert disk_gib(1024, 2304, 10240) == 4
    assert disk_gib(1500, 0, 10240) == 1  # no installs: as before
    assert disk_gib(10240, 2304, 10240) == 10
    assert disk_gib(20480, 2304, 10240) == 20  # never below the task's own ask
    assert disk_gib(512, 1024, 10240) == 2


class _PlannedSandbox:
    """A sandbox from the factory that records the rollout's layer plan."""

    def __init__(self) -> None:
        self.plans: list[dict] = []

    def configure_install_layers(self, **kwargs) -> None:
        self.plans.append(kwargs)


@pytest.mark.parametrize("fresh", [False, True])
async def test_rollout_plans_the_layers_before_the_sandbox_starts(
    monkeypatch, tmp_path, fresh
):
    from benchflow import rollout_planes
    from benchflow.rollout import Rollout, RolloutConfig

    if fresh:
        monkeypatch.setenv("BENCHFLOW_FRESH_INSTALL", "1")
    else:
        monkeypatch.delenv("BENCHFLOW_FRESH_INSTALL", raising=False)
    sandbox = _PlannedSandbox()
    monkeypatch.setattr(
        rollout_planes, "_create_environment", lambda *_args, **_kwargs: sandbox
    )
    config = RolloutConfig(
        task_path=Path(__file__).parent / "examples" / "hello-world-task",
        environment="daytona",
        agent=AGENT,
        model=MODEL,
        agent_env=dict(AGENT_ENV),
        jobs_dir=tmp_path,
    )
    rollout = await Rollout.create(config)
    await rollout.setup()

    assert len(sandbox.plans) == 1
    plan = sandbox.plans[0]
    assert (plan["agent"], plan["model"]) == (AGENT, MODEL)
    assert plan["fresh_install"] is fresh
    assert plan["install_agent"] is True


async def test_sandbox_json_keeps_its_created_at_when_layer_use_is_noted(
    client, tmp_path, monkeypatch
):
    from benchflow.sandbox import metadata

    sandbox = await _start(client, tmp_path / "trial-1")
    path = tmp_path / "trial-1" / "sandbox.json"
    created = json.loads(path.read_text())["created_at"]

    later = datetime.now() + timedelta(minutes=5)
    monkeypatch.setattr(
        metadata, "datetime", SimpleNamespace(now=lambda: later), raising=True
    )
    sandbox.note_install_layer("agent", AGENT, "used")

    record = json.loads(path.read_text())
    assert record["install_layers"]["layers"][f"agent-{AGENT}"]["used"] == "used"
    assert record["created_at"] == created


@pytest.mark.parametrize("name", ["private-agent", AGENT])
async def test_custom_installer_never_enters_the_recipe(
    client, tmp_path, monkeypatch, name
):
    """Guards the custom-installer regression introduced by 551716f4."""
    from benchflow.agents.registry import AGENT_INSTALLERS

    command = "printf synthetic-private-install > /tmp/private-install"
    monkeypatch.setitem(AGENT_INSTALLERS, name, command)
    sandbox = await _start(client, tmp_path / "trial", agent=name)

    assert all(
        command not in script
        for script in _scripts(client.requests[0].image.dockerfile())
    )
    assert sandbox._install_layers.agent is None
    assert client.requests[0].resources.disk >= 3  # fresh installs still need space


@pytest.mark.parametrize("value", ["A", "B"])
async def test_runtime_install_environment_uses_fresh_installs(client, tmp_path, value):
    """Guards the install-environment regression introduced by 551716f4."""
    sandbox = _sandbox(client, tmp_path)
    sandbox._persistent_env = {"npm_config_registry": value}
    _configure(sandbox, agent=AGENT, model=MODEL, agent_env=AGENT_ENV)

    await sandbox._strategy.start(force_build=False)

    assert client.requests[0].image.dockerfile() == f"FROM {TASK_IMAGE}\n"
    assert sandbox._install_layer_keys == {}
    assert client.requests[0].resources.disk == 4


def test_layers_force_copy_and_validate_after_cache_removal():
    """Guards the cache-removal regression introduced by 551716f4."""
    from benchflow.sandbox.daytona_layers import layered_recipe

    recipe = "\n".join(layered_recipe(_agent_layer(), f"FROM {TASK_IMAGE}\n").lines)
    assert all("UV_LINK_MODE=copy" in script for script in _scripts(recipe))
    proxy_line = next(
        line for line in recipe.splitlines() if line.startswith("RUN base64 -d ")
    )
    assert (
        proxy_line.index("rm -rf")
        < proxy_line.index("-c 'import litellm'")
        < proxy_line.index(".key")
    )


async def test_layer_build_budget_preserves_install_budgets(client, tmp_path):
    """Guards the layer build-timeout regression introduced by 551716f4."""
    from benchflow.agents.install import effective_install_timeout

    sandbox = _sandbox(client, tmp_path)
    sandbox.task_env_config.build_timeout_sec = 60
    _configure(sandbox, agent=AGENT, model=MODEL, agent_env=AGENT_ENV)
    await sandbox._strategy.start(force_build=False)

    assert client.timeouts == [60 + 600 + effective_install_timeout(AGENT)]


@pytest.mark.parametrize("timeout_kind", ["builtin", "sdk"])
async def test_layer_timeout_reconciles_before_plain_image_fallback(
    client, tmp_path, monkeypatch, timeout_kind
):
    """Guards the layer build-timeout regression introduced by 551716f4."""
    events = []
    from daytona.common.errors import DaytonaTimeoutError

    timeout_error = DaytonaTimeoutError if timeout_kind == "sdk" else TimeoutError
    original_create = client.create
    leftover = SimpleNamespace(
        id="timed-out-create",
        delete=AsyncMock(side_effect=lambda **kwargs: events.append("deleted")),
    )

    async def create(params, timeout=None):
        if "RUN " in params.image.dockerfile():
            raise timeout_error("layer build timeout")
        events.append("plain create")
        return await original_create(params, timeout)

    async def listed(_query):
        events.append("listed")
        yield leftover

    monkeypatch.setattr(client, "create", create)
    monkeypatch.setattr(client, "list", listed, raising=False)
    sandbox = _sandbox(client, tmp_path)
    _configure(sandbox, agent=AGENT, model=MODEL, agent_env=AGENT_ENV)
    # No retry sleeps; exercise the real attempt and the strategy's fallback.
    sandbox._create_sandbox = (
        daytona_mod.DaytonaSandbox._create_sandbox.__wrapped__.__get__(sandbox)
    )

    await sandbox._strategy.start(force_build=False)

    assert events == ["listed", "deleted", "plain create"]
    assert sandbox._install_layer_keys == {}


async def test_layer_timeout_with_unknown_resources_never_starts_fallback(
    client, tmp_path, monkeypatch
):
    """Guards the reconciliation boundary for the layer build timeout (551716f4)."""
    from benchflow.sandbox.protocol import SandboxStartupError

    create = AsyncMock(side_effect=TimeoutError("layer build timeout"))
    monkeypatch.setattr(client, "create", create)
    sandbox = _sandbox(client, tmp_path)
    _configure(sandbox, agent=AGENT, model=MODEL, agent_env=AGENT_ENV)
    sandbox._create_sandbox = (
        daytona_mod.DaytonaSandbox._create_sandbox.__wrapped__.__get__(sandbox)
    )
    with pytest.raises(SandboxStartupError):
        await sandbox._strategy.start(force_build=False)
    assert create.await_count == 1
    assert sandbox._create_outcome_unknown


async def test_layer_timeout_with_no_visible_resource_is_still_unknown(
    client, tmp_path, monkeypatch
):
    """Guards the layer build timeout (551716f4): absence is not remote completion."""
    from tenacity import wait_none

    from benchflow.sandbox.protocol import SandboxStartupError

    create = AsyncMock(side_effect=TimeoutError("layer build timeout"))
    monkeypatch.setattr(client, "create", create)

    async def empty(_query):
        for resource in ():
            yield resource

    monkeypatch.setattr(client, "list", empty, raising=False)
    sandbox = _sandbox(client, tmp_path)
    _configure(sandbox, agent=AGENT, model=MODEL, agent_env=AGENT_ENV)
    sandbox._create_sandbox = daytona_mod.DaytonaSandbox._create_sandbox.retry_with(
        wait=wait_none()
    ).__get__(sandbox)
    with pytest.raises(SandboxStartupError):
        await sandbox._strategy.start(force_build=False)
    assert create.await_count == 1
    assert sandbox._create_outcome_unknown


async def test_daytona_late_installer_override_runs_in_the_trial(tmp_path, monkeypatch):
    """Guards the installer-override sibling in Daytona's 551716f4 cache path."""
    from benchflow.agents.install import install_agent
    from benchflow.agents.registry import AGENT_INSTALLERS

    plan = _agent_layer()
    sandbox = _LayerSandbox(plan.agent, "k" * 32, "k" * 32)
    command = "printf synthetic-late-installer"
    monkeypatch.setitem(AGENT_INSTALLERS, AGENT, command)
    await install_agent(sandbox, AGENT, tmp_path)
    assert command in sandbox.exec_calls


@pytest.mark.parametrize(
    "mutator", ["hook", "upload", "plane", "setup", "healthcheck", "manifest"]
)
async def test_rollout_keeps_daytona_installs_after_preinstall_mutators(
    tmp_path, monkeypatch, mutator
):
    """Guards the pre-install mutator siblings in Daytona's 551716f4 layers."""
    from benchflow import rollout_planes
    from benchflow.rollout import Rollout, RolloutConfig

    sandbox = _PlannedSandbox()
    monkeypatch.setattr(
        rollout_planes, "_create_environment", lambda *_a, **_kw: sandbox
    )
    extra = {}
    task_path = Path(__file__).parent / "examples" / "hello-world-task"
    if mutator == "hook":
        extra["pre_agent_hooks"] = [AsyncMock()]
    elif mutator == "upload":
        extra["uploads"] = {"fixture": "/opt/benchflow/bin/fixture"}
    elif mutator == "plane":
        extra["planes"] = rollout_planes.DefaultRolloutPlanes()
    elif mutator == "manifest":
        from benchflow.environment.manifest import load_manifest

        extra["environment_manifest"] = load_manifest(
            Path(__file__).parent
            / "integration"
            / "deterministic"
            / "task"
            / "environment.toml"
        )
    else:
        copied = tmp_path / "fixture"
        shutil.copytree(task_path, copied, ignore=shutil.ignore_patterns("task.md"))
        task_path = copied
        addition = (
            '\n[[environment.setup_commands]]\ncommand = "printf setup"\n'
            if mutator == "setup"
            else '\n[environment.healthcheck]\ncommand = "true"\n'
        )
        with (copied / "task.toml").open("a") as handle:
            handle.write(addition)
    config = RolloutConfig(
        task_path=task_path,
        environment="daytona",
        agent=AGENT,
        model=MODEL,
        agent_env=dict(AGENT_ENV),
        jobs_dir=tmp_path,
        **extra,
    )
    rollout = await Rollout.create(config)
    await rollout.setup()
    assert sandbox.plans[-1]["fresh_install"] is True


async def test_daytona_rechecks_mutators_between_setup_and_start(tmp_path, monkeypatch):
    """Guards the public lifecycle boundary of the 551716f4 layer hook."""
    from benchflow import rollout as rollout_mod
    from benchflow import rollout_planes
    from benchflow.rollout import Rollout, RolloutConfig

    sandbox = _PlannedSandbox()
    monkeypatch.setattr(
        rollout_planes, "_create_environment", lambda *_a, **_kw: sandbox
    )
    for name in (
        "_start_env_and_upload",
        "_run_environment_healthcheck",
        "_run_environment_setup_commands",
    ):
        monkeypatch.setattr(rollout_mod, name, AsyncMock())
    rollout = await Rollout.create(
        RolloutConfig(
            task_path=Path(__file__).parent / "examples" / "hello-world-task",
            environment="daytona",
            agent=AGENT,
            model=MODEL,
            agent_env=dict(AGENT_ENV),
            jobs_dir=tmp_path,
        )
    )
    await rollout.setup()
    assert sandbox.plans[-1]["fresh_install"] is False
    rollout._config.pre_agent_hooks = [AsyncMock()]
    await rollout.start()
    assert sandbox.plans[-1]["fresh_install"] is True


@pytest.mark.parametrize("flag", ["_env_externally_owned", "_from_branch_snapshot"])
def test_daytona_replanning_leaves_borrowed_and_snapshot_sandboxes_alone(
    tmp_path, flag
):
    """Preserves the 551716f4 hook's ownership boundary on the new recheck."""
    from benchflow.rollout import Rollout, RolloutConfig

    sandbox = _PlannedSandbox()
    rollout = Rollout(
        RolloutConfig(task_path=tmp_path, environment="daytona", agent=AGENT)
    )
    rollout._env = sandbox
    setattr(rollout, flag, True)
    rollout._configure_daytona_layers()
    assert sandbox.plans == []
