"""DockerSandbox starts ``main`` from the agent's cached layer (PY8 U3a).

``use_agent_layer(recipe)`` before ``start()``: the task image is built as
before, the layer comes from :class:`DockerLayers` on top of it, and ``main``
starts from the trial's own tag on the layer. Teardown removes that tag,
never the cached layer or its borrowed base name (U10-CONFIRM-1, after 22672877).
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import docker as docker_module
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.env_layers import (
    AGENT,
    AgentLayerRecipe,
    BaseImage,
    DockerLayers,
    LayerError,
    LayerResult,
    alias_owner_state,
)
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths

RECIPE = AgentLayerRecipe(agent="fake-agent", version="1.0.0", install_cmd="true")
KEY = "c" * 64


@pytest.fixture(scope="module")
def loader_probe_image(tmp_path_factory):
    """A harmless constructor records only the install recipe's shell startup."""
    from tests.integration.deterministic import harness as h

    sandbox, reason = h.select_sandbox()
    if sandbox != "docker":
        pytest.skip(reason or "requires real Docker")
    root = tmp_path_factory.mktemp("loader-image")
    (root / "loader.c").write_text(
        r"""
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
__attribute__((constructor)) static void record(void) {
    char b[65536]; int f = open("/proc/self/cmdline", O_RDONLY);
    if (f < 0) return;
    int n = read(f, b, sizeof(b)-1); close(f); if (n < 0) return;
    for (int i = 0; i < n; i++) if (!b[i]) b[i] = ' ';
    b[n] = 0;
    if (strstr(b, "R3_LOADER_PROBE")) {
        f = open("/opt/review-loaded", O_CREAT|O_WRONLY|O_APPEND, 0644);
        if (f >= 0) { write(f, "loaded\n", 7); close(f); }
    }
}
"""
    )
    (root / "Dockerfile").write_text(
        "FROM ubuntu:24.04 AS compiler\n"
        "RUN apt-get update -qq && apt-get install -y -qq gcc libc6-dev\n"
        "COPY loader.c /tmp/loader.c\n"
        "RUN gcc -shared -fPIC -o /tmp/loader.so /tmp/loader.c\n"
        "FROM ubuntu:24.04\n"
        "COPY --from=compiler /tmp/loader.so /opt/review-loader.so\n"
        "RUN useradd -m reviewapp && mkdir /review-work\n"
        "WORKDIR /review-work\nUSER reviewapp\n"
    )
    tag = f"mini-swe-r3-loader:{uuid.uuid4().hex}"
    _probe_docker("build", "--quiet", "-t", tag, str(root))
    try:
        yield tag
    finally:
        _probe_docker("image", "rm", "--no-prune", tag)


def _probe_docker(*args):
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return result.stdout.strip()


@pytest.mark.integration
@pytest.mark.parametrize("variable", ["LD_PRELOAD", "LD_LIBRARY_PATH", None])
async def test_loader_env_skips_layer_and_installs_fresh(
    tmp_path, monkeypatch, loader_probe_image, variable
):
    """Guards residual M1 in 01727c09e: loader effects precede RUN's unsets."""
    from benchflow.agents.install import agent_layer_plan, install_agent
    from benchflow.agents.registry import register_agent

    rid = uuid.uuid4().hex
    name = f"r3-loader-{rid}"
    tag = f"mini-swe-r3:{rid}"
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(
        f"FROM {loader_probe_image}\n"
        + (f"ENV {variable}=/opt/review-loader.so\n" if variable else "")
    )
    _probe_docker("build", "--quiet", "-t", tag, str(context))
    # Isolate the test's trust/locks and prevent shared cache pruning.
    monkeypatch.setenv("BENCHFLOW_AGENT_LAYER_KEEP", "10000")
    monkeypatch.setenv("BENCHFLOW_AGENT_LAYER_MAX_GB", "0")
    monkeypatch.setenv("BENCHFLOW_AGENT_LAYER_MIN_FREE_GB", "0")
    monkeypatch.setattr(
        "benchflow.sandbox.env_layers._trust_root", lambda: tmp_path / "trust"
    )
    register_agent(
        name,
        ": R3_LOADER_PROBE; mkdir -p /opt/runtime; "
        "if test -f /opt/review-loaded; then echo loaded; else echo clean; fi "
        "> /opt/runtime/loader.txt",
        "unused",
        cacheable_version="1",
    )
    plan = agent_layer_plan(name)
    assert plan is not None
    images = [tag]
    records = []
    try:
        # Fresh is the control; both cache attempts must either skip or miss→hit.
        for arm in ("fresh", "cold", "warm"):
            paths = RolloutPaths(rollout_dir=tmp_path / arm)
            paths.mkdir()
            sandbox = DockerSandbox(
                environment_dir=context,
                environment_name=name,
                session_id=f"{name}-{arm}",
                rollout_paths=paths,
                task_env_config=SandboxConfig(docker_image=tag),
            )
            sandbox.default_user = "root"
            sandbox._use_prebuilt = True
            container = f"{name}-{arm}"
            try:
                if arm != "fresh":
                    sandbox.use_agent_layer(plan.recipe)
                    await sandbox._start_from_layers()
                layer = sandbox.agent_layer
                if layer is not None:
                    images.extend([layer.image, layer.tag])
                _probe_docker(
                    "run",
                    "-d",
                    "--name",
                    container,
                    layer.image if layer else tag,
                    "sleep",
                    "600",
                )
                sandbox._main_container = container
                await install_agent(sandbox, name, paths.rollout_dir)
                marker = _probe_docker(
                    "exec", container, "/bin/cat", "/opt/runtime/loader.txt"
                )
                record = sandbox.layer_records.get(AGENT)
                records.append(record)
                (tmp_path / f"{arm}-record.json").write_text(
                    json.dumps({"layer": record, "marker": marker}, indent=2)
                )
                assert marker == "clean", (arm, record, marker)
                if arm != "fresh" and variable:
                    assert layer is None
                    assert record == {
                        "status": "skipped",
                        "reason": f"image env sets {variable}",
                    }
                    log = (paths.rollout_dir / "agent/install-stdout.txt").read_text()
                    assert "=== ran in the cached image layer" not in log
            finally:
                subprocess.run(["docker", "rm", "-f", container], capture_output=True)
                await sandbox._release_layers(delete=True)
        if variable is None:
            assert records[1]["status"] == "miss", records
            assert records[2]["status"] == "hit", records
            assert records[1]["key"] == records[2]["key"]
    finally:
        for image in dict.fromkeys(reversed(images)):
            subprocess.run(
                ["docker", "image", "rm", "--no-prune", image], capture_output=True
            )


@pytest.fixture(autouse=True)
def _no_claims():
    DockerLayers._claims.clear()
    yield
    DockerLayers._claims.clear()


@dataclass
class Rig:
    sandbox: DockerSandbox
    cli: AsyncMock
    # (compose arguments, compose file names, compose environment) per call
    compose: list[tuple[list[str], list[str], dict[str, str]]] = field(
        default_factory=list
    )
    ensure_calls: list[dict[str, Any]] = field(default_factory=list)

    def up(self) -> tuple[list[str], list[str], dict[str, str]]:
        return next(entry for entry in self.compose if entry[0][0] == "up")

    def removed(self) -> list[list[str]]:
        # Guards U3a's alias cleanup across U10's 30518669 merge: image and
        # volume inventories are not layer-image removals.
        return [
            call.args[0]
            for call in self.cli.await_args_list
            if call.args[0][:2] == ["image", "rm"]
        ]


def _rig(
    tmp_path,
    monkeypatch,
    *,
    compose: str | None = None,
    docker_image: str | None = None,
    layer: bool = True,
    sandbox_class: type[DockerSandbox] = DockerSandbox,
) -> Rig:
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM alpine:3.20\n")
    if compose is not None:
        (environment / "docker-compose.yaml").write_text(compose)
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    sandbox = sandbox_class(
        environment_dir=environment,
        environment_name="layer-task",
        session_id="bf-layer",
        rollout_paths=paths,
        task_env_config=SandboxConfig(docker_image=docker_image),
    )
    rig = Rig(
        sandbox=sandbox,
        cli=AsyncMock(
            return_value=SimpleNamespace(stdout="", stderr="", return_code=0)
        ),
    )

    async def run_compose(
        command, check=True, timeout_sec=None, *, raw_output=False, merge_stderr=True
    ):
        rig.compose.append(
            (
                list(command),
                [p.name for p in sandbox._docker_compose_paths],
                sandbox._docker_compose_env(),
            )
        )
        if command == ["config", "--format", "json"]:
            return SimpleNamespace(
                stdout='{"services": {"main": {"image": "fixture-main:latest", "build": {"context": "."}}}}',
                stderr="",
                return_code=0,
            )
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(sandbox, "_run_docker_compose_command", run_compose)
    monkeypatch.setattr(sandbox, "_prepare_log_dirs_after_up", AsyncMock())
    # Guards the 85b87bc9 merge: this async layer rig has no Compose config client.
    monkeypatch.setattr(sandbox, "freeze_launch", lambda *args: None)
    # This rig models layer selection/ownership, not daemon mount identities.
    monkeypatch.setattr(sandbox, "_pin_or_verify_bind_targets", AsyncMock())
    monkeypatch.setattr(sandbox, "_prepare_inert_bootstrap", AsyncMock())
    monkeypatch.setattr(sandbox, "_release_inert_bootstrap", AsyncMock())
    monkeypatch.setattr(sandbox, "_chown_to_host_user", AsyncMock())
    monkeypatch.setattr(sandbox, "_docker_cli", rig.cli)

    async def ensure(
        self, recipe, base_ref, *, alias=None, pull=False, prune=True, prune_keep=None
    ):
        rig.ensure_calls.append(
            {"recipe": recipe, "base": base_ref, "alias": alias, "pull": pull}
        )
        DockerLayers._claims[("dummy-daemon", KEY)] = (
            DockerLayers._claims.get(("dummy-daemon", KEY), 0) + 1
        )
        base = BaseImage(
            ref=base_ref,
            image_id="sha256:" + "1" * 64,
            digest="sha256:" + "2" * 64,
            arch="amd64",
            libc="glibc",
        )
        tag = f"benchflow-layer:agent-{KEY}"
        return LayerResult(
            daemon_id="dummy-daemon",
            kind=AGENT,
            key=KEY,
            tag=tag,
            image=alias or tag,
            status="hit",
            base=base,
            seconds=0.1,
        )

    monkeypatch.setattr(DockerLayers, "ensure_agent_layer", ensure)
    if layer:
        sandbox.use_agent_layer(RECIPE)
    return rig


async def test_main_starts_from_the_trials_tag_on_the_layer(tmp_path, monkeypatch):
    rig = _rig(tmp_path, monkeypatch)
    sandbox = rig.sandbox
    await sandbox.start(force_build=False)

    (call,) = rig.ensure_calls
    assert (
        call["base"] == sandbox._env_vars.main_image_name
    )  # the task image compose built
    repository, _, owner = call["alias"].partition(":")
    assert repository == f"bf__layer__{sandbox.compose_project_name}"
    # The tag names this process: a prune drops it once the process is gone.
    assert alias_owner_state(owner) == "this"
    assert call["pull"] is False
    assert rig.compose[0][0] == ["build"]
    _, files, env = rig.up()
    assert "docker-compose-prebuilt.yaml" in files
    assert "docker-compose-build.yaml" not in files
    assert env["PREBUILT_IMAGE_NAME"] == call["alias"]
    assert sandbox.layer_records[AGENT]["status"] == "hit"
    assert sandbox.layer_records[AGENT]["key"] == KEY

    await sandbox.stop(delete=True)
    removed = rig.removed()
    assert ["image", "rm", "--no-prune", call["alias"]] in removed
    assert not any(sandbox._env_vars.main_image_name in args for args in removed)
    assert not any(f"agent-{KEY}" in " ".join(args) for args in removed)
    assert ("dummy-daemon", KEY) not in DockerLayers._claims


async def test_image_env_skip_is_recorded_without_values(tmp_path, monkeypatch):
    """Guards residual M1 in 01727c09e: trials explain layer ineligibility."""
    from benchflow.sandbox.env_layers import LayerSkipped

    rig = _rig(tmp_path, monkeypatch)

    async def skip(*args, **kwargs):
        raise LayerSkipped("image env sets LD_LIBRARY_PATH, LD_PRELOAD")

    monkeypatch.setattr(DockerLayers, "ensure_agent_layer", skip)
    await rig.sandbox.start(force_build=False)
    assert rig.sandbox.agent_layer is None
    assert rig.sandbox.layer_records[AGENT] == {
        "status": "skipped",
        "reason": "image env sets LD_LIBRARY_PATH, LD_PRELOAD",
    }
    await rig.sandbox.stop(delete=True)


async def test_without_delete_only_the_trials_tag_goes(tmp_path, monkeypatch):
    rig = _rig(tmp_path, monkeypatch)
    await rig.sandbox.start(force_build=False)
    await rig.sandbox.stop(delete=False)
    alias = rig.ensure_calls[0]["alias"]
    assert alias.startswith(f"bf__layer__{rig.sandbox.compose_project_name}:")
    assert rig.removed() == [["image", "rm", "--no-prune", alias]]


async def test_a_failed_teardown_still_drops_the_trials_tag(tmp_path, monkeypatch):
    """compose down fails and so does the force-kill: the trial is over,
    and its tag and claim must not outlive it."""
    rig = _rig(tmp_path, monkeypatch)
    sandbox = rig.sandbox
    await sandbox.start(force_build=False)
    alias = rig.ensure_calls[0]["alias"]

    async def failing_compose(command, check=True, timeout_sec=None):
        raise RuntimeError("compose down hung")

    async def failing_kill():
        raise RuntimeError("daemon wedged")

    monkeypatch.setattr(sandbox, "_run_docker_compose_command", failing_compose)
    monkeypatch.setattr(sandbox, "_force_kill_project", failing_kill)
    with pytest.raises(RuntimeError, match="daemon wedged"):
        await sandbox.stop(delete=False)
    assert ["image", "rm", "--no-prune", alias] in rig.removed()
    assert ("dummy-daemon", KEY) not in DockerLayers._claims


async def test_a_tag_that_cannot_be_removed_is_logged(tmp_path, monkeypatch, caplog):
    rig = _rig(tmp_path, monkeypatch)
    await rig.sandbox.start(force_build=False)
    rig.cli.return_value = SimpleNamespace(
        stdout="",
        stderr="Error response from daemon: dial unix: refused",
        return_code=1,
    )
    with caplog.at_level("WARNING"):
        await rig.sandbox.stop(delete=False)
    assert any(
        "layer tag" in r.getMessage() and "refused" in r.getMessage()
        for r in caplog.records
    )


async def test_a_tag_compose_down_already_removed_is_not_a_warning(
    tmp_path, monkeypatch, caplog
):
    rig = _rig(tmp_path, monkeypatch)
    await rig.sandbox.start(force_build=False)
    rig.cli.return_value = SimpleNamespace(
        stdout="", stderr="Error response from daemon: No such image: x", return_code=1
    )
    with caplog.at_level("WARNING"):
        await rig.sandbox.stop(delete=True)
    assert not any("layer tag" in r.getMessage() for r in caplog.records)


async def test_kept_containers_keep_their_tag(tmp_path, monkeypatch):
    rig = _rig(tmp_path, monkeypatch)
    rig.sandbox._keep_containers = True
    await rig.sandbox.start(force_build=False)
    await rig.sandbox.stop(delete=True)
    assert rig.removed() == []


async def test_a_prebuilt_task_image_is_the_base_and_may_be_pulled(
    tmp_path, monkeypatch
):
    rig = _rig(tmp_path, monkeypatch, docker_image="ghcr.io/example/task:1")
    await rig.sandbox.start(force_build=False)
    assert rig.ensure_calls[0]["base"] == "ghcr.io/example/task:1"
    assert rig.ensure_calls[0]["pull"] is True
    assert ["build"] not in [command for command, _, _ in rig.compose]


async def test_no_recipe_starts_from_the_task_image_as_before(tmp_path, monkeypatch):
    rig = _rig(tmp_path, monkeypatch, layer=False)
    await rig.sandbox.start(force_build=False)
    assert not rig.ensure_calls
    assert "docker-compose-build.yaml" in rig.up()[1]
    assert rig.sandbox.layer_records == {}
    await rig.sandbox.stop(delete=True)
    assert rig.removed() == []


async def test_a_task_compose_that_sets_mains_image_installs_fresh(
    tmp_path, monkeypatch
):
    rig = _rig(
        tmp_path, monkeypatch, compose="services:\n  main:\n    image: custom:1\n"
    )
    await rig.sandbox.start(force_build=False)
    assert not rig.ensure_calls
    assert rig.sandbox.layer_records[AGENT]["status"] == "unsupported"


async def test_a_task_compose_with_sidecars_still_uses_the_layer(tmp_path, monkeypatch):
    rig = _rig(
        tmp_path, monkeypatch, compose="services:\n  db:\n    image: postgres:16\n"
    )
    await rig.sandbox.start(force_build=False)
    assert len(rig.ensure_calls) == 1
    assert rig.sandbox.layer_records[AGENT]["status"] == "hit"


@pytest.mark.parametrize(
    "main",
    [
        "tmpfs: [/opt/runtime/bin]",
        "volumes: ['agent-bin:/opt/runtime/bin']",
        "volumes_from: [seed]",
        "entrypoint: [sh, -c, 'echo changed > /opt/runtime/bin/gemini']",
        "command: [sh, -c, 'echo changed > /opt/runtime/bin/gemini']",
        "healthcheck: {test: [CMD-SHELL, 'echo changed > /opt/runtime/bin/gemini']}",
        "environment: {PATH: /custom/bin}",
        "extends: {file: extra.yaml, service: seed}",
    ],
    ids=[
        "tmpfs",
        "volume",
        "volumes-from",
        "entrypoint",
        "command",
        "healthcheck",
        "env",
        "extends",
    ],
)
async def test_main_runtime_overrides_install_in_trial(tmp_path, monkeypatch, main):
    """Guards U3c-CONFIRM-1 at 1c019ec9: startup can invalidate the stamp."""
    # Keep the runtime override being tested, with the definitions the launch
    # gate needs to resolve a valid project (7b4907c8/S1 merge).
    rig = _rig(
        tmp_path,
        monkeypatch,
        compose=f"services:\n  seed:\n    image: alpine:3.20\n  main:\n    {main}\n"
        "volumes:\n  agent-bin: {}\n",
    )
    (rig.sandbox.environment_dir / "extra.yaml").write_text(
        "services:\n  seed:\n    image: alpine:3.20\n"
    )
    await rig.sandbox.start(force_build=False)
    assert not rig.ensure_calls
    assert rig.sandbox.agent_layer is None
    assert "main" in rig.sandbox.layer_records[AGENT]["reason"]


async def test_compose_include_installs_in_trial(tmp_path, monkeypatch):
    """Guards U3c-CONFIRM-1: included services may change main too."""
    rig = _rig(tmp_path, monkeypatch, compose="include: [extra.yaml]\nservices: {}\n")
    (rig.sandbox.environment_dir / "extra.yaml").write_text("services: {}\n")
    await rig.sandbox.start(force_build=False)
    assert not rig.ensure_calls


async def test_sdk_mounts_install_in_trial(tmp_path, monkeypatch):
    """Guards U3c-CONFIRM-1: SDK mounts have the same effect as Compose."""
    rig = _rig(tmp_path, monkeypatch)
    rig.sandbox._mounts_json = [{"type": "volume", "target": "/opt/runtime/bin"}]
    await rig.sandbox.start(force_build=False)
    assert not rig.ensure_calls
    assert "mount" in rig.sandbox.layer_records[AGENT]["reason"]


async def test_a_failed_layer_is_recorded_and_the_trial_goes_on(tmp_path, monkeypatch):
    rig = _rig(tmp_path, monkeypatch)

    async def broken(self, *args, **kwargs):
        raise LayerError("building the fake-agent 1.0.0 layer failed:\nnpm ERR! 404")

    monkeypatch.setattr(DockerLayers, "ensure_agent_layer", broken)
    await rig.sandbox.start(force_build=False)
    record = rig.sandbox.layer_records[AGENT]
    assert record["status"] == "fresh" and "npm ERR! 404" in record["reason"]
    assert "docker-compose-build.yaml" in rig.up()[1]
    assert rig.sandbox.agent_layer is None


@pytest.mark.parametrize("sandbox_class", [DockerSandbox, RemoteDockerSandbox])
async def test_log_mount_refusal_records_a_live_install(
    tmp_path, monkeypatch, sandbox_class
):
    """Guards U3c-CONFIRM-1 at fd85c046 across both Docker backends."""
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_HOST", "ssh://dummy.invalid")
    rig = _rig(tmp_path, monkeypatch, sandbox_class=sandbox_class)

    async def refuse(self, *args, **kwargs):
        raise LayerError("runtime log mount would hide image contents: /logs/agent")

    monkeypatch.setattr(DockerLayers, "ensure_agent_layer", refuse)
    await rig.sandbox._start_from_layers()
    assert rig.sandbox.agent_layer is None
    assert not rig.sandbox._use_prebuilt
    assert rig.sandbox.layer_records[AGENT]["status"] == "fresh"
    assert "runtime log mount" in rig.sandbox.layer_records[AGENT]["reason"]


def test_the_trials_tag_is_a_valid_image_name():
    name = docker_module._sanitize_docker_image_name("bf__layer__Run-1_abc-0f0f0f0f")
    assert name == name.lower() and ":" not in name


async def test_a_second_start_builds_on_the_task_image_not_the_last_layer(
    tmp_path, monkeypatch
):
    rig = _rig(tmp_path, monkeypatch, docker_image="ghcr.io/example/task:1")
    await rig.sandbox.start(force_build=False)
    await rig.sandbox.stop(delete=False)
    await rig.sandbox.start(force_build=False)
    assert [c["base"] for c in rig.ensure_calls] == ["ghcr.io/example/task:1"] * 2


async def test_a_snapshot_of_a_layer_container_is_not_a_layer(tmp_path, monkeypatch):
    """`docker commit` copies the layer's labels onto the snapshot; blanked,
    no layer filter or libc lookup takes a kept checkpoint for the layer."""
    rig = _rig(tmp_path, monkeypatch)
    sandbox = rig.sandbox
    sandbox._main_container_id = AsyncMock(return_value="c0ffee")
    sandbox._bind_baseline = {"main": {}}

    async def no_credentials(ops):
        return []

    monkeypatch.setattr(docker_module, "scrub_credentials", no_credentials)
    calls: list[tuple[str, ...]] = []

    class Proc:
        returncode = 0

        async def communicate(self):
            return b"sha256:" + b"0" * 64, b""

    async def fake_exec(*args, **kwargs):
        calls.append(args)
        return Proc()

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    image = await sandbox.snapshot()

    (args,) = calls
    assert args[:2] == ("docker", "commit") and args[-1] == image.ref
    changes = [args[i + 1] for i, a in enumerate(args) if a == "--change"]
    for label in (
        "benchflow.layer",
        "benchflow.layer.key",
        "benchflow.layer.base",
        "benchflow.layer.libc",
        "benchflow.layer.agent",
        "benchflow.layer.agent-version",
    ):
        assert f"LABEL {label}=" in changes
    assert "LABEL com.docker.compose.project=" in changes
