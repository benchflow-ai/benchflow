"""A trial uses the model proxy's cached layer, or says why not (PY8 U2b).

- The in-sandbox proxy runs from the layer's venv when the sandbox started
  from a proxy layer whose stamp names this exact install (one command
  instead of the install); otherwise it installs in the trial as before.
- ``Rollout`` asks Docker for the layer only when the proxy runs in the
  sandbox, never under ``--fresh-install`` / ``BENCHFLOW_FRESH_INSTALL=1``,
  and records ``layers.proxy`` in config.json; Modal, Apple Container and
  AgentCore record ``unsupported`` and install in the trial.
- ``DockerSandbox`` builds the agent's layer on top of the proxy's.
- #1169: on a disk with too little room the fresh install stops before the
  agent with a message that names the shortfall.
- ``--fresh-install`` reaches every trial's RolloutConfig.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import benchflow as bf
from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers import proxy_requirements as reqs
from benchflow.providers.litellm_config import resolve_litellm_route
from benchflow.providers.litellm_runtime import (
    PROXY_INSTALL_FOOTPRINT_MB,
    _disk_check_command,
    proxy_layer_recipe,
)
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.sandbox.env_layers import (
    AGENT,
    FRESH_INSTALL_ENV,
    PROXY,
    PROXY_LAYER_STAMP,
    PROXY_LAYER_VENV,
    BaseImage,
    DockerLayers,
    LayerError,
    LayerResult,
    alias_owner_tag,
)
from tests._root_stand_in import run_sh
from tests.test_docker_agent_layer import RECIPE as AGENT_RECIPE
from tests.test_docker_agent_layer import _rig
from tests.test_litellm_hardening import _ExecResult, _FakeSandbox
from tests.test_uv_pin import _Box, _LocalShellSandbox, _point_proxy_install_at

needs_sh = pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")
BASE = BaseImage(
    ref="bf__task",
    image_id="sha256:" + "1" * 64,
    digest="sha256:" + "2" * 64,
    arch="amd64",
    libc="glibc",
)
VERTEX_ENV = {"GOOGLE_CLOUD_PROJECT": "p", "GOOGLE_CLOUD_LOCATION": "us-central1"}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(reqs.EXTRA_REQUIREMENTS_ENV, raising=False)
    monkeypatch.delenv(FRESH_INSTALL_ENV, raising=False)
    DockerLayers._claims.clear()
    reqs._CHECKED.clear()
    yield
    DockerLayers._claims.clear()
    reqs._CHECKED.clear()


def _layer(kind: str = PROXY, key: str = "d" * 64) -> LayerResult:
    tag = f"benchflow-layer:{kind}-{key}"
    return LayerResult(
        daemon_id="dummy-daemon",
        kind=kind,
        key=key,
        tag=tag,
        image=tag,
        status="hit",
        base=BASE,
        seconds=0.1,
    )


class _LayeredSandbox(_FakeSandbox):
    """A sandbox that started from a proxy layer whose stamp is ``stamp``."""

    def __init__(self, stamp: str | None, *, venv_ok: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.proxy_layer = _layer()
        self.stamp = stamp
        self.venv_ok = venv_ok

    async def exec(self, command: str, timeout_sec=None, user=None):
        if f"cat {PROXY_LAYER_STAMP}" in command:
            self.exec_calls.append(command)
            if self.stamp is None:
                return _ExecResult(1, stderr="No such file")
            return _ExecResult(0 if self.venv_ok else 1, stdout=self.stamp + "\n")
        return await super().exec(command, timeout_sec=timeout_sec, user=user)


def _minimax_route():
    return resolve_litellm_route(
        "minimax/MiniMax-M3",
        {"MINIMAX_API_KEY": "k", "MINIMAX_BASE_URL": "https://api.minimax.io/v1"},
    )


async def _start(sandbox, route=None):
    route = route or _minimax_route()
    return await runtime_mod._start_sandbox_litellm(
        sandbox=sandbox,
        route=route,
        master_key="sk-master",
        agent_env={
            "MINIMAX_API_KEY": "k",
            "MINIMAX_BASE_URL": "https://api.minimax.io/v1",
            "GOOGLE_APPLICATION_CREDENTIALS_JSON": "{}",
        },
        session_id="s",
        agent_name="openhands",
    )


def _installs(sandbox) -> list[str]:
    return [c for c in sandbox.exec_calls if "pip install" in c and "litellm" in c]


def _launch_config(sandbox) -> dict[str, Any]:
    path = next(p for p in sandbox.uploaded if p.endswith("/launch_config.json"))
    return json.loads(sandbox.uploaded[path])


def _plain_stamp() -> str:
    return proxy_layer_recipe(reqs.proxy_install(None, {})).stamp


def _vertex_stamp() -> str:
    return proxy_layer_recipe(reqs.proxy_install("google-vertex", {})).stamp


# -- the proxy in a trial ---------------------------------------------------


class TestProxyFromTheLayer:
    async def test_a_matching_layer_replaces_the_install_with_one_read(self):
        sandbox = _LayeredSandbox(_plain_stamp())
        process = await _start(sandbox)
        assert _installs(sandbox) == []
        reads = [c for c in sandbox.exec_calls if f"cat {PROXY_LAYER_STAMP}" in c]
        assert f"cat {PROXY_LAYER_STAMP}" in reads[0]
        # No requirement file goes into the sandbox; the proxy runs from the
        # layer's venv.
        assert not any(p.endswith("litellm-proxy.txt") for p in sandbox.uploaded)
        launch = _launch_config(sandbox)
        assert launch["python"] == f"{PROXY_LAYER_VENV}/bin/python"
        assert launch["litellm"] == f"{PROXY_LAYER_VENV}/bin/litellm"
        assert process.layer_use == {"install": "from layer"}

    async def test_a_layer_for_another_install_installs_in_the_trial(self):
        sandbox = _LayeredSandbox("sha256:" + "0" * 64)
        process = await _start(sandbox)
        assert len(_installs(sandbox)) == 1
        launch = _launch_config(sandbox)
        assert launch["litellm"].startswith(runtime_mod.LITELLM_SANDBOX_ROOT)
        assert process.layer_use is not None
        assert process.layer_use["install"] == "in trial"
        assert "holds no litellm-proxy venv" in process.layer_use["reason"]

    async def test_a_layer_whose_venv_is_gone_installs_in_the_trial(self):
        sandbox = _LayeredSandbox(_plain_stamp(), venv_ok=False)
        process = await _start(sandbox)
        assert len(_installs(sandbox)) == 1
        assert process.layer_use["install"] == "in trial"

    async def test_extras_need_a_layer_built_with_them(self, monkeypatch):
        monkeypatch.setenv(reqs.EXTRA_REQUIREMENTS_ENV, "humanize==4.11.0")
        monkeypatch.setattr(reqs, "_resolve", lambda extras, group, uv: None)
        monkeypatch.setattr(reqs, "_uv_executable", lambda: "uv")
        sandbox = _LayeredSandbox(_plain_stamp())
        process = await _start(sandbox)
        (install,) = _installs(sandbox)
        assert "-c " in install and "humanize==4.11.0" in install
        assert process.layer_use["install"] == "in trial"

    async def test_a_sandbox_without_a_proxy_layer_installs_as_before(self):
        sandbox = _FakeSandbox()
        process = await _start(sandbox)
        assert len(_installs(sandbox)) == 1
        assert not any(f"cat {PROXY_LAYER_STAMP}" in c for c in sandbox.exec_calls)
        assert process.layer_use is None

    async def test_a_vertex_route_runs_from_the_vertex_layer(self):
        route = resolve_litellm_route("google-vertex/gemini-2.5-flash", VERTEX_ENV)
        assert reqs.group_for_provider(route.provider_name) == reqs.VERTEX_GROUP
        sandbox = _LayeredSandbox(_vertex_stamp())
        process = await _start(sandbox, route=route)
        assert _installs(sandbox) == []
        assert process.layer_use == {"install": "from layer"}

    async def test_a_vertex_route_on_the_plain_layer_installs_the_vertex_set(self):
        route = resolve_litellm_route("google-vertex/gemini-2.5-flash", VERTEX_ENV)
        sandbox = _LayeredSandbox(_plain_stamp())
        process = await _start(sandbox, route=route)
        (install,) = _installs(sandbox)
        assert "litellm-proxy-vertex.txt" in install
        assert process.layer_use["install"] == "in trial"


# -- #1169: the fresh install on a small disk -------------------------------

_FAKE_DF = """#!/bin/sh
echo "Filesystem     1024-blocks  Used Available Capacity Mounted on"
echo "tmpfs          ${FAKE_DF_FREE_KB} 0 ${FAKE_DF_FREE_KB} 0% /tmp/runtime-litellm"
"""


def _with_df(box: _Box) -> _Box:
    df = box.bin / "df"
    df.write_text(_FAKE_DF)
    df.chmod(0o755)
    return box


@needs_sh
class TestDiskShortfall:
    def _run(self, command: str, free_kb: int | None, tmp_path: Path):
        path = tmp_path / "bin"
        path.mkdir(exist_ok=True)
        if free_kb is not None:
            (path / "df").write_text(_FAKE_DF)
            (path / "df").chmod(0o755)
        return run_sh(
            f"set -eu\n{command}\necho went-on",
            env={"PATH": str(path), "FAKE_DF_FREE_KB": str(free_kb or 0)},
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_too_little_room_stops_naming_the_shortfall(self, tmp_path):
        # What reasoning-gym-hard's 1 GiB leaves after its 809 MB image.
        done = self._run(_disk_check_command("/tmp/victim-a"), 215 * 1024, tmp_path)
        assert done.returncode == 28
        assert "went-on" not in done.stdout
        assert (
            f"needs about {PROXY_INSTALL_FOOTPRINT_MB} MB free under /tmp/victim-a; "
            f"the sandbox has 215 MB free there, "
            f"{PROXY_INSTALL_FOOTPRINT_MB - 215} MB short (#1169)"
        ) in done.stderr
        assert "storage_mb" in done.stderr and "--fresh-install" in done.stderr

    def test_enough_room_goes_on(self, tmp_path):
        free = (PROXY_INSTALL_FOOTPRINT_MB + 1) * 1024
        done = self._run(_disk_check_command("/tmp/victim-a"), free, tmp_path)
        assert done.returncode == 0 and "went-on" in done.stdout

    def test_without_df_the_install_tries(self, tmp_path):
        done = self._run(_disk_check_command("/tmp/victim-a"), None, tmp_path)
        assert done.returncode == 0 and "went-on" in done.stdout

    async def test_the_install_stops_before_uv_runs(self, tmp_path, monkeypatch):
        box = _with_df(_Box(tmp_path))
        venv = _point_proxy_install_at(box, monkeypatch)
        with pytest.raises(RuntimeError) as raised:
            await runtime_mod._ensure_sandbox_litellm(
                _LocalShellSandbox(box, FAKE_DF_FREE_KB=str(215 * 1024)),
                venv_dir=str(venv),
                requirements_path=f"{venv}.txt",
            )
        message = str(raised.value)
        assert "install LiteLLM in sandbox failed with exit code 28" in message
        assert "MB short (#1169)" in message
        assert box.downloads() == [] and box.uv_calls() == []
        assert not venv.exists()

    async def test_running_out_of_disk_mid_install_says_so(self):
        class Full(_FakeSandbox):
            async def exec(self, command, timeout_sec=None, user=None):
                if "pip install" in command:
                    return _ExecResult(
                        2, stderr="error: No space left on device (os error 28)"
                    )
                return await super().exec(command, timeout_sec, user)

        with pytest.raises(RuntimeError) as raised:
            await runtime_mod._ensure_sandbox_litellm(
                Full(), venv_dir="/tmp/victim-a/venv", requirements_path="/tmp/r.txt"
            )
        message = str(raised.value)
        assert message.startswith("the sandbox ran out of disk installing the model")
        assert "#1169" in message and "No space left on device" in message


# -- what Rollout asks for and records --------------------------------------


class _DockerLike:
    """A sandbox type that builds layers (has use_proxy_layer/use_agent_layer)."""

    def __init__(self) -> None:
        self.proxy_recipe: Any = "unset"
        self.agent_recipe: Any = "unset"
        self.layer_records: dict[str, dict[str, Any]] = {}

    def use_proxy_layer(self, recipe) -> None:
        self.proxy_recipe = recipe

    def use_agent_layer(self, recipe) -> None:
        self.agent_recipe = recipe


class _FakeModal:
    pass


class _FakeAppleContainer:
    pass


class _FakeAgentCore:
    pass


def _trial(tmp_path: Path, env: Any, *, location: str | None, **config) -> Rollout:
    task = tmp_path / "task"
    task.mkdir(parents=True, exist_ok=True)
    rollout = Rollout(
        RolloutConfig.from_legacy(
            task_path=task,
            agent="claude-agent-acp",
            model="anthropic/claude-haiku-4-5",
            jobs_dir=tmp_path / "jobs",
            **config,
        )
    )
    rollout._env = env
    rollout._rollout_dir = tmp_path / "trial"
    rollout._rollout_dir.mkdir(exist_ok=True)
    (rollout._rollout_dir / "config.json").write_text(json.dumps({"agent": "x"}))
    if location is not None:
        install = reqs.proxy_install(None, {})
        rollout._litellm_proxy_record = install.record(location=location)
    return rollout


def _config_layers(rollout: Rollout) -> dict[str, Any]:
    return json.loads((rollout._rollout_dir / "config.json").read_text()).get(
        "layers", {}
    )


class TestRolloutAsks:
    def test_a_sandbox_proxy_asks_docker_for_its_layer(self, tmp_path):
        env = _DockerLike()
        rollout = _trial(tmp_path, env, location="sandbox", environment="docker")
        rollout._request_layers()
        recipe = env.proxy_recipe
        assert recipe is not None and recipe != "unset"
        assert recipe.group == reqs.PROXY_GROUP
        assert recipe.stamp == _plain_stamp()
        assert PROXY not in rollout._layer_records

    def test_a_vertex_record_asks_for_the_vertex_layer(self, tmp_path):
        env = _DockerLike()
        rollout = _trial(tmp_path, env, location=None, environment="docker")
        rollout._litellm_proxy_record = reqs.proxy_install("google-vertex", {}).record(
            location="sandbox"
        )
        rollout._request_layers()
        assert env.proxy_recipe.group == reqs.VERTEX_GROUP
        assert env.proxy_recipe.stamp == _vertex_stamp()

    def test_extras_ask_for_no_proxy_layer_and_say_so(self, tmp_path):
        """An extra such as provider-sdk>=1 (or a pinned one with unpinned
        dependencies) resolves at install time: a layer built before a new
        release would keep the old one under the same key."""
        env = _DockerLike()
        rollout = _trial(tmp_path, env, location=None, environment="docker")
        install = reqs.proxy_install(
            None, {reqs.EXTRA_REQUIREMENTS_ENV: "provider-sdk>=1"}
        )
        rollout._litellm_proxy_record = install.record(location="sandbox")
        rollout._request_layers()
        rollout._record_layers()
        assert env.proxy_recipe is None
        assert _config_layers(rollout)[PROXY]["status"] == "fresh"
        assert "extras" in _config_layers(rollout)[PROXY]["reason"]

    def test_a_host_proxy_asks_for_no_proxy_layer(self, tmp_path):
        env = _DockerLike()
        rollout = _trial(tmp_path, env, location="host", environment="docker")
        rollout._request_layers()
        assert env.proxy_recipe is None
        assert PROXY not in rollout._layer_records

    @pytest.mark.parametrize("how", ["flag", "env"])
    def test_a_fresh_install_asks_for_no_layer_and_says_so(
        self, tmp_path, monkeypatch, how
    ):
        env = _DockerLike()
        config = {"environment": "docker"}
        if how == "env":
            monkeypatch.setenv(FRESH_INSTALL_ENV, "1")
        rollout = _trial(tmp_path, env, location="sandbox", **config)
        if how == "flag":
            rollout._config.fresh_install = True
        rollout._request_layers()
        rollout._record_layers()
        assert env.proxy_recipe is None and env.agent_recipe is None
        layers = _config_layers(rollout)
        assert layers[PROXY] == {"status": "fresh", "reason": "fresh install asked for"}
        assert layers[AGENT] == {"status": "fresh", "reason": "fresh install asked for"}

    @pytest.mark.parametrize(
        "backend, env_type",
        [
            ("modal", _FakeModal),
            ("apple-container", _FakeAppleContainer),
            ("agentcore", _FakeAgentCore),
        ],
    )
    async def test_backends_without_layers_record_unsupported_and_install_fresh(
        self, tmp_path, backend, env_type
    ):
        rollout = _trial(tmp_path, env_type(), location="sandbox", environment=backend)
        rollout._request_layers()
        rollout._record_layers()
        layers = _config_layers(rollout)
        for kind in (PROXY, AGENT):
            assert layers[kind]["status"] == "unsupported", kind
            assert backend in layers[kind]["reason"]
        # And the proxy installs in the trial there: no layer to read.
        sandbox = _FakeSandbox()
        process = await _start(sandbox)
        assert len(_installs(sandbox)) == 1 and process.layer_use is None

    def test_daytona_records_its_layers_elsewhere(self, tmp_path):
        rollout = _trial(
            tmp_path, _FakeModal(), location="sandbox", environment="daytona"
        )
        rollout._request_layers()
        assert rollout._layer_records == {}

    def test_the_proxys_use_of_its_layer_joins_the_sandbox_record(self, tmp_path):
        env = _DockerLike()
        env.layer_records[PROXY] = {"status": "hit", "key": "d" * 64}
        rollout = _trial(tmp_path, env, location="sandbox", environment="docker")
        rollout._usage_runtime = SimpleNamespace(
            server=SimpleNamespace(layer_use={"install": "from layer"})
        )
        rollout._record_proxy_layer_use()
        assert _config_layers(rollout)[PROXY] == {
            "status": "hit",
            "key": "d" * 64,
            "install": "from layer",
        }


# -- Docker: the agent's layer on the proxy's -------------------------------


def _patch_proxy(monkeypatch, rig, *, fail: bool = False) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    proxy_key = "e" * 64

    async def ensure_proxy(self, recipe, base_ref, *, alias=None, pull=False, **_):
        calls.append({"base": base_ref, "alias": alias, "pull": pull})
        if fail:
            raise LayerError("building the litellm-proxy proxy layer failed: boom")
        DockerLayers._claims[("dummy-daemon", proxy_key)] = (
            DockerLayers._claims.get(("dummy-daemon", proxy_key), 0) + 1
        )
        tag = f"benchflow-layer:proxy-{proxy_key}"
        return LayerResult(
            daemon_id="dummy-daemon",
            kind=PROXY,
            key=proxy_key,
            tag=tag,
            image=alias or tag,
            status="hit",
            base=BASE,
            seconds=0.1,
            as_base=BaseImage(
                ref=tag,
                image_id="sha256:" + "5" * 64,
                digest="sha256:" + "6" * 64,
                arch="amd64",
                libc="glibc",
            ),
        )

    async def ensure_agent(
        self, recipe, base_ref, *, alias=None, pull=False, base=None, **_
    ):
        rig.ensure_calls.append(
            {
                "recipe": recipe,
                "base": base_ref,
                "alias": alias,
                "pull": pull,
                "on": base,
            }
        )
        key = "c" * 64
        DockerLayers._claims[("dummy-daemon", key)] = (
            DockerLayers._claims.get(("dummy-daemon", key), 0) + 1
        )
        tag = f"benchflow-layer:agent-{key}"
        return LayerResult(
            daemon_id="dummy-daemon",
            kind=AGENT,
            key=key,
            tag=tag,
            image=alias or tag,
            status="hit",
            base=base or BASE,
            seconds=0.1,
        )

    monkeypatch.setattr(DockerLayers, "ensure_proxy_layer", ensure_proxy)
    monkeypatch.setattr(DockerLayers, "ensure_agent_layer", ensure_agent)
    return calls


RECIPE = proxy_layer_recipe(reqs.proxy_install(None, {}))


class TestDockerStacksTheLayers:
    async def test_the_agent_layer_is_built_on_the_proxy_layer(
        self, tmp_path, monkeypatch
    ):
        rig = _rig(tmp_path, monkeypatch)
        calls = _patch_proxy(monkeypatch, rig)
        sandbox = rig.sandbox
        sandbox.use_proxy_layer(RECIPE)
        await sandbox.start(force_build=False)

        # The trial's tags name this process (alias_owner_tag), so a prune
        # drops them if it dies before teardown.
        repository = f"bf__layer__{sandbox.compose_project_name}"
        alias = f"{repository}:{alias_owner_tag()}"
        proxy_alias = f"{repository}__proxy:{alias_owner_tag()}"
        (proxy,) = calls
        assert proxy == {
            "base": sandbox._env_vars.main_image_name,
            "alias": proxy_alias,
            "pull": False,
        }
        (agent,) = rig.ensure_calls
        assert agent["base"] == f"benchflow-layer:proxy-{'e' * 64}"
        assert agent["on"].digest == "sha256:" + "6" * 64
        assert agent["pull"] is False and agent["alias"] == alias
        assert rig.up()[2]["PREBUILT_IMAGE_NAME"] == alias
        assert sandbox.proxy_layer is not None and sandbox.agent_layer is not None
        assert sandbox.layer_records[PROXY]["status"] == "hit"
        assert sandbox.layer_records[AGENT]["status"] == "hit"

        await sandbox.stop(delete=True)
        removed = rig.removed()
        # Guards U10-CONFIRM-1/4 after 22672877: only private layer aliases go.
        assert ["image", "rm", "--no-prune", alias] in removed
        assert ["image", "rm", "--no-prune", proxy_alias] in removed
        assert not any(sandbox._env_vars.main_image_name in args for args in removed)
        assert not any("benchflow-layer:" in " ".join(args) for args in removed)
        assert DockerLayers._claims == {}

    async def test_the_proxy_layer_alone(self, tmp_path, monkeypatch):
        rig = _rig(tmp_path, monkeypatch, layer=False)
        calls = _patch_proxy(monkeypatch, rig)
        sandbox = rig.sandbox
        sandbox.use_proxy_layer(RECIPE)
        await sandbox.start(force_build=False)
        alias = f"bf__layer__{sandbox.compose_project_name}:{alias_owner_tag()}"
        assert calls[0]["alias"] == alias
        assert rig.ensure_calls == []
        assert rig.up()[2]["PREBUILT_IMAGE_NAME"] == alias
        assert AGENT not in sandbox.layer_records

    async def test_a_failed_proxy_layer_leaves_the_agent_layer_on_the_task_image(
        self, tmp_path, monkeypatch
    ):
        rig = _rig(tmp_path, monkeypatch)
        _patch_proxy(monkeypatch, rig, fail=True)
        sandbox = rig.sandbox
        sandbox.use_proxy_layer(RECIPE)
        await sandbox.start(force_build=False)
        (agent,) = rig.ensure_calls
        assert (
            agent["base"] == sandbox._env_vars.main_image_name and agent["on"] is None
        )
        record = sandbox.layer_records[PROXY]
        assert record["status"] == "fresh" and "layer failed" in record["reason"]
        assert sandbox.proxy_layer is None

    async def test_a_compose_that_sets_mains_image_records_both_unsupported(
        self, tmp_path, monkeypatch
    ):
        rig = _rig(
            tmp_path, monkeypatch, compose="services:\n  main:\n    image: custom:1\n"
        )
        calls = _patch_proxy(monkeypatch, rig)
        rig.sandbox.use_proxy_layer(RECIPE)
        await rig.sandbox.start(force_build=False)
        assert calls == [] and rig.ensure_calls == []
        for kind in (PROXY, AGENT):
            assert rig.sandbox.layer_records[kind]["status"] == "unsupported"


def test_the_agent_recipe_is_the_rigs():
    assert AGENT_RECIPE.agent == "fake-agent"


# -- --fresh-install reaches every trial -------------------------------------

TASK = Path(__file__).parent / "examples" / "hello-world-task"


def test_the_cli_flag_reaches_the_evaluation(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from benchflow.cli import main as cli_main

    seen: dict[str, Any] = {}

    async def fake_run(self):
        seen["config"] = self._config
        raise SystemExit(0)

    monkeypatch.setattr(bf.Evaluation, "run", fake_run)
    args = ["eval", "run", "--tasks-dir", str(TASK.parent), "--agent", "nop"]
    args += ["--jobs-dir", str(tmp_path / "jobs")]
    CliRunner().invoke(cli_main.app, [*args, "--fresh-install"])
    assert seen["config"].fresh_install is True
    CliRunner().invoke(cli_main.app, args)
    assert seen["config"].fresh_install is False


def test_the_cli_flag_overrides_a_config_file(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from benchflow.cli import main as cli_main

    seen: dict[str, Any] = {}

    async def fake_run(self):
        seen["config"] = self._config
        raise SystemExit(0)

    monkeypatch.setattr(bf.Evaluation, "run", fake_run)
    ev = bf.Evaluation(
        tasks_dir=TASK.parent,
        jobs_dir=tmp_path / "jobs",
        config=bf.EvaluationConfig(agent="nop"),
    )
    path = ev.to_yaml(tmp_path / "job.yaml")
    CliRunner().invoke(
        cli_main.app, ["eval", "run", "--config", str(path), "--fresh-install"]
    )
    assert seen["config"].fresh_install is True


def test_the_setting_survives_the_job_record_and_its_workers(tmp_path):
    from benchflow.eval_sharding import EvalShard, _config_payload
    from benchflow.eval_worker import _evaluation_config

    config = bf.EvaluationConfig(agent="nop", fresh_install=True)
    ev = bf.Evaluation(tasks_dir=TASK.parent, jobs_dir=tmp_path / "j", config=config)
    back = bf.Evaluation.from_dict(json.loads(json.dumps(ev.to_dict())))
    assert back._config.fresh_install is True
    payload = _config_payload(
        config, shard=EvalShard(index=0, task_names=("t",), concurrency=1)
    )
    assert _evaluation_config(json.loads(json.dumps(payload))).fresh_install is True


async def test_every_trial_of_the_evaluation_gets_it(tmp_path, monkeypatch):
    seen: dict[str, Any] = {}

    async def create(config):
        seen["config"] = config
        raise RuntimeError("stop here")

    monkeypatch.setattr(Rollout, "create", staticmethod(create))
    config = bf.EvaluationConfig(agent="nop", fresh_install=True)
    ev = bf.Evaluation(tasks_dir=TASK.parent, jobs_dir=tmp_path / "j", config=config)
    with pytest.raises(RuntimeError, match="stop here"):
        await ev._run_single_task(TASK, config)
    assert seen["config"].fresh_install is True


def test_the_rollout_config_file_keeps_it(tmp_path):
    config = bf.RolloutConfig(task_path=TASK, agent="nop", fresh_install=True)
    path = tmp_path / "rollout.yaml"
    config.to_yaml(path)
    assert bf.RolloutConfig.from_yaml(path).fresh_install is True


def test_an_unrelated_mock_sandbox_never_grows_layers():
    """Rollout looks use_proxy_layer up on the type: a MagicMock has none."""
    assert getattr(type(MagicMock()), "use_proxy_layer", None) is None


# -- the same through Rollout.start() ----------------------------------------


def _quiet_start(monkeypatch) -> None:
    """start() without a real sandbox: the env "starts" and nothing else runs."""
    import benchflow.rollout as rollout_module

    async def start_env(env, task_path, timing, *, on_started=None, **_):
        if on_started is not None:
            await on_started()

    async def nothing(*args, **kwargs):
        return None

    monkeypatch.setattr(rollout_module, "_start_env_and_upload", start_env)
    monkeypatch.setattr(rollout_module, "_run_environment_healthcheck", nothing)
    monkeypatch.setattr(rollout_module, "_run_environment_setup_commands", nothing)
    monkeypatch.setattr(rollout_module, "persist_sandbox_info", lambda *a, **k: None)


@pytest.mark.parametrize(
    "backend, env_type",
    [
        ("modal", _FakeModal),
        ("apple-container", _FakeAppleContainer),
        ("agentcore", _FakeAgentCore),
    ],
)
async def test_start_records_unsupported_layers_on_backends_without_them(
    tmp_path, monkeypatch, backend, env_type
):
    _quiet_start(monkeypatch)
    rollout = _trial(tmp_path, env_type(), location="sandbox", environment=backend)
    await rollout.start()
    layers = _config_layers(rollout)
    assert layers["proxy"]["status"] == "unsupported"
    assert layers["agent"]["status"] == "unsupported"


async def test_start_asks_docker_for_the_proxy_layer_and_records_fresh(
    tmp_path, monkeypatch
):
    _quiet_start(monkeypatch)
    env = _DockerLike()
    rollout = _trial(tmp_path, env, location="sandbox", environment="docker")
    await rollout.start()
    assert getattr(env.proxy_recipe, "group", None) == "litellm-proxy"

    monkeypatch.setenv(FRESH_INSTALL_ENV, "1")
    env = _DockerLike()
    rollout = _trial(tmp_path / "again", env, location="sandbox", environment="docker")
    await rollout.start()
    assert env.proxy_recipe is None
    assert _config_layers(rollout)["proxy"]["status"] == "fresh"
