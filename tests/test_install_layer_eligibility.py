"""Guards U3b-CONFIRM-1 against the Docker cache gap at 4a281fe6."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.agents.install import agent_layer_plan
from benchflow.rollout_planes import DefaultRolloutPlanes
from benchflow.sandbox.env_layers import AGENT, PROXY
from benchflow.task import Task
from tests.test_agent_install_split import StepSandbox, _layer_result
from tests.test_agent_install_split import _trial as agent_trial
from tests.test_proxy_layer_use import (
    _DockerLike,
    _installs,
    _LayeredSandbox,
    _plain_stamp,
    _trial,
)


def _task(tmp_path, addition=""):
    path = tmp_path / "fixture"
    shutil.copytree(
        Path(__file__).parent / "examples" / "hello-world-task",
        path,
        ignore=shutil.ignore_patterns("task.md"),
    )
    with (path / "task.toml").open("a") as handle:
        handle.write(addition)
    return Task(path)


def _mutate(trial, mutator):
    if mutator == "hook":
        trial._config.pre_agent_hooks = [AsyncMock()]
    elif mutator == "upload":
        trial._config.uploads = {"fixture": "/opt/runtime/bin/fixture"}
    elif mutator == "plane":
        trial._config.planes = DefaultRolloutPlanes()
    elif mutator == "manifest":
        from benchflow.environment.manifest import load_manifest

        trial._config.environment_manifest = load_manifest(
            Path(__file__).parent / "integration/deterministic/task/environment.toml"
        )
    elif mutator == "persistent-env":
        trial._env._persistent_env = {"npm_config_registry": "https://registry.invalid"}


ADDITIONS = {
    "setup": '\n[[environment.setup_commands]]\ncommand = "rm -f /opt/runtime/bin/opencode"\n',
    "healthcheck": '\n[environment.healthcheck]\ncommand = "true"\n',
    "task-env": '\n[environment.env]\nnpm_config_registry = "https://registry.invalid"\n',
}
MUTATORS = [*ADDITIONS, "hook", "upload", "plane", "manifest", "persistent-env"]


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("mutator", ["none", *MUTATORS])
def test_both_layers_follow_task_and_caller_install_inputs(tmp_path, backend, mutator):
    """Guards 4a281fe6's agent gap and f1f15096's shared proxy policy."""
    env = _DockerLike()
    trial = _trial(tmp_path, env, location="sandbox", environment=backend)
    trial._task = _task(tmp_path, ADDITIONS.get(mutator, ""))
    _mutate(trial, mutator)
    trial._request_layers()
    if mutator == "none":
        assert env.agent_recipe is not None and env.proxy_recipe is not None
        assert trial._agent_layer_plan is not None
    else:
        assert env.agent_recipe is None and env.proxy_recipe is not None
        assert trial._agent_layer_plan is None
        assert trial._layer_records[AGENT]["status"] == "fresh"
        assert trial._layer_records[AGENT]["reason"]
        assert PROXY not in trial._layer_records


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("mutator", MUTATORS)
async def test_agent_rechecks_eligibility_after_layer_selection(
    tmp_path, backend, mutator
):
    """Guards U3b-CONFIRM-1: an intact old stamp cannot skip a live install."""
    plan = agent_layer_plan("opencode")
    assert plan is not None
    env = StepSandbox(layer=_layer_result(plan), holds=plan.stamp)
    trial = agent_trial(tmp_path, "trial", "opencode", env, environment=backend)
    trial._task = _task(tmp_path)
    trial._request_agent_layer()
    assert trial._agent_layer_plan is not None
    if mutator in ADDITIONS:
        task_file = trial._task.paths.task_dir / "task.toml"
        with task_file.open("a") as handle:
            handle.write(ADDITIONS[mutator])
        trial._task = Task(task_file.parent)
    _mutate(trial, mutator)
    assert await trial._install_from_agent_layer(trial._rollout_dir) is False
    assert ("exec", f"cat {plan.stamp_path}") not in env.steps
    assert trial._layer_records[AGENT]["install"] == "in trial"


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("mutator", ["none", "hook", "persistent-env"])
@pytest.mark.parametrize("entrypoint", ["connect", "connect_as"])
async def test_proxy_connect_rechecks_eligibility_before_using_its_stamp(
    tmp_path, monkeypatch, backend, mutator, entrypoint
):
    """Guards 4502ff29's public input and f1f15096's proxy policy regression."""
    from benchflow.rollout import Role

    env = _LayeredSandbox(_plain_stamp())
    trial = _trial(tmp_path, env, location="sandbox", environment=backend)
    trial._task = _task(tmp_path)
    trial._agent_env = {"ANTHROPIC_API_KEY": "dummy-marker"}
    trial._disallow_web_tools = True
    _mutate(trial, mutator)
    captured = []

    class ProxyStarted(Exception):
        pass

    # Exercise the default plane and public provider boundary too: 4502ff29
    # added the bypass input only to the concrete runtime implementation.
    ensure_runtime = trial._planes.ensure_litellm_runtime

    async def start_then_stop(**kwargs):
        _, runtime = await ensure_runtime(**kwargs)
        captured.append(runtime.server)
        raise ProxyStarted

    monkeypatch.setattr(trial._planes, "ensure_litellm_runtime", start_then_stop)
    with pytest.raises(ProxyStarted):
        if entrypoint == "connect":
            await trial.connect()
        else:
            await trial.connect_as(
                Role(
                    name="agent",
                    agent=trial._config.primary_agent,
                    model=trial._config.primary_model,
                    env=trial._agent_env,
                )
            )
    assert len(_installs(env)) == 0
    assert captured[0].layer_use["install"] == "from layer"
