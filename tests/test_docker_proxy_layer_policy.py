"""Guards the Docker proxy reuse regression introduced by f1f15096."""

from __future__ import annotations

import pytest

from benchflow.rollout import Role
from benchflow.sandbox.env_layers import AGENT, PROXY
from tests.test_install_layer_eligibility import ADDITIONS, MUTATORS, _mutate, _task
from tests.test_proxy_layer_use import (
    _DockerLike,
    _installs,
    _LayeredSandbox,
    _plain_stamp,
    _trial,
)


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("mutator", ["none", *MUTATORS])
def test_proxy_selection_is_independent_of_agent_install_policy(
    tmp_path, backend, mutator
):
    """Guards f1f15096: provisioning must not disable the private proxy venv."""
    env = _DockerLike()
    trial = _trial(tmp_path, env, location="sandbox", environment=backend)
    trial._task = _task(tmp_path, ADDITIONS.get(mutator, ""))
    _mutate(trial, mutator)
    trial._request_layers()
    assert env.proxy_recipe.stamp == _plain_stamp()
    assert PROXY not in trial._layer_records
    if mutator == "none":
        assert env.agent_recipe is not None
    else:
        assert env.agent_recipe is None
        assert trial._layer_records[AGENT]["status"] == "fresh"


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona"])
@pytest.mark.parametrize("entrypoint", ["connect", "connect_as"])
@pytest.mark.parametrize("mutator", ["none", *MUTATORS, "fresh-flag", "fresh-env"])
async def test_current_proxy_policy_reaches_the_public_runtime(
    tmp_path, monkeypatch, backend, entrypoint, mutator
):
    """Guards f1f15096 at consumption, including late fresh-install requests."""
    env = _LayeredSandbox(_plain_stamp())
    trial = _trial(tmp_path, env, location="sandbox", environment=backend)
    trial._task = _task(tmp_path, ADDITIONS.get(mutator, ""))
    trial._agent_env = {"ANTHROPIC_API_KEY": "dummy-marker"}
    trial._disallow_web_tools = True
    _mutate(trial, mutator)
    if mutator == "fresh-flag":
        trial._config.fresh_install = True
    elif mutator == "fresh-env":
        monkeypatch.setenv("BENCHFLOW_FRESH_INSTALL", "1")
    started = []
    ensure_runtime = trial._planes.ensure_litellm_runtime

    class ProxyStarted(Exception):
        pass

    async def start_then_stop(**kwargs):
        _, runtime = await ensure_runtime(**kwargs)
        started.append(runtime.server)
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
    fresh = mutator in {"fresh-flag", "fresh-env"} or (
        backend == "daytona" and mutator != "none"
    )
    assert len(_installs(env)) == int(fresh)
    assert started[0].layer_use["install"] == ("in trial" if fresh else "from layer")


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("how", ["flag", "env"])
def test_fresh_proxy_selection_still_bypasses_both_layers(
    tmp_path, monkeypatch, backend, how
):
    """Guards f1f15096: separating policies must retain fresh-install priority."""
    env = _DockerLike()
    trial = _trial(tmp_path, env, location="sandbox", environment=backend)
    trial._task = _task(tmp_path)
    if how == "flag":
        trial._config.fresh_install = True
    else:
        monkeypatch.setenv("BENCHFLOW_FRESH_INSTALL", "1")
    trial._request_layers()
    assert env.proxy_recipe is None and env.agent_recipe is None
    for kind in (PROXY, AGENT):
        assert trial._layer_records[kind] == {
            "status": "fresh",
            "reason": "fresh install asked for",
        }
