"""Guards U3b-CONFIRM-2/3 against stale cache decisions at ce2bed56."""

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from benchflow.agents.install import agent_layer_plan
from benchflow.agents.registry import AGENT_INSTALLERS
from benchflow.native_harness.harnesses import NATIVE_HARNESSES
from benchflow.sandbox.env_layers import AGENT
from tests._receipt_fakes import receipt_aware
from tests.test_agent_install_split import StepSandbox, _layer_result, _trial
from tests.test_install_layer_eligibility import ADDITIONS, MUTATORS, _mutate, _task


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("change", ["none", "installer", "harness", "native-cli"])
async def test_docker_revalidates_the_current_agent_and_harness_plan(
    tmp_path, monkeypatch, backend, change
):
    """Guards U3b-CONFIRM-2: an old stamp cannot bless a new install plan."""
    agent = "opencode" if change in {"none", "installer"} else "codex-acp"
    harness = "native" if change == "native-cli" else "acp"
    plan = agent_layer_plan(agent, harness=harness)
    assert plan is not None
    env = StepSandbox(layer=_layer_result(plan), holds=plan.stamp)
    trial = _trial(tmp_path, "trial", agent, env, environment=backend, harness=harness)
    trial._request_agent_layer()
    if change == "installer":
        monkeypatch.setitem(AGENT_INSTALLERS, agent, "printf synthetic-new-version")
    elif change == "harness":
        trial._config.harness = "native"
    elif change == "native-cli":
        monkeypatch.setitem(
            NATIVE_HARNESSES,
            agent,
            replace(NATIVE_HARNESSES[agent], install_cmd="printf synthetic-new-cli"),
        )
    await trial.install_agent()
    if change == "none":
        assert ("exec", f"cat {plan.stamp_path}") in env.steps
        assert ("exec", plan.install) not in env.steps
        assert trial._layer_records[AGENT]["install"] == "from layer"
    else:
        assert ("exec", f"cat {plan.stamp_path}") not in env.steps
        assert ("exec", AGENT_INSTALLERS[agent]) in env.steps
        if trial._config.harness == "native":
            assert (
                "exec",
                NATIVE_HARNESSES[agent].install_cmd,
            ) in env.steps
        assert trial._layer_records[AGENT]["install"] == "in trial"


class DaytonaLayerSandbox(StepSandbox):
    use_agent_layer = None

    def __init__(self, layer):
        super().__init__()
        self.daytona_layer = layer
        self.launcher_present = True
        self.notes = []

    def cached_install_layer(self, kind, name):
        if (kind, name) == ("agent", self.daytona_layer.name):
            return self.daytona_layer, "synthetic-layer-key"
        return None

    def note_install_layer(self, kind, name, outcome):
        self.notes.append((kind, name, outcome))

    @receipt_aware
    async def exec(self, command, **kwargs):
        result = await super().exec(command, **kwargs)
        if command == f"cat {self.daytona_layer.marker}":
            result.return_code = 0
            result.stdout = "synthetic-layer-key\n"
        if command == AGENT_INSTALLERS["opencode"]:
            self.launcher_present = True
        return result


@pytest.mark.parametrize("change", ["none", "fresh-flag", "fresh-env", *MUTATORS])
async def test_daytona_rechecks_eligibility_through_the_default_install_plane(
    tmp_path, monkeypatch, change
):
    """Guards U3b-CONFIRM-3: current policy precedes Daytona's key lookup."""
    from benchflow.sandbox.daytona_layers import plan_install_layers
    from benchflow.task import Task

    monkeypatch.delenv("BENCHFLOW_FRESH_INSTALL", raising=False)
    plan = plan_install_layers(agent="opencode", model=None, agent_env={})
    assert plan.agent is not None
    env = DaytonaLayerSandbox(plan.agent)
    trial = _trial(tmp_path, "trial", "opencode", env, environment="daytona")
    trial._task = _task(tmp_path)
    trial._request_agent_layer()
    assert trial._agent_layer_plan is None
    if change == "fresh-flag":
        trial._config.fresh_install = True
    elif change == "fresh-env":
        monkeypatch.setenv("BENCHFLOW_FRESH_INSTALL", "1")
    elif change in ADDITIONS:
        task_file = trial._task.paths.task_dir / "task.toml"
        with task_file.open("a") as handle:
            handle.write(ADDITIONS[change])
        trial._task = Task(task_file.parent)
    _mutate(trial, change)
    if change != "none":
        env.launcher_present = False
    await trial.install_agent()
    assert env.launcher_present
    calls = [command for kind, command in env.steps if kind == "exec"]
    assert (AGENT_INSTALLERS["opencode"] in calls) is (change != "none")
    assert (f"cat {plan.agent.marker}" in calls) is (change == "none")
    if change != "none":
        assert env.notes and env.notes[0][2].startswith("not used:")


@pytest.mark.parametrize("fresh", [False, True])
async def test_daytona_role_install_obeys_current_policy(tmp_path, monkeypatch, fresh):
    """Guards U3b-CONFIRM-3 at the secondary connect_as install entry point."""
    from benchflow.rollout import Role
    from benchflow.sandbox.daytona_layers import plan_install_layers

    monkeypatch.delenv("BENCHFLOW_FRESH_INSTALL", raising=False)
    plan = plan_install_layers(agent="opencode", model=None, agent_env={})
    env = DaytonaLayerSandbox(plan.agent)
    trial = _trial(tmp_path, "role", "claude-agent-acp", env, environment="daytona")
    trial._agent_cwd = "/app"
    trial._config.fresh_install = fresh
    if fresh:
        env.launcher_present = False
    monkeypatch.setattr(
        trial._planes, "ensure_litellm_runtime", AsyncMock(return_value=({}, None))
    )

    class Connected(Exception):
        pass

    monkeypatch.setattr(trial._planes, "connect_acp", AsyncMock(side_effect=Connected))
    with pytest.raises(Connected):
        await trial.connect_as(Role(name="second", agent="opencode", model=None))
    assert env.launcher_present
    assert (("exec", AGENT_INSTALLERS["opencode"]) in env.steps) is fresh


async def test_custom_install_plane_keeps_its_calling_convention(tmp_path):
    """Guards ce2bed56's caller-owned install interface while adding policy."""
    from benchflow.agents.registry import AGENTS
    from benchflow.rollout_planes import DefaultRolloutPlanes

    calls = []

    class CustomPlane(DefaultRolloutPlanes):
        async def install_agent(
            self, env, agent, rollout_dir, *, sandbox_setup_timeout=120
        ):
            calls.append(agent)
            return AGENTS[agent]

    trial = _trial(tmp_path, "custom", "opencode", StepSandbox(), planes=CustomPlane())
    await trial.install_agent()
    assert calls == ["opencode"]
