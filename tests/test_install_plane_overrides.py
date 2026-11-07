"""Guards U3b-CONFIRM-4's install override regression from 3c036b77."""

from unittest.mock import AsyncMock

import pytest

from benchflow.agents.registry import AGENTS
from benchflow.contracts.planes import install_agent_with_policy
from benchflow.rollout import Role
from benchflow.rollout_planes import DefaultRolloutPlanes
from tests.test_agent_install_split import StepSandbox, _trial


@pytest.mark.parametrize("entrypoint", ["primary", "role"])
@pytest.mark.parametrize("override", ["subclass", "function", "method", "callable"])
async def test_custom_install_override_keeps_its_arguments(
    tmp_path, monkeypatch, entrypoint, override
):
    """Guards 3c036b77: invoke the selected installer's original interface."""
    calls = []

    async def custom_install(env, agent, rollout_dir, *, sandbox_setup_timeout=120):
        calls.append((env, agent, rollout_dir, sandbox_setup_timeout))
        return AGENTS[agent]

    class CustomPlane(DefaultRolloutPlanes):
        async def install_agent(
            self, env, agent, rollout_dir, *, sandbox_setup_timeout=120
        ):
            return await custom_install(
                env, agent, rollout_dir, sandbox_setup_timeout=sandbox_setup_timeout
            )

    class CustomInstaller:
        async def __call__(self, env, agent, rollout_dir, *, sandbox_setup_timeout=120):
            return await custom_install(
                env, agent, rollout_dir, sandbox_setup_timeout=sandbox_setup_timeout
            )

    planes = CustomPlane() if override == "subclass" else DefaultRolloutPlanes()
    if override != "subclass":
        installer = {
            "function": custom_install,
            "method": CustomPlane().install_agent,
            "callable": CustomInstaller(),
        }[override]
        monkeypatch.setattr(planes, "install_agent", installer)
    env = StepSandbox()
    trial = _trial(
        tmp_path,
        "custom",
        "opencode" if entrypoint == "primary" else "claude-agent-acp",
        env,
        planes=planes,
        sandbox_setup_timeout=37,
    )
    if entrypoint == "primary":
        await trial.install_agent()
        assert trial._agent_cfg is AGENTS["opencode"]
    else:
        trial._agent_cwd = "/app"
        monkeypatch.setattr(
            planes, "ensure_litellm_runtime", AsyncMock(return_value=({}, None))
        )

        class Connected(Exception):
            pass

        monkeypatch.setattr(planes, "connect_acp", AsyncMock(side_effect=Connected))
        with pytest.raises(Connected):
            await trial.connect_as(Role(name="second", agent="opencode", model=None))
    assert calls == [(env, "opencode", trial._rollout_dir, 37)]


@pytest.mark.parametrize("inherited", [False, True])
async def test_default_install_method_still_receives_policy(
    tmp_path, monkeypatch, inherited
):
    """Guards 3c036b77's policy forwarding for default and inherited methods."""

    class InheritedPlane(DefaultRolloutPlanes):
        pass

    planes = InheritedPlane() if inherited else DefaultRolloutPlanes()
    env = StepSandbox()
    install = AsyncMock(return_value=AGENTS["opencode"])
    monkeypatch.setattr("benchflow.rollout_planes.install_agent", install)
    result = await install_agent_with_policy(
        planes,
        env,
        "opencode",
        tmp_path,
        sandbox_setup_timeout=37,
        install_layer_bypass_reason="fresh install requested",
    )
    assert result is AGENTS["opencode"]
    install.assert_awaited_once_with(
        env,
        "opencode",
        tmp_path,
        sandbox_setup_timeout=37,
        install_layer_bypass_reason="fresh install requested",
    )
