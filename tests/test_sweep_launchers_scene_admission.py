"""Guards the missed role admission at 349f2ff3."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from benchflow._types import Role, Scene, Turn
from benchflow.agents import registry
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.sandbox.providers import PROVIDERS_BY_NAME
from benchflow.task.runtime_capabilities import UnsupportedTaskFeatureError
from tests.test_check_run_plan import _task


class _SandboxBoundary(Exception):
    pass


@pytest.fixture
def modal_without_live_transport(monkeypatch):
    """Modal as it used to be: a backend with no live transport.

    Modal now hosts ACP agents (ModalProcess); the role admission a
    backend without one needs stays guarded on its registry entry.
    """
    monkeypatch.setitem(
        PROVIDERS_BY_NAME,
        "modal",
        replace(PROVIDERS_BY_NAME["modal"], supports_live_process=False),
    )


@pytest.fixture
def factories(monkeypatch):
    for name, protocol, factory in [
        ("fixture-factory", "session-factory", "fixture:connect"),
        ("fixture-incomplete", "session-factory", ""),
        ("fixture-acp", "acp", "fixture:connect"),
    ]:
        monkeypatch.setitem(
            registry.AGENTS,
            name,
            registry.AgentConfig(
                name=name,
                install_cmd="",
                launch_cmd="",
                protocol=protocol,
                session_factory=factory,
            ),
        )
    monkeypatch.setitem(registry.AGENT_ALIASES, "fixture-alias", "fixture-factory")
    monkeypatch.setitem(registry.AGENT_ALIASES, "fixture-acp-alias", "codex-acp")


def _scenes(agent, layout):
    first = Scene.single(agent="fixture-factory")
    if layout == "later-scene":
        return [first, Scene(), Scene.single(agent=agent)]
    if layout == "same-scene":
        first.roles.append(Role(name="later", agent=agent))
        first.turns.extend([Turn(role="later"), Turn(role="agent")])
        return [first]
    # Execution resolves duplicate names to the last declaration, rather than
    # the primary role which setup and install_agent otherwise consult.
    first.roles.append(Role(name="agent", agent=agent))
    return [first]


def _rollout(tmp_path, scenes, *, sandbox="modal", harness="acp", loop=False):
    planes = MagicMock()
    planes.resolve_locked_paths.return_value = []
    planes.resolve_agent_env.return_value = {}
    planes.agent_launch.return_value = "fixture --agent"
    planes.create_environment.side_effect = _SandboxBoundary
    rollout = Rollout(
        RolloutConfig(
            task_path=_task(tmp_path / "task"),
            scenes=scenes,
            environment=sandbox,
            harness=harness,
            jobs_dir=tmp_path / "jobs",
            planes=planes,
            loop_strategy="verify-retry:k=3" if loop else None,
        )
    )
    return rollout, planes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent",
    [
        "codex-acp",
        "claude-agent-acp",
        "acp/codex-acp",
        "acp/claude-agent-acp",
        "acpx/codex-acp",
        "acpx/claude-agent-acp",
        "fixture-acp-alias",
        "fixture-incomplete",
        "fixture-acp",
        "custom-agent --stdio",
    ],
)
@pytest.mark.parametrize("layout", ["later-scene", "same-scene", "duplicate-name"])
@pytest.mark.parametrize("loop", [False, True])
async def test_all_selected_roles_refused_before_costly_setup(
    tmp_path, monkeypatch, factories, modal_without_live_transport, agent, layout, loop
):
    """Guards C5 at 349f2ff3 through setup, including both scene drivers."""
    rollout, planes = _rollout(tmp_path, _scenes(agent, layout), loop=loop)
    proxy = MagicMock(side_effect=AssertionError("proxy resolution reached"))
    monkeypatch.setattr(
        "benchflow.providers.litellm_runtime.proxy_install_record", proxy
    )
    with pytest.raises(UnsupportedTaskFeatureError, match=r"live.*transport"):
        await rollout.setup()
    proxy.assert_not_called()
    planes.resolve_agent_env.assert_not_called()
    planes.create_environment.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent", ["fixture-factory", "fixture-alias", "acp/fixture-factory"]
)
async def test_modal_factory_roles_remain_admitted(
    tmp_path, monkeypatch, factories, agent
):
    rollout, planes = _rollout(tmp_path, _scenes(agent, "later-scene"))
    # A declared companion without a turn is never connected by either driver.
    rollout._config.scenes[-1].roles.append(Role(name="unused", agent="codex-acp"))
    monkeypatch.setattr(
        "benchflow.providers.litellm_runtime.proxy_install_record", lambda **kw: None
    )
    with pytest.raises(_SandboxBoundary):
        await rollout.setup()
    planes.create_environment.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["oracle", "nop", "task-runtime"])
async def test_modal_noninteractive_selection_remains_admitted(
    tmp_path, monkeypatch, agent
):
    scenes = [Scene.single(agent=agent)]
    if agent in {"oracle", "nop"}:
        scenes.append(Scene.single(agent="codex-acp"))  # Scripted run ignores it.
    rollout, planes = _rollout(tmp_path, scenes)
    monkeypatch.setattr(
        "benchflow.providers.litellm_runtime.proxy_install_record", lambda **kw: None
    )
    with pytest.raises(_SandboxBoundary):
        await rollout.setup()
    planes.create_environment.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sandbox",
    ["docker", "remote-docker", "daytona", "modal", "apple-container", "agentcore"],
)
@pytest.mark.parametrize("harness", ["acp", "native"])
async def test_live_backends_admit_later_roles(tmp_path, monkeypatch, sandbox, harness):
    rollout, planes = _rollout(
        tmp_path,
        [Scene.single(agent="codex-acp"), Scene.single(agent="claude-agent-acp")],
        sandbox=sandbox,
        harness=harness,
    )
    monkeypatch.setattr(
        "benchflow.providers.litellm_runtime.proxy_install_record", lambda **kw: None
    )
    with pytest.raises(_SandboxBoundary):
        await rollout.setup()
    planes.create_environment.assert_called_once()
