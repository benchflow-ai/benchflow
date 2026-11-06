"""Backend-independent writes into agent-controlled paths drop privilege first."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.agents.install import _link_skill_paths, apply_web_tool_policy
from benchflow.agents.registry import AGENTS


def environment():
    return SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(return_code=0, stdout="", stderr="")
        ),
        upload_file=AsyncMock(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "discovery", sorted({p for cfg in AGENTS.values() for p in cfg.skill_paths})
)
async def test_skill_destinations_are_written_as_agent(discovery):
    env = environment()
    await _link_skill_paths(
        env, "/skills", [discovery], "/tmp/agent-home", "/tmp/workspace", "agent"
    )
    assert env.exec.await_args.kwargs.get("user") == "agent"
    assert "chown" not in env.exec.await_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent", [name for name, cfg in AGENTS.items() if cfg.disallow_web_tools_setup_cmd]
)
async def test_policy_destinations_are_written_as_agent(agent):
    env = environment()
    # This is an account home string passed to the backend, never a host path.
    await apply_web_tool_policy(env, agent, AGENTS[agent], "/home/agent", disallow=True)
    assert env.exec.await_args.kwargs.get("user") == "agent"
    assert "chown" not in env.exec.await_args.args[0]
