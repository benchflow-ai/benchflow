from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.rollout import Role, Rollout, RolloutConfig


@pytest.mark.asyncio
async def test_trial_connect_starts_litellm_before_connect_acp(tmp_path: Path):
    """PR #919: forward required skill names before the ACP connection."""
    rollout = Rollout.__new__(Rollout)
    rollout._config = RolloutConfig(
        task_path=tmp_path / "task",
        agent="codex-acp",
        model="aws-bedrock/us.anthropic.claude-opus-4-8",
        environment="docker",
    )
    rollout._rollout_dir = tmp_path
    rollout._rollout_name = "rollout"
    rollout._agent_env = {
        "AWS_BEARER_TOKEN_BEDROCK": "token",
        "AWS_REGION": "us-west-2",
    }
    rollout._agent_launch = "codex-acp"
    rollout._agent_cwd = "/workspace"
    rollout._env = SimpleNamespace()
    rollout._usage_runtime = None
    rollout._required_skill_names = ("mesh-analysis",)
    rollout._timing = {}
    rollout._reapply_ask_user_handler = lambda: None
    rollout._attach_trajectory_writer = lambda _rollout_dir: None
    calls: list[str] = []

    async def fake_litellm(**kwargs):
        calls.append("litellm")
        assert kwargs["environment"] == "docker"
        assert kwargs["required_skill_names"] == ("mesh-analysis",)
        env = dict(kwargs["agent_env"])
        env["OPENAI_BASE_URL"] = "http://host.docker.internal:4000/v1"
        return env, SimpleNamespace(kind="litellm")

    async def fake_connect_acp(**kwargs):
        calls.append("acp")
        assert kwargs["agent_env"]["OPENAI_BASE_URL"] == (
            "http://host.docker.internal:4000/v1"
        )
        return (AsyncMock(), AsyncMock(), AsyncMock(), "codex-acp")

    rollout._planes = SimpleNamespace(
        ensure_litellm_runtime=fake_litellm,
        connect_acp=fake_connect_acp,
    )

    await rollout.connect()

    assert calls == ["litellm", "acp"]


@pytest.mark.asyncio
async def test_trial_connect_as_starts_litellm_for_role(tmp_path: Path):
    rollout = Rollout.__new__(Rollout)
    rollout._config = RolloutConfig(
        task_path=tmp_path / "task",
        agent="codex-acp",
        model="openai/gpt-4.1-mini",
        environment="docker",
        agent_env={"OPENAI_API_KEY": "sk-openai"},
    )
    rollout._rollout_dir = tmp_path
    rollout._rollout_name = "rollout"
    rollout._agent_cwd = "/workspace"
    rollout._env = SimpleNamespace()
    rollout._usage_runtime = None
    rollout._timing = {}
    rollout._disallow_web_tools = False
    rollout._agent_cfg = SimpleNamespace()
    rollout._reapply_ask_user_handler = lambda: None
    rollout._attach_trajectory_writer = lambda _rollout_dir: None
    role = Role(
        name="reviewer",
        agent="claude-agent-acp",
        model="claude-sonnet-4-6",
        env={"ANTHROPIC_API_KEY": "sk-ant"},
    )
    calls: list[str] = []

    async def fake_litellm(**kwargs):
        calls.append("litellm")
        assert kwargs["agent"] == "claude-agent-acp"
        env = dict(kwargs["agent_env"])
        env["ANTHROPIC_BASE_URL"] = "http://host.docker.internal:4000"
        return env, SimpleNamespace(kind="litellm")

    async def fake_connect_acp(**kwargs):
        calls.append("acp")
        assert kwargs["agent_env"]["ANTHROPIC_BASE_URL"] == (
            "http://host.docker.internal:4000"
        )
        return (AsyncMock(), AsyncMock(), AsyncMock(), "claude-agent-acp")

    rollout._planes = SimpleNamespace(
        agent_launch=lambda agent, disallow_web_tools: agent,
        resolve_agent_env=lambda agent, model, env: dict(env or {}),
        ensure_litellm_runtime=fake_litellm,
        install_agent=AsyncMock(return_value=SimpleNamespace()),
        write_credential_files=AsyncMock(),
        upload_subscription_auth=AsyncMock(),
        apply_web_tool_policy=AsyncMock(),
        connect_acp=fake_connect_acp,
    )

    await rollout.connect_as(role)

    assert calls == ["litellm", "acp"]
