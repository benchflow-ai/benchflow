"""Rollout-side tests for Bedrock proxy startup integration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.rollout import Role, Rollout, RolloutConfig, Scene


@pytest.mark.asyncio
async def test_trial_connect_starts_bedrock_runtime_before_connect_acp(tmp_path):
    cfg = RolloutConfig(
        task_path=tmp_path / "task",
        agent="codex-acp",
        model="aws-bedrock/openai.gpt-oss-20b-1:0",
        agent_env={
            "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
            "AWS_REGION": "us-east-1",
            "OPENAI_API_KEY": "bedrock-proxy",
        },
    )
    trial = Rollout.__new__(Rollout)
    trial._config = cfg
    trial._env = SimpleNamespace()
    trial._rollout_dir = tmp_path
    trial._timing = {}
    trial._agent_launch = "codex-acp"
    trial._agent_cwd = "/app"
    trial._agent_env = dict(cfg.agent_env)
    planes = MagicMock()
    planes.ensure_bedrock_proxy_runtime = AsyncMock(
        return_value=(
            {
                **cfg.agent_env,
                "OPENAI_BASE_URL": "http://host.docker.internal:32123",
                "BENCHFLOW_PROVIDER_BASE_URL": "http://host.docker.internal:32123",
            },
            object(),
        )
    )
    planes.ensure_usage_proxy_runtime = AsyncMock(
        side_effect=lambda **kwargs: (kwargs["agent_env"], None)
    )
    planes.connect_acp = AsyncMock(
        return_value=(AsyncMock(), AsyncMock(), AsyncMock(), "codex-acp")
    )
    trial._planes = planes

    await trial.connect()

    planes.ensure_bedrock_proxy_runtime.assert_awaited_once()
    assert (
        planes.ensure_bedrock_proxy_runtime.await_args.kwargs["environment"] == "docker"
    )
    assert (
        planes.connect_acp.await_args.kwargs["agent_env"]["OPENAI_BASE_URL"]
        == "http://host.docker.internal:32123"
    )


@pytest.mark.asyncio
async def test_trial_connect_as_starts_bedrock_runtime_for_role(tmp_path):
    role = Role(
        name="assistant",
        agent="claude-agent-acp",
        model="aws-bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
        env={
            "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
            "AWS_REGION": "us-east-1",
        },
    )
    cfg = RolloutConfig(
        task_path=tmp_path / "task",
        scenes=[Scene(roles=[role])],
        agent="codex-acp",
        model="gpt-4.1-mini",
    )
    trial = Rollout.__new__(Rollout)
    trial._config = cfg
    trial._env = SimpleNamespace()
    trial._rollout_dir = tmp_path
    trial._timing = {}
    trial._agent_cwd = "/app"
    trial._phase = "idle"
    trial._task = SimpleNamespace(
        config=SimpleNamespace(environment=SimpleNamespace(allow_internet=True))
    )
    planes = MagicMock()
    planes.agent_launch.return_value = "claude-agent-acp"
    planes.resolve_agent_env.return_value = {
        "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
        "AWS_REGION": "us-east-1",
        "ANTHROPIC_AUTH_TOKEN": "bedrock-proxy",
    }
    planes.ensure_bedrock_proxy_runtime = AsyncMock(
        return_value=(
            {
                "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
                "AWS_REGION": "us-east-1",
                "ANTHROPIC_AUTH_TOKEN": "bedrock-proxy",
                "ANTHROPIC_BASE_URL": "http://host.docker.internal:32123",
                "BENCHFLOW_PROVIDER_BASE_URL": "http://host.docker.internal:32123",
            },
            object(),
        )
    )
    planes.ensure_usage_proxy_runtime = AsyncMock(
        side_effect=lambda **kwargs: (kwargs["agent_env"], None)
    )
    planes.install_agent = AsyncMock(return_value=SimpleNamespace())
    planes.write_credential_files = AsyncMock()
    planes.upload_subscription_auth = AsyncMock()
    planes.apply_web_tool_policy = AsyncMock()
    planes.connect_acp = AsyncMock(
        return_value=(AsyncMock(), AsyncMock(), AsyncMock(), "claude-agent-acp")
    )
    trial._planes = planes

    await trial.connect_as(role)

    planes.ensure_bedrock_proxy_runtime.assert_awaited_once()
    assert (
        planes.ensure_bedrock_proxy_runtime.await_args.kwargs["environment"] == "docker"
    )
    assert (
        planes.connect_acp.await_args.kwargs["agent_env"]["ANTHROPIC_BASE_URL"]
        == "http://host.docker.internal:32123"
    )
