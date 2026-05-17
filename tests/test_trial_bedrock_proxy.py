"""Trial-side tests for Bedrock proxy startup integration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.trial import Role, Scene, Trial, TrialConfig


@pytest.mark.asyncio
async def test_trial_connect_starts_bedrock_runtime_before_connect_acp(tmp_path):
    cfg = TrialConfig(
        task_path=tmp_path / "task",
        agent="codex-acp",
        model="aws-bedrock/openai.gpt-oss-20b-1:0",
        agent_env={
            "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
            "AWS_REGION": "us-east-1",
            "OPENAI_API_KEY": "bedrock-proxy",
        },
    )
    trial = Trial.__new__(Trial)
    trial._config = cfg
    trial._env = SimpleNamespace()
    trial._trial_dir = tmp_path
    trial._timing = {}
    trial._agent_launch = "codex-acp"
    trial._agent_cwd = "/app"
    trial._agent_env = dict(cfg.agent_env)

    with (
        patch(
            "benchflow.rollout.ensure_bedrock_proxy_runtime",
            new=AsyncMock(
                return_value=(
                    {
                        **cfg.agent_env,
                        "OPENAI_BASE_URL": "http://host.docker.internal:32123",
                        "BENCHFLOW_PROVIDER_BASE_URL": "http://host.docker.internal:32123",
                    },
                    object(),
                )
            ),
        ) as ensure_runtime,
        patch("benchflow.rollout.connect_acp", new_callable=AsyncMock) as connect_acp,
    ):
        connect_acp.return_value = (AsyncMock(), AsyncMock(), "codex-acp")
        await trial.connect()

    ensure_runtime.assert_awaited_once()
    assert ensure_runtime.await_args.kwargs["environment"] == "docker"
    assert (
        connect_acp.await_args.kwargs["agent_env"]["OPENAI_BASE_URL"]
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
    cfg = TrialConfig(
        task_path=tmp_path / "task",
        scenes=[Scene(roles=[role])],
        agent="codex-acp",
        model="gpt-4.1-mini",
    )
    trial = Trial.__new__(Trial)
    trial._config = cfg
    trial._env = SimpleNamespace()
    trial._trial_dir = tmp_path
    trial._timing = {}
    trial._agent_cwd = "/app"
    trial._phase = "idle"
    trial._task = SimpleNamespace(
        config=SimpleNamespace(environment=SimpleNamespace(allow_internet=True))
    )

    with (
        patch(
            "benchflow.rollout.resolve_agent_env",
            return_value={
                "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
                "AWS_REGION": "us-east-1",
                "ANTHROPIC_AUTH_TOKEN": "bedrock-proxy",
            },
        ),
        patch(
            "benchflow.rollout.ensure_bedrock_proxy_runtime",
            new=AsyncMock(
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
            ),
        ) as ensure_runtime,
        patch(
            "benchflow.rollout.install_agent",
            new=AsyncMock(return_value=SimpleNamespace()),
        ),
        patch("benchflow.rollout.write_credential_files", new=AsyncMock()),
        patch("benchflow.rollout.apply_web_tool_policy", new=AsyncMock()),
        patch("benchflow.rollout.connect_acp", new_callable=AsyncMock) as connect_acp,
    ):
        connect_acp.return_value = (AsyncMock(), AsyncMock(), "claude-agent-acp")
        await trial.connect_as(role)

    ensure_runtime.assert_awaited_once()
    assert ensure_runtime.await_args.kwargs["environment"] == "docker"
    assert (
        connect_acp.await_args.kwargs["agent_env"]["ANTHROPIC_BASE_URL"]
        == "http://host.docker.internal:32123"
    )
