from __future__ import annotations

import pytest

from benchflow.providers.litellm_bedrock_patch import (
    BEDROCK_ADAPTIVE_THINKING_RE,
    BEDROCK_THINKING_EFFORT_ENV,
)
from benchflow.providers.litellm_config import resolve_litellm_route


@pytest.mark.parametrize(
    "model",
    [
        "us.anthropic.claude-opus-4-8",
        "global.anthropic.claude-sonnet-4-9",
        "anthropic.claude-haiku-4-10",
    ],
)
def test_provider_patch_matcher_covers_bedrock_claude_4_8_plus(model):
    assert BEDROCK_ADAPTIVE_THINKING_RE.search(model)


@pytest.mark.parametrize(
    "model",
    [
        "us.anthropic.claude-opus-4-7",
        "claude-3-7-sonnet",
        "gemini-3.5-flash",
    ],
)
def test_provider_patch_matcher_rejects_older_or_non_claude_models(model):
    assert BEDROCK_ADAPTIVE_THINKING_RE.search(model) is None


def test_bedrock_thinking_effort_env_is_forwarded_into_litellm_env(monkeypatch):
    monkeypatch.setenv(BEDROCK_THINKING_EFFORT_ENV, "max")
    route = resolve_litellm_route(
        "aws-bedrock/us.anthropic.claude-opus-4-8",
        {"AWS_BEARER_TOKEN_BEDROCK": "token", "AWS_REGION": "us-west-2"},
    )

    assert route.upstream_model == "bedrock/us.anthropic.claude-opus-4-8"
    assert BEDROCK_THINKING_EFFORT_ENV == "BENCHFLOW_BEDROCK_THINKING_EFFORT"
