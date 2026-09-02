"""Codex launch configuration ownership tests."""

import json

import pytest

from benchflow.agents.codex_config import apply_codex_launch_config


@pytest.mark.parametrize(
    "raw_config",
    [None, "{", "[]", "{}", '{"model":"another-model"}'],
    ids=["missing", "malformed", "non-object", "missing-model", "mismatch"],
)
def test_launch_config_rejects_missing_invalid_or_mismatched_model(raw_config):
    """Guards PR #1076: only exact valid Codex config owns model selection."""
    agent_env = {
        "BENCHFLOW_PROVIDER_MODEL": "benchflow-openai-gpt-5.4-mini",
        "BENCHFLOW_LITELLM_MODEL_VIA_ENV": "1",
    }
    if raw_config is not None:
        agent_env["CODEX_CONFIG"] = raw_config

    updated_env, owns_model = apply_codex_launch_config(
        "codex-acp", agent_env, model="openai/gpt-5.4-mini", reasoning_effort="high"
    )

    assert updated_env is agent_env
    assert not owns_model


def test_launch_config_applies_effort_to_exact_model():
    """Guards PR #1076: launch-owned model carries requested effort."""
    agent_env = {
        "BENCHFLOW_PROVIDER_MODEL": "benchflow-openai-gpt-5.4-mini",
        "BENCHFLOW_LITELLM_MODEL_VIA_ENV": "1",
        "CODEX_CONFIG": '{"model":"benchflow-openai-gpt-5.4-mini"}',
    }

    updated_env, owns_model = apply_codex_launch_config(
        "codex-acp", agent_env, model="openai/gpt-5.4-mini", reasoning_effort="high"
    )

    assert owns_model
    assert json.loads(updated_env["CODEX_CONFIG"])["model_reasoning_effort"] == "high"
    assert updated_env is not agent_env
    assert "model_reasoning_effort" not in agent_env["CODEX_CONFIG"]
