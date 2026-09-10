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


def test_launch_config_rejects_alias_for_a_different_requested_model():
    """Guards PR #1076: stale proxy aliases cannot claim requested-model ownership."""
    agent_env = {
        "BENCHFLOW_PROVIDER_MODEL": "benchflow-openai-gpt-5.4-mini",
        "BENCHFLOW_LITELLM_MODEL_VIA_ENV": "1",
        "CODEX_CONFIG": '{"model":"benchflow-openai-gpt-5.4-mini"}',
    }

    updated_env, owns_model = apply_codex_launch_config(
        "codex-acp", agent_env, model="openai/gpt-5.5", reasoning_effort="high"
    )

    assert updated_env is agent_env
    assert not owns_model


@pytest.mark.parametrize("sandboxed", [True, False])
@pytest.mark.parametrize("requested_mode", [None, "read-only"])
def test_nested_sandbox_default_preserves_explicit_mode(sandboxed, requested_mode):
    """Guards PR #1118's Terra E2E failure: bwrap cannot start inside the sandbox."""
    env = {"INITIAL_AGENT_MODE": requested_mode} if requested_mode else {}
    result, owns_model = apply_codex_launch_config(
        "codex-acp", env, model=None, reasoning_effort=None, sandboxed=sandboxed
    )
    expected = requested_mode or ("agent-full-access" if sandboxed else None)
    assert result.get("INITIAL_AGENT_MODE") == expected
    assert env == ({"INITIAL_AGENT_MODE": requested_mode} if requested_mode else {})
    assert not owns_model


@pytest.mark.parametrize(
    "policy", ["BENCHFLOW_EGRESS_DENYLIST", "BENCHFLOW_DISALLOW_WEB_TOOLS"]
)
def test_web_policy_overrides_search_without_clobbering_provider_config(policy):
    """Guards PR #1118: supported config must disable provider-side search."""
    config = {
        "web_search": "live",
        "model_providers": {"custom": {"wire_api": "responses"}},
    }
    env = {policy: "1", "CODEX_CONFIG": json.dumps(config)}
    result, owns_model = apply_codex_launch_config(
        "codex-acp", env, model="custom/opaque-model", reasoning_effort=None
    )
    assert json.loads(result["CODEX_CONFIG"]) == {**config, "web_search": "disabled"}
    assert json.loads(env["CODEX_CONFIG"])["web_search"] == "live"
    assert not owns_model


@pytest.mark.parametrize("config", ["{", "[]"])
def test_web_policy_rejects_invalid_config_instead_of_leaving_search_enabled(config):
    """Guards PR #1118: malformed caller configuration cannot bypass web policy."""
    with pytest.raises(ValueError, match="CODEX_CONFIG"):
        apply_codex_launch_config(
            "codex-acp",
            {"CODEX_CONFIG": config, "BENCHFLOW_EGRESS_DENYLIST": "1"},
            model=None,
            reasoning_effort=None,
        )


def test_other_harness_keeps_its_native_configuration():
    """Guards PR #1118: Codex's nested sandbox setting belongs only to its adapter."""
    env = {"BENCHFLOW_EGRESS_DENYLIST": "1"}
    result, owns_model = apply_codex_launch_config(
        "custom", env, model=None, reasoning_effort=None, sandboxed=True
    )
    assert result is env
    assert not owns_model
