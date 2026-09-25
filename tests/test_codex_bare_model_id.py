"""Guards the fix for #1145: Codex is handed the bare model slug, not the proxy alias.

Codex resolves model metadata (tool mode, ``apply_patch`` tool type,
multi-agent version) by slug. Given the proxy alias
``benchflow-azure-foundry-openai-gpt-5.6-luna`` it warns "Model metadata for
`...` not found. Defaulting to fallback metadata" and offers a reduced tool
surface: eight function tools, no native ``apply_patch``, no code mode, no
subagents. The LiteLLM proxy serves the bare slug next to the alias, so the
bare slug is what CODEX_CONFIG must name.
"""

from __future__ import annotations

import json

import pytest

from benchflow.agents import codex_config
from benchflow.agents import env as agents_env
from benchflow.agents.codex_config import CODEX_CONFIG_ENV
from benchflow.agents.registry import AGENTS
from benchflow.providers.litellm_config import safe_model_alias

MODEL = "azure-foundry-openai/gpt-5.6-luna"
ALIAS = safe_model_alias(MODEL)


def test_alias_carries_the_provider_prefix():
    """The value Codex used to receive: unknown to its model catalog."""
    assert ALIAS == "benchflow-azure-foundry-openai-gpt-5.6-luna"


def test_custom_provider_config_names_the_bare_slug(monkeypatch: pytest.MonkeyPatch):
    """env.py hands Codex ``gpt-5.6-luna`` while routing through the proxy provider."""
    monkeypatch.setattr(
        agents_env, "uses_native_subscription_auth", lambda *a, **k: False
    )
    agent_env = {
        "BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:4000/v1",
        "BENCHFLOW_PROVIDER_MODEL": ALIAS,
        "BENCHFLOW_PROVIDER_NAME": "azure-foundry-openai",
    }
    agents_env._configure_codex_custom_provider("codex-acp", MODEL, agent_env)
    config = json.loads(agent_env[CODEX_CONFIG_ENV])
    assert config["model"] == "gpt-5.6-luna"
    provider = config["model_providers"][config["model_provider"]]
    assert provider["base_url"] == "http://127.0.0.1:4000/v1"
    assert provider["wire_api"] == "responses"


def test_provider_config_writer_keeps_the_given_model():
    """apply_codex_provider_config writes whatever model it is given; callers pass the bare slug."""
    agent_env: dict[str, str] = {}
    codex_config.apply_codex_provider_config(
        agent_env,
        base_url="http://127.0.0.1:4000/v1",
        model="gpt-6-astra",
        provider_name="litellm",
    )
    assert json.loads(agent_env[CODEX_CONFIG_ENV])["model"] == "gpt-6-astra"


def test_launch_owns_model_with_the_bare_slug():
    """The reasoning effort still lands when CODEX_CONFIG names the bare slug."""
    agent_env = {
        CODEX_CONFIG_ENV: json.dumps(
            {"model": "gpt-5.6-luna", "model_provider": "benchflow-litellm"}
        ),
        "BENCHFLOW_PROVIDER_MODEL": ALIAS,
        "BENCHFLOW_LITELLM_MODEL_VIA_ENV": "1",
    }
    updated, owns = codex_config.apply_codex_launch_config(
        "codex-acp", agent_env, model=MODEL, reasoning_effort="xhigh"
    )
    assert owns is True
    config = json.loads(updated[CODEX_CONFIG_ENV])
    assert config["model"] == "gpt-5.6-luna"
    assert config["model_reasoning_effort"] == "xhigh"


def test_launch_owns_model_with_the_alias_too():
    """Configs written before this change (alias) keep working."""
    agent_env = {
        CODEX_CONFIG_ENV: json.dumps(
            {"model": ALIAS, "model_provider": "benchflow-litellm"}
        ),
        "BENCHFLOW_PROVIDER_MODEL": ALIAS,
        "BENCHFLOW_LITELLM_MODEL_VIA_ENV": "1",
    }
    _, owns = codex_config.apply_codex_launch_config(
        "codex-acp", agent_env, model=MODEL, reasoning_effort="xhigh"
    )
    assert owns is True


def test_launch_does_not_own_a_foreign_model():
    """A CODEX_CONFIG naming some other model is left alone."""
    agent_env = {
        CODEX_CONFIG_ENV: json.dumps({"model": "gpt-5.6-sol"}),
        "BENCHFLOW_PROVIDER_MODEL": ALIAS,
        "BENCHFLOW_LITELLM_MODEL_VIA_ENV": "1",
    }
    updated, owns = codex_config.apply_codex_launch_config(
        "codex-acp", agent_env, model=MODEL, reasoning_effort="xhigh"
    )
    assert owns is False
    assert "model_reasoning_effort" not in json.loads(updated[CODEX_CONFIG_ENV])


def test_codex_acp_pin_bundles_a_codex_that_knows_current_models():
    """codex-acp 1.13.1 bundles codex 0.156.1; 0.148 (1.6.0) has no metadata for gpt-6-astra."""
    assert "@agentclientprotocol/codex-acp@1.13.1" in AGENTS["codex-acp"].install_cmd
