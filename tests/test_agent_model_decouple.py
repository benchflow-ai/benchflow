"""Tests for agent/model decoupling (issue #107)."""

import pytest
from benchflow.agents.registry import (
    AGENTS,
    _AGENT_ALIASES,
    get_agent,
    infer_env_key_for_model,
)


class TestAgentAliases:
    """openclaw-gemini alias resolves correctly."""

    def test_openclaw_gemini_alias_resolves(self):
        config, model = get_agent("openclaw-gemini")
        assert config.name == "openclaw"
        assert model == "google/gemini-3.1-flash-lite-preview"

    def test_openclaw_direct(self):
        config, model = get_agent("openclaw")
        assert config.name == "openclaw"
        assert model == ""

    def test_openclaw_no_hardcoded_requires_env(self):
        config, _ = get_agent("openclaw")
        assert config.requires_env == []

    def test_unknown_agent_raises(self):
        with pytest.raises(KeyError, match="Unknown agent"):
            get_agent("nonexistent-agent")

    def test_unknown_agent_error_lists_aliases(self):
        with pytest.raises(KeyError, match="openclaw-gemini"):
            get_agent("nonexistent-agent")


class TestInferEnvKey:
    """Model → API key inference."""

    def test_gemini_model(self):
        assert infer_env_key_for_model("gemini-3.1-pro") == "GEMINI_API_KEY"

    def test_gemini_with_provider_prefix(self):
        assert infer_env_key_for_model("google/gemini-3.1-flash-lite-preview") == "GEMINI_API_KEY"

    def test_claude_model(self):
        assert infer_env_key_for_model("claude-opus-4-6") == "ANTHROPIC_API_KEY"

    def test_haiku_model(self):
        assert infer_env_key_for_model("claude-haiku-4-5-20251001") == "ANTHROPIC_API_KEY"

    def test_sonnet_model(self):
        assert infer_env_key_for_model("claude-sonnet-4-6") == "ANTHROPIC_API_KEY"

    def test_gpt_model(self):
        assert infer_env_key_for_model("gpt-5.4") == "OPENAI_API_KEY"

    def test_o1_model(self):
        assert infer_env_key_for_model("o1-preview") == "OPENAI_API_KEY"

    def test_o3_model(self):
        assert infer_env_key_for_model("o3-mini") == "OPENAI_API_KEY"

    def test_unknown_model_returns_none(self):
        assert infer_env_key_for_model("some-custom-model") is None


class TestNoOpenclawGeminiEntry:
    """openclaw-gemini should not exist as a direct registry entry."""

    def test_no_direct_entry(self):
        assert "openclaw-gemini" not in AGENTS

    def test_is_alias(self):
        assert "openclaw-gemini" in _AGENT_ALIASES


class TestResultMetadata:
    """RunResult stores agent and model separately."""

    def test_run_result_has_model(self):
        from benchflow.sdk import RunResult
        r = RunResult(
            task_name="test",
            agent="openclaw",
            agent_name="openclaw-acp",
            model="google/gemini-3.1-pro",
        )
        assert r.agent == "openclaw"
        assert r.agent_name == "openclaw-acp"
        assert r.model == "google/gemini-3.1-pro"

    def test_run_result_defaults(self):
        from benchflow.sdk import RunResult
        r = RunResult(task_name="test")
        assert r.agent == ""
        assert r.model == ""
