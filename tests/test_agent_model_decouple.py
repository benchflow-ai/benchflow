"""Tests for agent/model decoupling (issue #107)."""

import pytest

from benchflow.agents.registry import (
    AGENTS,
    get_agent,
    infer_env_key_for_model,
    is_vertex_model,
)


class TestGetAgent:
    """get_agent resolves agents correctly."""

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

    def test_no_openclaw_gemini_entry(self):
        assert "openclaw-gemini" not in AGENTS


class TestIsVertexModel:
    """is_vertex_model — single source of truth for vertex prefixes."""

    def test_google_vertex(self):
        assert is_vertex_model("google-vertex/gemini-2.5-flash") is True

    def test_anthropic_vertex(self):
        assert is_vertex_model("anthropic-vertex/claude-sonnet-4-6") is True

    def test_plain_gemini_is_not_vertex(self):
        assert is_vertex_model("google/gemini-3.1-pro") is False

    def test_plain_claude_is_not_vertex(self):
        assert is_vertex_model("claude-sonnet-4-6") is False


class TestInferEnvKey:
    """Model → API key inference."""

    def test_gemini_model(self):
        assert infer_env_key_for_model("gemini-3.1-pro") == "GEMINI_API_KEY"

    def test_gemini_with_provider_prefix(self):
        assert (
            infer_env_key_for_model("google/gemini-3.1-flash-lite-preview")
            == "GEMINI_API_KEY"
        )

    def test_claude_model(self):
        assert infer_env_key_for_model("claude-opus-4-6") == "ANTHROPIC_API_KEY"

    def test_haiku_model(self):
        assert (
            infer_env_key_for_model("claude-haiku-4-5-20251001") == "ANTHROPIC_API_KEY"
        )

    def test_sonnet_model(self):
        assert infer_env_key_for_model("claude-sonnet-4-6") == "ANTHROPIC_API_KEY"

    def test_gpt_model(self):
        assert infer_env_key_for_model("gpt-5.4") == "OPENAI_API_KEY"

    def test_o1_model(self):
        assert infer_env_key_for_model("o1-preview") == "OPENAI_API_KEY"

    def test_o3_model(self):
        assert infer_env_key_for_model("o3-mini") == "OPENAI_API_KEY"

    def test_vertex_gemini_returns_none(self):
        """google-vertex/ models use ADC, not API keys."""
        assert infer_env_key_for_model("google-vertex/gemini-2.5-flash") is None

    def test_vertex_claude_returns_none(self):
        """anthropic-vertex/ models use ADC, not API keys."""
        assert infer_env_key_for_model("anthropic-vertex/claude-sonnet-4-6") is None

    def test_unknown_model_returns_none(self):
        assert infer_env_key_for_model("some-custom-model") is None

    @pytest.mark.parametrize(
        "model",
        [
            "google-vertex/any-model",
            "anthropic-vertex/any-model",
        ],
    )
    def test_infer_delegates_to_is_vertex(self, monkeypatch, model):
        """infer_env_key_for_model defers to is_vertex_model for ADC prefixes.

        Forces is_vertex_model → True so we test the delegation path, not
        the unrelated fallback that returns None for unknown prefixes.
        """
        from benchflow.agents import registry as registry_mod

        monkeypatch.setattr(registry_mod, "is_vertex_model", lambda m: True)
        assert registry_mod.infer_env_key_for_model(model) is None


class TestResultMetadata:
    """RunResult stores agent and model separately."""
