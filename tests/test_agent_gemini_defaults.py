"""Tests for the gemini agent's default model wiring (issue #343).

Guards against the silent fallback to ``DEFAULT_MODEL`` (a Claude model) when
a user runs ``--agent gemini`` without ``--model``. The fallback cross-wires
providers and previously demanded ``ANTHROPIC_API_KEY`` even though the user
selected the Gemini agent.
"""

import pytest

from benchflow.agents.registry import AGENTS, get_agent, infer_env_key_for_model
from benchflow.evaluation import effective_model


class TestGeminiHasDefaultModel:
    """The gemini AgentConfig must declare its own default model."""

    def test_gemini_default_model_non_empty(self):
        config, default_model = get_agent("gemini")
        assert config.name == "gemini"
        assert default_model, (
            "gemini AgentConfig must declare a default_model so --agent gemini "
            "without --model does not silently substitute a Claude default"
        )

    def test_gemini_default_model_is_gemini_family(self):
        _, default_model = get_agent("gemini")
        assert "gemini" in default_model.lower(), (
            f"default_model {default_model!r} must be a Gemini-family model"
        )

    def test_gemini_default_model_resolves_to_gemini_api_key(self):
        """The agent default must require GEMINI_API_KEY, not ANTHROPIC_API_KEY."""
        _, default_model = get_agent("gemini")
        assert infer_env_key_for_model(default_model) == "GEMINI_API_KEY"


class TestEffectiveModelHonorsAgentDefault:
    """effective_model must prefer the agent's own default_model over the
    global DEFAULT_MODEL when --model is omitted."""

    def test_no_model_uses_agent_default_model(self):
        _, default_model = get_agent("gemini")
        assert effective_model("gemini", None) == default_model

    def test_no_model_for_gemini_is_not_claude(self):
        """The user-visible bug from #343: --agent gemini → claude-haiku."""
        result = effective_model("gemini", None)
        assert result is not None
        assert "claude" not in result.lower(), (
            f"effective_model('gemini', None) returned {result!r} — must not "
            "fall back to a Claude model when the gemini agent is selected"
        )

    def test_explicit_model_overrides_agent_default(self):
        assert effective_model("gemini", "gemini-3.1-pro") == "gemini-3.1-pro"

    def test_default_agent_still_uses_global_default(self):
        """Backwards-compat: the DEFAULT_AGENT still falls back to DEFAULT_MODEL
        so `benchflow eval run` with no flags keeps working."""
        from benchflow.evaluation import DEFAULT_AGENT, DEFAULT_MODEL

        assert effective_model(DEFAULT_AGENT, None) == DEFAULT_MODEL


class TestGeminiAgentEnv:
    """The gemini agent must advertise an env requirement consistent with
    what its native CLI actually reads."""

    def test_gemini_requires_env_matches_cli_native(self):
        """Issue #342: ``agent show gemini`` previously advertised GOOGLE_API_KEY
        but the Gemini CLI reads GEMINI_API_KEY. The advertised key must be
        what the CLI consumes (the reverse alias is handled by env mirroring)."""
        cfg = AGENTS["gemini"]
        assert "GEMINI_API_KEY" in cfg.requires_env, (
            f"gemini requires_env={cfg.requires_env!r} should advertise "
            "GEMINI_API_KEY (what the CLI reads); GOOGLE_API_KEY is an alias"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
