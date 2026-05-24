"""Defense-in-depth: never silently fall back to a cross-provider DEFAULT_MODEL.

When an agent has no ``default_model`` set and the user passes no ``--model``,
the historical behavior was to substitute ``DEFAULT_MODEL`` (a Claude haiku
model) regardless of which agent was selected. That cross-wired providers and
produced confusing ANTHROPIC_API_KEY errors for users running, e.g., the
gemini agent (#343).

These tests pin the new contract: such a configuration must raise a clear
"no default model" error instead of silently picking a Claude default.
"""

import pytest

from benchflow.agents.registry import AGENTS, AgentConfig
from benchflow.evaluation import DEFAULT_AGENT, DEFAULT_MODEL, effective_model


class TestNoCrossProviderFallback:
    def test_unknown_default_model_non_default_agent_raises(self, monkeypatch):
        """An agent with empty default_model that is not DEFAULT_AGENT must
        refuse to substitute the global default."""
        fake = AgentConfig(
            name="fake-no-default",
            install_cmd="true",
            launch_cmd="true",
            requires_env=[],
            default_model="",  # no default
        )
        monkeypatch.setitem(AGENTS, "fake-no-default", fake)

        with pytest.raises(ValueError, match="no default model"):
            effective_model("fake-no-default", None)

    def test_error_does_not_mention_claude_haiku_as_answer(self, monkeypatch):
        """The error must not silently suggest a cross-provider default — it
        names the offending fallback only as context, and tells the user to
        pass --model."""
        fake = AgentConfig(
            name="fake-no-default-2",
            install_cmd="true",
            launch_cmd="true",
            requires_env=[],
            default_model="",
        )
        monkeypatch.setitem(AGENTS, "fake-no-default-2", fake)

        with pytest.raises(ValueError) as excinfo:
            effective_model("fake-no-default-2", None)
        msg = str(excinfo.value)
        assert "--model" in msg
        assert "fake-no-default-2" in msg

    def test_empty_string_model_still_raises_for_non_default_agent(
        self, monkeypatch
    ):
        """Empty string == no model (legacy YAML shape) must also raise."""
        fake = AgentConfig(
            name="fake-no-default-3",
            install_cmd="true",
            launch_cmd="true",
            requires_env=[],
            default_model="",
        )
        monkeypatch.setitem(AGENTS, "fake-no-default-3", fake)

        with pytest.raises(ValueError, match="no default model"):
            effective_model("fake-no-default-3", "")

    def test_default_agent_with_no_model_still_works(self):
        """Backwards-compat: the DEFAULT_AGENT keeps falling back to
        DEFAULT_MODEL so `benchflow eval create` with no flags works."""
        assert effective_model(DEFAULT_AGENT, None) == DEFAULT_MODEL

    def test_oracle_short_circuits_before_default_check(self):
        """Oracle never gets a model regardless of default_model rules."""
        assert effective_model("oracle", None) is None

    def test_unknown_agent_falls_back_to_default(self):
        """Raw-command agents not in the registry stay on the legacy path so
        users can still run arbitrary commands as agents."""
        assert effective_model("totally-unregistered-agent", None) == DEFAULT_MODEL


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
