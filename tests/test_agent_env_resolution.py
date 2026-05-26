"""Tests for gemini agent env auth resolution (issue #342).

Users may set either ``GEMINI_API_KEY`` (the CLI-native var) or
``GOOGLE_API_KEY`` (the public-docs alias). Both must satisfy auth, and
both must make it into the sandbox environment so the Gemini CLI itself
can read whichever variable it expects.

Without the bidirectional mirror, setting only GOOGLE_API_KEY passed the
host requires_env check but failed inside the sandbox with
``ACP error -32000: Gemini API key is missing or not configured.``
"""

from pathlib import Path

import pytest

from benchflow.agents.env import auto_inherit_env, resolve_agent_env


def _patch_no_subscription(monkeypatch, tmp_path):
    """Redirect ~/ lookups to an empty tmp_path so subscription-auth detection
    does not see real host files (otherwise it would mask the API-key path)."""
    orig = Path.expanduser

    def fake(self):
        s = str(self)
        if s.startswith("~"):
            return tmp_path / s[2:]
        return orig(self)

    monkeypatch.setattr(Path, "expanduser", fake)


def _clear_keys(monkeypatch):
    for k in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)


class TestAutoInheritEnvBidirectionalMirror:
    """auto_inherit_env mirrors GEMINI_API_KEY ↔ GOOGLE_API_KEY both ways."""

    def test_gemini_only_mirrors_to_google(self):
        env = {"GEMINI_API_KEY": "gk-test"}
        auto_inherit_env(env)
        assert env["GEMINI_API_KEY"] == "gk-test"
        assert env["GOOGLE_API_KEY"] == "gk-test"

    def test_google_only_mirrors_to_gemini(self):
        """Issue #342: setting GOOGLE_API_KEY alone must populate GEMINI_API_KEY
        so the in-sandbox Gemini CLI can authenticate."""
        env = {"GOOGLE_API_KEY": "gk-test"}
        auto_inherit_env(env)
        assert env["GOOGLE_API_KEY"] == "gk-test"
        assert env["GEMINI_API_KEY"] == "gk-test"

    def test_both_set_preserves_explicit_values(self):
        env = {"GEMINI_API_KEY": "gemini-val", "GOOGLE_API_KEY": "google-val"}
        auto_inherit_env(env)
        assert env["GEMINI_API_KEY"] == "gemini-val"
        assert env["GOOGLE_API_KEY"] == "google-val"


class TestResolveAgentEnvGemini:
    """End-to-end env resolution for the gemini agent with each auth shape."""

    def test_gemini_api_key_only_succeeds(self, monkeypatch, tmp_path):
        _clear_keys(monkeypatch)
        _patch_no_subscription(monkeypatch, tmp_path)
        monkeypatch.setenv("GEMINI_API_KEY", "gk-only-gemini")

        result = resolve_agent_env(
            agent="gemini", model="gemini-2.5-flash", agent_env=None
        )
        assert result["GEMINI_API_KEY"] == "gk-only-gemini"
        assert result["GOOGLE_API_KEY"] == "gk-only-gemini"

    def test_google_api_key_only_succeeds(self, monkeypatch, tmp_path):
        """Regression for #342: GOOGLE_API_KEY alone must satisfy auth and
        also populate GEMINI_API_KEY for the in-sandbox CLI."""
        _clear_keys(monkeypatch)
        _patch_no_subscription(monkeypatch, tmp_path)
        monkeypatch.setenv("GOOGLE_API_KEY", "gk-only-google")

        result = resolve_agent_env(
            agent="gemini", model="gemini-2.5-flash", agent_env=None
        )
        assert result["GOOGLE_API_KEY"] == "gk-only-google"
        assert result["GEMINI_API_KEY"] == "gk-only-google", (
            "GOOGLE_API_KEY alone must mirror to GEMINI_API_KEY so the "
            "Gemini CLI can authenticate inside the sandbox"
        )

    def test_neither_key_set_raises_clear_error(self, monkeypatch, tmp_path):
        _clear_keys(monkeypatch)
        _patch_no_subscription(monkeypatch, tmp_path)
        # also clear the dotenv path so it can't supply keys
        monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "no.env"))

        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            resolve_agent_env(agent="gemini", model="gemini-2.5-flash", agent_env=None)

    def test_explicit_agent_env_overrides_host(self, monkeypatch, tmp_path):
        """Explicit --agent-env GEMINI_API_KEY=... wins over the host env."""
        _clear_keys(monkeypatch)
        _patch_no_subscription(monkeypatch, tmp_path)
        monkeypatch.setenv("GOOGLE_API_KEY", "host-google")

        result = resolve_agent_env(
            agent="gemini",
            model="gemini-2.5-flash",
            agent_env={"GEMINI_API_KEY": "explicit"},
        )
        assert result["GEMINI_API_KEY"] == "explicit"


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
