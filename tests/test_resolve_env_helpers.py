"""Tests for extracted _resolve_agent_env helper methods.

Covers SDK._auto_inherit_env, _inject_vertex_credentials,
_resolve_provider_env, _check_subscription_auth, and the no-model
subscription auth path in _resolve_agent_env.
"""

from pathlib import Path

import pytest

from benchflow.sdk import SDK


# ── _auto_inherit_env ──


class TestAutoInheritEnv:
    """Tests for SDK._auto_inherit_env — host env key inheritance."""

    def test_inherits_anthropic_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-host")
        env = {}
        SDK._auto_inherit_env(env)
        assert env["ANTHROPIC_API_KEY"] == "sk-host"

    def test_inherits_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
        env = {}
        SDK._auto_inherit_env(env)
        assert env["OPENAI_API_KEY"] == "sk-oai"

    def test_does_not_overwrite_explicit(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-host")
        env = {"ANTHROPIC_API_KEY": "sk-explicit"}
        SDK._auto_inherit_env(env)
        assert env["ANTHROPIC_API_KEY"] == "sk-explicit"

    def test_inherits_provider_auth_env(self, monkeypatch):
        """Custom provider auth keys (e.g. ZAI_API_KEY) are inherited."""
        monkeypatch.setenv("ZAI_API_KEY", "zk-host")
        env = {}
        SDK._auto_inherit_env(env)
        assert env["ZAI_API_KEY"] == "zk-host"

    def test_gemini_mirrored_to_google(self):
        env = {"GEMINI_API_KEY": "gk-test"}
        SDK._auto_inherit_env(env)
        assert env["GOOGLE_API_KEY"] == "gk-test"

    def test_gemini_mirror_no_overwrite(self):
        env = {"GEMINI_API_KEY": "gk-test", "GOOGLE_API_KEY": "gk-explicit"}
        SDK._auto_inherit_env(env)
        assert env["GOOGLE_API_KEY"] == "gk-explicit"

    def test_missing_host_key_not_added(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        env = {}
        SDK._auto_inherit_env(env)
        assert "ANTHROPIC_API_KEY" not in env


# ── _inject_vertex_credentials ──


class TestInjectVertexCredentials:
    """Tests for SDK._inject_vertex_credentials — Vertex AI ADC setup."""

    def test_non_vertex_model_is_noop(self):
        env = {}
        SDK._inject_vertex_credentials(env, "claude-sonnet-4-6")
        assert "GOOGLE_APPLICATION_CREDENTIALS_JSON" not in env

    def test_missing_adc_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))
        with pytest.raises(ValueError, match="requires ADC credentials"):
            SDK._inject_vertex_credentials(
                {"GOOGLE_CLOUD_PROJECT": "proj"},
                "google-vertex/gemini-3-flash",
            )

    def test_missing_project_raises(self, monkeypatch, tmp_path):
        adc_dir = tmp_path / ".config" / "gcloud"
        adc_dir.mkdir(parents=True)
        (adc_dir / "application_default_credentials.json").write_text('{"ok": true}')
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT required"):
            SDK._inject_vertex_credentials({}, "google-vertex/gemini-3-flash")

    def test_success_injects_adc(self, monkeypatch, tmp_path):
        adc_dir = tmp_path / ".config" / "gcloud"
        adc_dir.mkdir(parents=True)
        (adc_dir / "application_default_credentials.json").write_text('{"key": "val"}')
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))
        env = {"GOOGLE_CLOUD_PROJECT": "my-proj"}
        SDK._inject_vertex_credentials(env, "google-vertex/gemini-3-flash")
        assert env["GOOGLE_APPLICATION_CREDENTIALS_JSON"] == '{"key": "val"}'
        assert env["GOOGLE_CLOUD_LOCATION"] == "global"


# ── _resolve_provider_env ──


class TestResolveProviderEnv:
    """Tests for SDK._resolve_provider_env — provider detection and env_mapping."""

    def test_sets_anthropic_model(self):
        env = {"ANTHROPIC_API_KEY": "sk-test"}
        SDK._resolve_provider_env(env, "claude-haiku-4-5-20251001", "claude-agent-acp")
        assert env["ANTHROPIC_MODEL"] == "claude-haiku-4-5-20251001"

    def test_strips_provider_prefix(self):
        env = {"ZAI_API_KEY": "zk-test"}
        SDK._resolve_provider_env(env, "zai/glm-5", "claude-agent-acp")
        assert env["ANTHROPIC_MODEL"] == "glm-5"

    def test_injects_benchflow_provider_vars(self):
        env = {"ZAI_API_KEY": "zk-test"}
        SDK._resolve_provider_env(env, "zai/glm-5", "claude-agent-acp")
        assert "BENCHFLOW_PROVIDER_NAME" in env
        assert "BENCHFLOW_PROVIDER_BASE_URL" in env
        assert "BENCHFLOW_PROVIDER_PROTOCOL" in env
        assert env["BENCHFLOW_PROVIDER_API_KEY"] == "zk-test"

    def test_env_mapping_applied(self):
        """claude-agent-acp maps BENCHFLOW_PROVIDER_* → agent-native vars."""
        env = {"ZAI_API_KEY": "zk-test"}
        SDK._resolve_provider_env(env, "zai/glm-5", "claude-agent-acp")
        assert "ANTHROPIC_BASE_URL" in env
        assert env["ANTHROPIC_AUTH_TOKEN"] == "zk-test"

    def test_no_provider_still_sets_model(self):
        """Model with no registered provider still sets ANTHROPIC_MODEL."""
        env = {"ANTHROPIC_API_KEY": "sk-test"}
        SDK._resolve_provider_env(env, "claude-sonnet-4-6", "claude-agent-acp")
        assert env["ANTHROPIC_MODEL"] == "claude-sonnet-4-6"


# ── _check_subscription_auth ──


class TestCheckSubscriptionAuth:
    """Tests for SDK._check_subscription_auth — host auth file detection."""

    def _patch_expanduser(self, monkeypatch, tmp_path):
        orig = Path.expanduser

        def fake(self):
            s = str(self)
            if s.startswith("~"):
                return tmp_path / s[2:]
            return orig(self)

        monkeypatch.setattr(Path, "expanduser", fake)

    def test_returns_true_when_file_exists(self, monkeypatch, tmp_path):
        creds_dir = tmp_path / ".claude"
        creds_dir.mkdir()
        (creds_dir / ".credentials.json").write_text("{}")
        self._patch_expanduser(monkeypatch, tmp_path)
        assert SDK._check_subscription_auth("claude-agent-acp", "ANTHROPIC_API_KEY") is True

    def test_returns_false_when_file_missing(self, monkeypatch, tmp_path):
        self._patch_expanduser(monkeypatch, tmp_path)
        assert SDK._check_subscription_auth("claude-agent-acp", "ANTHROPIC_API_KEY") is False

    def test_returns_false_for_wrong_key(self, monkeypatch, tmp_path):
        """subscription_auth.replaces_env must match required_key."""
        creds_dir = tmp_path / ".claude"
        creds_dir.mkdir()
        (creds_dir / ".credentials.json").write_text("{}")
        self._patch_expanduser(monkeypatch, tmp_path)
        # claude-agent-acp replaces ANTHROPIC_API_KEY, not OPENAI_API_KEY
        assert SDK._check_subscription_auth("claude-agent-acp", "OPENAI_API_KEY") is False

    def test_returns_false_for_unknown_agent(self):
        assert SDK._check_subscription_auth("nonexistent-agent", "ANTHROPIC_API_KEY") is False

    def test_returns_false_when_no_subscription_auth(self):
        """Agents without subscription_auth (e.g. openclaw) return False."""
        assert SDK._check_subscription_auth("openclaw", "ANTHROPIC_API_KEY") is False

    def test_codex_auth(self, monkeypatch, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text("{}")
        self._patch_expanduser(monkeypatch, tmp_path)
        assert SDK._check_subscription_auth("codex-acp", "OPENAI_API_KEY") is True


# ── _resolve_agent_env: no-model subscription auth ──


class TestResolveAgentEnvNoModel:
    """Tests for the no-model path in _resolve_agent_env."""

    def _patch_expanduser(self, monkeypatch, tmp_path):
        orig = Path.expanduser

        def fake(self):
            s = str(self)
            if s.startswith("~"):
                return tmp_path / s[2:]
            return orig(self)

        monkeypatch.setattr(Path, "expanduser", fake)

    def _resolve(self, agent="claude-agent-acp", model=None, agent_env=None):
        return SDK._resolve_agent_env(agent, model, agent_env)

    def test_no_model_with_api_key_works(self):
        """When API key is present and no model, no subscription auth needed."""
        result = self._resolve(agent_env={"ANTHROPIC_API_KEY": "sk-test"})
        assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result

    def test_no_model_subscription_auth_detected(self, monkeypatch, tmp_path):
        """No model + no API key + host credentials → subscription auth."""
        for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        creds_dir = tmp_path / ".claude"
        creds_dir.mkdir()
        (creds_dir / ".credentials.json").write_text('{"claudeAiOauth": {}}')
        self._patch_expanduser(monkeypatch, tmp_path)

        result = self._resolve(agent_env={})
        assert result["_BENCHFLOW_SUBSCRIPTION_AUTH"] == "1"

    def test_no_model_no_auth_no_error(self, monkeypatch, tmp_path):
        """No model + no API key + no host credentials → no error (no model to validate)."""
        for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        self._patch_expanduser(monkeypatch, tmp_path)
        # Should not raise — no model means no required key validation
        result = self._resolve(agent_env={})
        assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result

    def test_no_model_codex_subscription_auth(self, monkeypatch, tmp_path):
        """No model + codex agent + host auth file → subscription auth."""
        for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text("{}")
        self._patch_expanduser(monkeypatch, tmp_path)

        result = self._resolve(agent="codex-acp", agent_env={})
        assert result["_BENCHFLOW_SUBSCRIPTION_AUTH"] == "1"

    def test_no_model_empty_requires_env(self, monkeypatch, tmp_path):
        """Agent with empty requires_env (e.g. openclaw) needs no auth."""
        for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        self._patch_expanduser(monkeypatch, tmp_path)
        result = self._resolve(agent="openclaw", agent_env={})
        assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result
