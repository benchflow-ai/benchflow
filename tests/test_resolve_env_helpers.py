"""Tests for extracted resolve_agent_env helper functions.

Covers auto_inherit_env, inject_vertex_credentials, resolve_provider_env,
check_subscription_auth, and the no-model subscription auth path in
resolve_agent_env.
"""

from pathlib import Path

import pytest

from benchflow._agent_env import (
    auto_inherit_env,
    check_subscription_auth,
    inject_vertex_credentials,
    resolve_agent_env,
    resolve_provider_env,
)

# ── auto_inherit_env ──


class TestAutoInheritEnv:
    """Tests for auto_inherit_env — host env key inheritance."""

    @pytest.mark.parametrize(
        ("env_name", "env_value"),
        [
            pytest.param("ANTHROPIC_API_KEY", "sk-host", id="anthropic"),
            pytest.param("OPENAI_API_KEY", "sk-oai", id="openai"),
            pytest.param("ZAI_API_KEY", "zk-host", id="provider"),
        ],
    )
    def test_inherits_key(self, monkeypatch, env_name, env_value):
        monkeypatch.setenv(env_name, env_value)
        env = {}
        auto_inherit_env(env)
        assert env[env_name] == env_value

    def test_does_not_overwrite_explicit(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-host")
        env = {"ANTHROPIC_API_KEY": "sk-explicit"}
        auto_inherit_env(env)
        assert env["ANTHROPIC_API_KEY"] == "sk-explicit"

    def test_gemini_mirrored_to_google(self):
        env = {"GEMINI_API_KEY": "gk-test"}
        auto_inherit_env(env)
        assert env["GOOGLE_API_KEY"] == "gk-test"

    def test_gemini_mirror_no_overwrite(self):
        env = {"GEMINI_API_KEY": "gk-test", "GOOGLE_API_KEY": "gk-explicit"}
        auto_inherit_env(env)
        assert env["GOOGLE_API_KEY"] == "gk-explicit"

    def test_missing_host_key_not_added(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        env = {}
        auto_inherit_env(env)
        assert "ANTHROPIC_API_KEY" not in env


# ── inject_vertex_credentials ──


class TestInjectVertexCredentials:
    """Tests for inject_vertex_credentials — Vertex AI ADC setup."""

    def test_non_vertex_model_is_noop(self):
        env = {}
        inject_vertex_credentials(env, "claude-sonnet-4-6")
        assert "GOOGLE_APPLICATION_CREDENTIALS_JSON" not in env

    def test_missing_adc_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))
        with pytest.raises(ValueError, match="requires ADC credentials"):
            inject_vertex_credentials(
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
            inject_vertex_credentials({}, "google-vertex/gemini-3-flash")

    def test_success_injects_adc(self, monkeypatch, tmp_path):
        adc_dir = tmp_path / ".config" / "gcloud"
        adc_dir.mkdir(parents=True)
        (adc_dir / "application_default_credentials.json").write_text('{"key": "val"}')
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))
        env = {"GOOGLE_CLOUD_PROJECT": "my-proj"}
        inject_vertex_credentials(env, "google-vertex/gemini-3-flash")
        assert env["GOOGLE_APPLICATION_CREDENTIALS_JSON"] == '{"key": "val"}'
        assert env["GOOGLE_CLOUD_LOCATION"] == "global"


# ── resolve_provider_env ──


class TestResolveProviderEnv:
    """Tests for resolve_provider_env — provider detection and env_mapping."""

    def test_sets_provider_model(self):
        env = {"ANTHROPIC_API_KEY": "sk-test"}
        resolve_provider_env(env, "claude-haiku-4-5-20251001", "claude-agent-acp")
        assert env["BENCHFLOW_PROVIDER_MODEL"] == "claude-haiku-4-5-20251001"
        # env_mapping translates to agent-native var
        assert env["ANTHROPIC_MODEL"] == "claude-haiku-4-5-20251001"

    def test_strips_provider_prefix(self):
        env = {"ZAI_API_KEY": "zk-test"}
        resolve_provider_env(env, "zai/glm-5", "claude-agent-acp")
        assert env["BENCHFLOW_PROVIDER_MODEL"] == "glm-5"
        assert env["ANTHROPIC_MODEL"] == "glm-5"

    def test_injects_benchflow_provider_vars(self):
        env = {"ZAI_API_KEY": "zk-test"}
        resolve_provider_env(env, "zai/glm-5", "claude-agent-acp")
        assert env["BENCHFLOW_PROVIDER_NAME"] == "zai"
        assert "BENCHFLOW_PROVIDER_BASE_URL" in env
        assert "BENCHFLOW_PROVIDER_PROTOCOL" in env
        assert env["BENCHFLOW_PROVIDER_API_KEY"] == "zk-test"

    def test_env_mapping_applied(self):
        """claude-agent-acp maps BENCHFLOW_PROVIDER_* → agent-native vars."""
        env = {"ZAI_API_KEY": "zk-test"}
        resolve_provider_env(env, "zai/glm-5", "claude-agent-acp")
        assert "ANTHROPIC_BASE_URL" in env
        assert env["ANTHROPIC_AUTH_TOKEN"] == "zk-test"

    def test_zai_picks_anthropic_endpoint_for_claude_agent(self):
        """claude-agent-acp speaks anthropic-messages → routes to zai's anthropic endpoint."""
        env = {"ZAI_API_KEY": "zk-test"}
        resolve_provider_env(env, "zai/glm-5", "claude-agent-acp")
        assert env["BENCHFLOW_PROVIDER_BASE_URL"] == "https://api.z.ai/api/anthropic"
        assert env["BENCHFLOW_PROVIDER_PROTOCOL"] == "anthropic-messages"
        # env_mapping translates to ANTHROPIC_BASE_URL
        assert env["ANTHROPIC_BASE_URL"] == "https://api.z.ai/api/anthropic"

    def test_zai_picks_openai_endpoint_for_codex_agent(self):
        """codex-acp speaks openai-completions → routes to zai's openai endpoint."""
        env = {"ZAI_API_KEY": "zk-test"}
        resolve_provider_env(env, "zai/glm-5", "codex-acp")
        assert env["BENCHFLOW_PROVIDER_BASE_URL"] == "https://api.z.ai/api/paas/v4"
        assert env["BENCHFLOW_PROVIDER_PROTOCOL"] == "openai-completions"
        assert env["OPENAI_BASE_URL"] == "https://api.z.ai/api/paas/v4"

    def test_explicit_base_url_not_overwritten(self):
        """User-supplied ANTHROPIC_BASE_URL must win over derived value."""
        env = {
            "ZAI_API_KEY": "zk-test",
            "ANTHROPIC_BASE_URL": "https://custom.example/anthropic",
        }
        resolve_provider_env(env, "zai/glm-5", "claude-agent-acp")
        assert env["ANTHROPIC_BASE_URL"] == "https://custom.example/anthropic"

    def test_no_provider_still_sets_model(self):
        """Model with no registered provider still sets BENCHFLOW_PROVIDER_MODEL."""
        env = {"ANTHROPIC_API_KEY": "sk-test"}
        resolve_provider_env(env, "claude-sonnet-4-6", "claude-agent-acp")
        assert env["BENCHFLOW_PROVIDER_MODEL"] == "claude-sonnet-4-6"
        assert env["ANTHROPIC_MODEL"] == "claude-sonnet-4-6"


# ── check_subscription_auth ──


class TestCheckSubscriptionAuth:
    """Tests for check_subscription_auth — host auth file detection."""

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
        assert check_subscription_auth("claude-agent-acp", "ANTHROPIC_API_KEY") is True

    def test_returns_false_when_file_missing(self, monkeypatch, tmp_path):
        self._patch_expanduser(monkeypatch, tmp_path)
        assert check_subscription_auth("claude-agent-acp", "ANTHROPIC_API_KEY") is False

    def test_returns_false_for_wrong_key(self, monkeypatch, tmp_path):
        """subscription_auth.replaces_env must match required_key."""
        creds_dir = tmp_path / ".claude"
        creds_dir.mkdir()
        (creds_dir / ".credentials.json").write_text("{}")
        self._patch_expanduser(monkeypatch, tmp_path)
        # claude-agent-acp replaces ANTHROPIC_API_KEY, not OPENAI_API_KEY
        assert check_subscription_auth("claude-agent-acp", "OPENAI_API_KEY") is False

    def test_returns_false_for_unknown_agent(self):
        assert (
            check_subscription_auth("nonexistent-agent", "ANTHROPIC_API_KEY") is False
        )

    def test_returns_false_when_no_subscription_auth(self):
        """Agents without subscription_auth (e.g. openclaw) return False."""
        assert check_subscription_auth("openclaw", "ANTHROPIC_API_KEY") is False

    def test_codex_auth(self, monkeypatch, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text("{}")
        self._patch_expanduser(monkeypatch, tmp_path)
        assert check_subscription_auth("codex-acp", "OPENAI_API_KEY") is True


# ── resolve_agent_env: no-model subscription auth ──


class TestResolveAgentEnvNoModel:
    """Tests for the no-model path in resolve_agent_env."""

    def _patch_expanduser(self, monkeypatch, tmp_path):
        orig = Path.expanduser

        def fake(self):
            s = str(self)
            if s.startswith("~"):
                return tmp_path / s[2:]
            return orig(self)

        monkeypatch.setattr(Path, "expanduser", fake)

    def _resolve(self, agent="claude-agent-acp", model=None, agent_env=None):
        return resolve_agent_env(agent, model, agent_env)

    def test_no_model_with_api_key_works(self):
        """When API key is present and no model, no subscription auth needed."""
        result = self._resolve(agent_env={"ANTHROPIC_API_KEY": "sk-test"})
        assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result

    def test_no_model_subscription_auth_detected(self, monkeypatch, tmp_path):
        """No model + no API key + host credentials → subscription auth."""
        for k in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        ):
            monkeypatch.delenv(k, raising=False)
        creds_dir = tmp_path / ".claude"
        creds_dir.mkdir()
        (creds_dir / ".credentials.json").write_text('{"claudeAiOauth": {}}')
        self._patch_expanduser(monkeypatch, tmp_path)

        result = self._resolve(agent_env={})
        assert result["_BENCHFLOW_SUBSCRIPTION_AUTH"] == "1"

    def test_no_model_no_auth_no_error(self, monkeypatch, tmp_path):
        """No model + no API key + no host credentials → no error (no model to validate)."""
        for k in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        ):
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
