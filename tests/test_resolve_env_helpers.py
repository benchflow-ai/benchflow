"""Tests for extracted resolve_agent_env helper functions.

Covers auto_inherit_env, inject_vertex_credentials, resolve_provider_env,
check_subscription_auth, and the no-model subscription auth path in
resolve_agent_env.
"""

import json
from pathlib import Path

import pytest

from benchflow.agents.codex_config import CODEX_DEFAULT_AUTH_REQUEST_ENV
from benchflow.agents.env import (
    auto_inherit_env,
    check_subscription_auth,
    inject_vertex_credentials,
    resolve_agent_env,
    resolve_provider_env,
    validate_aws_bedrock_env,
)

# auto_inherit_env


class TestAutoInheritEnv:
    """Tests for auto_inherit_env — host env key inheritance."""

    @pytest.mark.parametrize(
        ("env_name", "env_value"),
        [
            pytest.param("ANTHROPIC_API_KEY", "sk-host", id="anthropic"),
            pytest.param("CODEX_AUTH_JSON", '{"tokens": {}}', id="codex-auth-json"),
            pytest.param("CODEX_ACCESS_TOKEN", "codex-access", id="codex-token"),
            pytest.param("CODEX_API_KEY", "codex-key", id="codex-api-key"),
            pytest.param("CLAUDE_OAUTH_TOKEN", "claude-oauth", id="claude-oauth"),
            pytest.param("OPENAI_API_KEY", "sk-oai", id="openai"),
            pytest.param("AWS_BEARER_TOKEN_BEDROCK", "bedrock-token", id="bedrock"),
            pytest.param("AWS_REGION", "us-east-1", id="bedrock-region"),
            pytest.param("ZAI_API_KEY", "zk-host", id="provider"),
            pytest.param("KIMI_API_KEY", "sk-kimi", id="kimi-api-key"),
            pytest.param("KIMI_BASE_URL", "https://api.moonshot.ai/v1", id="kimi-url"),
            pytest.param("AZURE_API_KEY", "az-host", id="azure-api-key"),
            pytest.param(
                "AZURE_API_ENDPOINT",
                "https://example.openai.azure.com/",
                id="azure-api-endpoint",
            ),
        ],
    )
    def test_inherits_key(self, env_name, env_value):
        env = {}
        auto_inherit_env(env, source_env={env_name: env_value})
        assert env[env_name] == env_value

    def test_does_not_overwrite_explicit(self):
        env = {"ANTHROPIC_API_KEY": "sk-explicit"}
        auto_inherit_env(env, source_env={"ANTHROPIC_API_KEY": "sk-host"})
        assert env["ANTHROPIC_API_KEY"] == "sk-explicit"

    def test_gemini_mirrored_to_google(self):
        env = {"GEMINI_API_KEY": "gk-test"}
        auto_inherit_env(env, source_env={})
        assert env["GOOGLE_API_KEY"] == "gk-test"

    def test_gemini_mirror_no_overwrite(self):
        env = {"GEMINI_API_KEY": "gk-test", "GOOGLE_API_KEY": "gk-explicit"}
        auto_inherit_env(env, source_env={})
        assert env["GOOGLE_API_KEY"] == "gk-explicit"

    def test_missing_host_key_not_added(self):
        env = {}
        auto_inherit_env(env, source_env={})
        assert "ANTHROPIC_API_KEY" not in env

    def test_aws_default_region_mirrored_to_aws_region(self):
        env = {"AWS_DEFAULT_REGION": "us-east-1"}
        auto_inherit_env(env, source_env={})
        assert env["AWS_REGION"] == "us-east-1"

    def test_aws_region_mirrored_to_aws_default_region(self):
        env = {"AWS_REGION": "us-east-1"}
        auto_inherit_env(env, source_env={})
        assert env["AWS_DEFAULT_REGION"] == "us-east-1"

    def test_claude_oauth_alias_mirrored_to_claude_code_token(self):
        """Guards PR #587: pasted Claude Code OAuth vars use both common names."""
        env = {"CLAUDE_OAUTH_TOKEN": "oauth-token"}
        auto_inherit_env(env, source_env={})
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token"

    def test_inherits_openai_base_url(self):
        """Guards fix from PR #255: OPENAI_BASE_URL must be inherited.

        codex-acp and opencode map BENCHFLOW_PROVIDER_BASE_URL → OPENAI_BASE_URL,
        but OPENAI_BASE_URL was missing from auto_inherit_env's key set, so
        users setting OPENAI_BASE_URL on the host had it silently dropped.
        """
        env: dict[str, str] = {}
        auto_inherit_env(
            env, source_env={"OPENAI_BASE_URL": "https://custom.openai.example/v1"}
        )
        assert env["OPENAI_BASE_URL"] == "https://custom.openai.example/v1"

    def test_inherits_benchflow_provider_api_key(self):
        """Guards issue #817: host BENCHFLOW_PROVIDER_API_KEY must be inherited.

        Users on self-hosted / proxy endpoints export BENCHFLOW_PROVIDER_API_KEY
        directly; without it on the allowlist the host value is silently dropped.
        """
        env: dict[str, str] = {}
        auto_inherit_env(
            env, source_env={"BENCHFLOW_PROVIDER_API_KEY": "sk-provider-host"}
        )
        assert env["BENCHFLOW_PROVIDER_API_KEY"] == "sk-provider-host"

    def test_inherits_benchflow_provider_base_url(self):
        """Guards issue #817: host BENCHFLOW_PROVIDER_BASE_URL must be inherited."""
        env: dict[str, str] = {}
        auto_inherit_env(
            env,
            source_env={"BENCHFLOW_PROVIDER_BASE_URL": "http://my-vllm-host:8000/v1"},
        )
        assert env["BENCHFLOW_PROVIDER_BASE_URL"] == "http://my-vllm-host:8000/v1"

    def test_empty_string_host_value_not_inherited(self):
        """An exported-but-empty host var ('export X=') must not be inherited.

        Copying '' would shadow a real value resolved downstream — an empty
        BENCHFLOW_PROVIDER_BASE_URL would block resolve_provider_env's
        setdefault from filling the provider's real endpoint.
        """
        env: dict[str, str] = {}
        auto_inherit_env(env, source_env={"BENCHFLOW_PROVIDER_BASE_URL": ""})
        assert "BENCHFLOW_PROVIDER_BASE_URL" not in env

    def test_whitespace_only_host_value_not_inherited(self):
        """A whitespace-only host var ('export X=" "') is also effectively unset.

        '   ' is truthy, so a bare `if value:` guard would still copy it and
        shadow downstream resolution exactly like an empty string does.
        """
        env: dict[str, str] = {}
        auto_inherit_env(env, source_env={"BENCHFLOW_PROVIDER_BASE_URL": "   "})
        assert "BENCHFLOW_PROVIDER_BASE_URL" not in env

    def test_empty_openai_base_url_not_inherited(self):
        """Empty-skip applies to every allowlisted key, not just the #817 keys.

        OPENAI_BASE_URL has always been on the allowlist, so this pins the
        empty-skip guard itself — independent of the #817 allowlist additions.
        """
        env: dict[str, str] = {}
        auto_inherit_env(env, source_env={"OPENAI_BASE_URL": ""})
        assert "OPENAI_BASE_URL" not in env

    def test_azure_endpoint_derives_resource(self):
        env = {"AZURE_API_ENDPOINT": "https://example-resource.openai.azure.com/"}
        auto_inherit_env(env, source_env={})
        assert env["AZURE_RESOURCE"] == "example-resource"

    def test_azure_resource_takes_precedence_over_endpoint(self):
        env = {
            "AZURE_RESOURCE": "explicit-resource",
            "AZURE_API_ENDPOINT": "https://endpoint-resource.openai.azure.com/",
        }
        auto_inherit_env(env, source_env={})
        assert env["AZURE_RESOURCE"] == "explicit-resource"

    def test_azure_anthropic_endpoint_alias_derives_resource(self):
        """AZURE_API_ENDPOINT also derives resource from the Anthropic-surface host."""
        env = {"AZURE_API_ENDPOINT": "https://example-resource.services.ai.azure.com/"}
        auto_inherit_env(env, source_env={})
        assert env["AZURE_RESOURCE"] == "example-resource"


# inject_vertex_credentials


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


# resolve_provider_env


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

    def test_openai_compatible_provider_maps_to_openhands_env(self):
        """Guards PR #587: direct provider envs reach OpenHands before LiteLLM rewrite."""
        env = {
            "KIMI_API_KEY": "sk-kimi",
            "KIMI_BASE_URL": "https://api.moonshot.ai/v1",
        }

        resolve_provider_env(env, "kimi/kimi-k2.6", "openhands")

        assert env["BENCHFLOW_PROVIDER_NAME"] == "kimi"
        assert env["BENCHFLOW_PROVIDER_MODEL"] == "kimi-k2.6"
        assert env["BENCHFLOW_PROVIDER_BASE_URL"] == "https://api.moonshot.ai/v1"
        assert env["BENCHFLOW_PROVIDER_API_KEY"] == "sk-kimi"
        assert env["LLM_BASE_URL"] == "https://api.moonshot.ai/v1"
        assert env["LLM_API_KEY"] == "sk-kimi"
        assert env["LLM_MODEL"] == "openai/kimi-k2.6"

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
        """codex-acp speaks openai-responses → routes to zai's OpenAI endpoint."""
        env = {"ZAI_API_KEY": "zk-test"}
        resolve_provider_env(env, "zai/glm-5", "codex-acp")
        assert env["BENCHFLOW_PROVIDER_BASE_URL"] == "https://api.z.ai/api/paas/v4"
        assert env["BENCHFLOW_PROVIDER_PROTOCOL"] == "openai-responses"
        assert env["OPENAI_BASE_URL"] == "https://api.z.ai/api/paas/v4"

    def test_openai_provider_sets_pi_openai_completions_env(self):
        """Guards PR #158 follow-up: pi-acp must not fallback to Anthropic for openai/...."""
        env = {"OPENAI_API_KEY": "sk-test"}

        resolve_provider_env(env, "openai/gpt-5.4-mini", "pi-acp")

        assert env["BENCHFLOW_PROVIDER_NAME"] == "openai"
        assert env["BENCHFLOW_PROVIDER_MODEL"] == "gpt-5.4-mini"
        assert env["BENCHFLOW_PROVIDER_PROTOCOL"] == "openai-completions"
        assert env["BENCHFLOW_PROVIDER_BASE_URL"] == "https://api.openai.com/v1"
        assert env["BENCHFLOW_PROVIDER_API_KEY"] == "sk-test"

    def test_openai_provider_sets_codex_responses_env(self):
        """Guards PR #158 follow-up: codex-acp must use Responses for openai/... models."""
        env = {"OPENAI_API_KEY": "sk-test"}

        resolve_provider_env(env, "openai/gpt-5.4-mini", "codex-acp")

        assert env["BENCHFLOW_PROVIDER_NAME"] == "openai"
        assert env["BENCHFLOW_PROVIDER_MODEL"] == "gpt-5.4-mini"
        assert env["BENCHFLOW_PROVIDER_PROTOCOL"] == "openai-responses"
        assert env["BENCHFLOW_PROVIDER_BASE_URL"] == "https://api.openai.com/v1"
        assert env["BENCHFLOW_PROVIDER_API_KEY"] == "sk-test"
        assert env["OPENAI_BASE_URL"] == "https://api.openai.com/v1"
        assert env["OPENAI_API_KEY"] == "sk-test"

    def test_azure_foundry_openai_maps_to_codex_env(self):
        env = {
            "AZURE_API_KEY": "az-test",
            "AZURE_RESOURCE": "example-resource",
        }
        resolve_provider_env(env, "azure-foundry-openai/gpt-5.5", "codex-acp")
        assert env["BENCHFLOW_PROVIDER_NAME"] == "azure-foundry-openai"
        assert env["BENCHFLOW_PROVIDER_MODEL"] == "gpt-5.5"
        assert env["BENCHFLOW_PROVIDER_PROTOCOL"] == "openai-responses"
        assert (
            env["BENCHFLOW_PROVIDER_BASE_URL"]
            == "https://example-resource.openai.azure.com/openai/v1"
        )
        assert env["OPENAI_API_KEY"] == "az-test"
        assert (
            env["OPENAI_BASE_URL"]
            == "https://example-resource.openai.azure.com/openai/v1"
        )

    def test_azure_foundry_anthropic_maps_to_claude_env(self):
        env = {
            "AZURE_API_KEY": "az-test",
            "AZURE_RESOURCE": "example-resource",
        }
        resolve_provider_env(
            env,
            "azure-foundry-anthropic/claude-opus-4-5",
            "claude-agent-acp",
        )
        assert env["BENCHFLOW_PROVIDER_NAME"] == "azure-foundry-anthropic"
        assert env["BENCHFLOW_PROVIDER_MODEL"] == "claude-opus-4-5"
        assert env["BENCHFLOW_PROVIDER_PROTOCOL"] == "anthropic-messages"
        assert (
            env["BENCHFLOW_PROVIDER_BASE_URL"]
            == "https://example-resource.services.ai.azure.com/anthropic"
        )
        assert env["ANTHROPIC_AUTH_TOKEN"] == "az-test"
        assert (
            env["ANTHROPIC_BASE_URL"]
            == "https://example-resource.services.ai.azure.com/anthropic"
        )

    def test_aws_bedrock_sets_placeholder_provider_key_for_codex(self):
        env = {
            "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
            "AWS_REGION": "us-east-1",
        }
        resolve_provider_env(env, "aws-bedrock/openai.gpt-oss-20b-1:0", "codex-acp")
        assert env["BENCHFLOW_PROVIDER_NAME"] == "aws-bedrock"
        assert env["BENCHFLOW_PROVIDER_MODEL"] == "openai.gpt-oss-20b-1:0"
        assert env["BENCHFLOW_PROVIDER_PROTOCOL"] == "openai-responses"
        assert env["BENCHFLOW_PROVIDER_API_KEY"] == "benchflow-litellm"
        assert env["OPENAI_API_KEY"] == "benchflow-litellm"

    def test_aws_bedrock_sets_placeholder_provider_key_for_claude(self):
        env = {
            "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
            "AWS_REGION": "us-east-1",
        }
        resolve_provider_env(
            env,
            "aws-bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
            "claude-agent-acp",
        )
        assert env["BENCHFLOW_PROVIDER_PROTOCOL"] == "anthropic-messages"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "benchflow-litellm"

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


# check_subscription_auth


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


# validate_aws_bedrock_env


class TestValidateAwsBedrockEnv:
    def test_requires_bearer_token(self):
        with pytest.raises(ValueError, match="AWS_BEARER_TOKEN_BEDROCK required"):
            validate_aws_bedrock_env({"AWS_REGION": "us-east-1"}, "aws-bedrock/model")

    def test_requires_region(self):
        with pytest.raises(
            ValueError, match="AWS_REGION or AWS_DEFAULT_REGION required"
        ):
            validate_aws_bedrock_env(
                {"AWS_BEARER_TOKEN_BEDROCK": "bedrock-token"},
                "aws-bedrock/model",
            )

    def test_normalizes_region_aliases(self):
        env = {
            "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
            "AWS_DEFAULT_REGION": "us-east-1",
        }
        validate_aws_bedrock_env(env, "aws-bedrock/model")
        assert env["AWS_REGION"] == "us-east-1"
        assert env["AWS_DEFAULT_REGION"] == "us-east-1"


# resolve_agent_env: no-model subscription auth


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
            "CODEX_ACCESS_TOKEN",
            "CODEX_API_KEY",
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
            "CODEX_ACCESS_TOKEN",
            "CODEX_API_KEY",
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
        for k in (
            "CODEX_ACCESS_TOKEN",
            "CODEX_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ):
            monkeypatch.delenv(k, raising=False)
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text("{}")
        self._patch_expanduser(monkeypatch, tmp_path)

        result = self._resolve(agent="codex-acp", agent_env={})
        assert result["_BENCHFLOW_SUBSCRIPTION_AUTH"] == "1"

    def test_no_model_codex_access_token_wins_over_host_auth(
        self, monkeypatch, tmp_path
    ):
        """Guards PR #296: CODEX_ACCESS_TOKEN is already usable auth."""
        for k in ("CODEX_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text("{}")
        self._patch_expanduser(monkeypatch, tmp_path)

        result = self._resolve(
            agent="codex-acp",
            agent_env={"CODEX_ACCESS_TOKEN": "access-token"},
        )

        assert result["CODEX_ACCESS_TOKEN"] == "access-token"
        assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result

    def test_no_model_codex_api_key_alias_normalizes(self, monkeypatch, tmp_path):
        """Guards PR #296: CODEX_API_KEY is Codex-native API-key auth."""
        for k in ("CODEX_ACCESS_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        self._patch_expanduser(monkeypatch, tmp_path)

        result = self._resolve(
            agent="codex-acp",
            agent_env={"CODEX_API_KEY": "codex-key"},
        )

        assert result["CODEX_API_KEY"] == "codex-key"
        assert result["OPENAI_API_KEY"] == "codex-key"
        assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result

    def test_no_model_empty_requires_env(self, monkeypatch, tmp_path):
        """Agent with empty requires_env (e.g. openclaw) needs no auth."""
        for k in (
            "ANTHROPIC_API_KEY",
            "CODEX_ACCESS_TOKEN",
            "CODEX_API_KEY",
            "OPENAI_API_KEY",
        ):
            monkeypatch.delenv(k, raising=False)
        self._patch_expanduser(monkeypatch, tmp_path)
        result = self._resolve(agent="openclaw", agent_env={})
        assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result


def test_task_env_resolves_from_dotenv(monkeypatch, tmp_path):
    """Guards ENG-80 dogfood regression: verifier env can resolve from .env."""
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=from-dotenv\n")
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(env_file))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    from benchflow.task.env import resolve_env_vars

    assert resolve_env_vars({"GOOGLE_API_KEY": "${GEMINI_API_KEY}"}) == {
        "GOOGLE_API_KEY": "from-dotenv"
    }


class TestResolveAgentEnvOracle:
    """Oracle runs solve.sh and never calls an LLM — must skip model-related env.

    Regression for the PR #173 follow-up: commit 360c460 removed the
    `agent != "oracle"` guard from resolve_agent_env, betting that CLI callers
    would pass model=None for oracle. But cli/main.py:eval_create (the live
    `bench eval run`) still passes `model or DEFAULT_MODEL`, so oracle
    reaches the chokepoint with a real model and triggers ANTHROPIC_API_KEY
    validation — breaking offline oracle runs that have no API key set.
    """

    def _patch_expanduser(self, monkeypatch, tmp_path):
        orig = Path.expanduser

        def fake(self):
            s = str(self)
            if s.startswith("~"):
                return tmp_path / s[2:]
            return orig(self)

        monkeypatch.setattr(Path, "expanduser", fake)

    def test_oracle_with_default_model_does_not_validate_api_key(
        self, monkeypatch, tmp_path
    ):
        """Oracle + DEFAULT_MODEL + no API key + no host auth must not raise."""
        for k in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        ):
            monkeypatch.delenv(k, raising=False)
        self._patch_expanduser(monkeypatch, tmp_path)

        result = resolve_agent_env("oracle", "claude-haiku-4-5-20251001", {})

        # Provider env never resolved — oracle never calls an LLM.
        assert "BENCHFLOW_PROVIDER_MODEL" not in result
        assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in result


class TestResolveAgentEnvAzureFoundry:
    """AZURE_API_KEY + AZURE_API_ENDPOINT resolve through the provider/env pipeline."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch, tmp_path):
        """Isolate from the host environment and any real .env file."""
        for k in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "AZURE_API_ENDPOINT",
            "AZURE_API_KEY",
            "AZURE_RESOURCE",
            "BENCHFLOW_PROVIDER_API_KEY",
            "BENCHFLOW_PROVIDER_BASE_URL",
            "CODEX_ACCESS_TOKEN",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
        ):
            monkeypatch.delenv(k, raising=False)
        empty = tmp_path / "empty.env"
        empty.write_text("")
        monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(empty))

    def test_codex_uses_azure_api_key_and_endpoint(self, monkeypatch):
        monkeypatch.setenv("AZURE_API_KEY", "az-test")
        monkeypatch.setenv(
            "AZURE_API_ENDPOINT", "https://example-resource.openai.azure.com/"
        )

        result = resolve_agent_env("codex-acp", "azure-foundry-openai/gpt-5.5", {})

        assert result["AZURE_API_KEY"] == "az-test"
        assert result["AZURE_RESOURCE"] == "example-resource"
        assert result["OPENAI_API_KEY"] == "az-test"
        assert (
            result["OPENAI_BASE_URL"]
            == "https://example-resource.openai.azure.com/openai/v1"
        )
        assert result["MODEL_PROVIDER"] == "benchflow-azure-foundry-openai"
        codex_config = json.loads(result["CODEX_CONFIG"])
        assert codex_config["model"] == "gpt-5.5"
        assert codex_config["model_provider"] == "benchflow-azure-foundry-openai"
        provider = codex_config["model_providers"]["benchflow-azure-foundry-openai"]
        assert provider == {
            "name": "azure-foundry-openai",
            "base_url": "https://example-resource.openai.azure.com/openai/v1",
            "env_key": "OPENAI_API_KEY",
            "wire_api": "responses",
            "supports_websockets": False,
        }

    def test_claude_uses_same_azure_key_on_anthropic_surface(self, monkeypatch):
        monkeypatch.setenv("AZURE_API_KEY", "az-test")
        monkeypatch.setenv(
            "AZURE_API_ENDPOINT", "https://example-resource.openai.azure.com/"
        )

        result = resolve_agent_env(
            "claude-agent-acp",
            "azure-foundry-anthropic/claude-opus-4-5",
            {},
        )

        assert result["AZURE_API_KEY"] == "az-test"
        assert result["AZURE_RESOURCE"] == "example-resource"
        assert result["ANTHROPIC_AUTH_TOKEN"] == "az-test"
        assert (
            result["ANTHROPIC_BASE_URL"]
            == "https://example-resource.services.ai.azure.com/anthropic"
        )

    def test_azure_openai_rejects_anthropic_agent_protocol(self, monkeypatch):
        """Guards PR #422: unsupported agent/provider protocol pairs fail fast."""
        monkeypatch.setenv("AZURE_API_KEY", "az-test")
        monkeypatch.setenv(
            "AZURE_API_ENDPOINT", "https://example-resource.openai.azure.com/"
        )

        with pytest.raises(
            ValueError,
            match=(
                r"claude-agent-acp.*requires provider protocol "
                r"'anthropic-messages'.*azure-foundry-openai.*only supports "
                r"openai-completions, openai-responses"
            ),
        ):
            resolve_agent_env(
                "claude-agent-acp",
                "azure-foundry-openai/gpt-5.5",
                {},
            )

    def test_azure_anthropic_rejects_openai_agent_protocol(self, monkeypatch):
        """Guards PR #422: fallback must not send OpenAI traffic to Anthropic."""
        monkeypatch.setenv("AZURE_API_KEY", "az-test")
        monkeypatch.setenv(
            "AZURE_API_ENDPOINT", "https://example-resource.openai.azure.com/"
        )

        with pytest.raises(
            ValueError,
            match=(
                r"codex-acp.*requires provider protocol 'openai-responses'.*"
                r"azure-foundry-anthropic.*only supports anthropic-messages"
            ),
        ):
            resolve_agent_env(
                "codex-acp",
                "azure-foundry-anthropic/claude-opus-4-5",
                {},
            )

    def test_azure_api_key_without_endpoint_fails_fast(self, monkeypatch):
        """Guards PR #3: Azure routes must not fall through to default OpenAI."""
        monkeypatch.setenv("AZURE_API_KEY", "az-test")

        with pytest.raises(
            ValueError,
            match=r"Azure AI Foundry model .* requires AZURE_RESOURCE",
        ):
            resolve_agent_env("codex-acp", "azure-foundry-openai/gpt-4o", {})

    def test_azure_api_key_with_unrecognized_endpoint_fails_fast(self, monkeypatch):
        """Guards PR #3: non-Azure endpoint hosts must fail before launch."""
        monkeypatch.setenv("AZURE_API_KEY", "az-test")
        monkeypatch.setenv("AZURE_API_ENDPOINT", "https://example.com/")

        with pytest.raises(
            ValueError,
            match=r"AZURE_API_ENDPOINT=https://<resource>\.openai\.azure\.com/",
        ):
            resolve_agent_env("codex-acp", "azure-foundry-openai/gpt-4o", {})

    def test_explicit_provider_base_url_can_override_azure_resource(self):
        """Guards PR #3: explicit provider base URL remains a valid override."""
        result = resolve_agent_env(
            "codex-acp",
            "azure-foundry-openai/gpt-4o",
            {
                "AZURE_API_KEY": "az-test",
                "BENCHFLOW_PROVIDER_BASE_URL": "https://proxy.example/openai/v1",
            },
        )

        assert (
            result["BENCHFLOW_PROVIDER_BASE_URL"] == "https://proxy.example/openai/v1"
        )
        assert result["OPENAI_BASE_URL"] == "https://proxy.example/openai/v1"


class TestResolveAgentEnvHostProviderEndpoint:
    """Guards issue #817: host BENCHFLOW_PROVIDER_* must reach the agent env.

    Users running self-hosted / OpenAI-compatible endpoints export
    BENCHFLOW_PROVIDER_BASE_URL (and BENCHFLOW_PROVIDER_API_KEY) on the host.
    auto_inherit_env's allowlist must include them — otherwise the host value
    is silently dropped, resolve_provider_env fills BENCHFLOW_PROVIDER_BASE_URL
    with the vllm provider's empty default, and the agent falls back to
    api.openai.com (401 Unauthorized).
    """

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch, tmp_path):
        """Isolate from the host environment and any real .env file."""
        for k in (
            "BENCHFLOW_PROVIDER_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY",
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_BASE_URL",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "LLM_API_KEY",
            "LLM_BASE_URL",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "ZAI_API_KEY",
            "KIMI_API_KEY",
            "KIMI_BASE_URL",
        ):
            monkeypatch.delenv(k, raising=False)
        empty = tmp_path / "empty.env"
        empty.write_text("")
        monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(empty))

    def test_host_provider_base_url_survives_into_resolved_env(self, monkeypatch):
        """vllm has an empty registry base_url — the host value must fill it."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-vllm-test")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "http://my-vllm-host:8000/v1")

        result = resolve_agent_env("codex-acp", "vllm/Qwen-test", {})

        assert result["BENCHFLOW_PROVIDER_BASE_URL"] == "http://my-vllm-host:8000/v1"
        # codex-acp env_mapping translates it to the agent-native var too.
        assert result["OPENAI_BASE_URL"] == "http://my-vllm-host:8000/v1"

    def test_explicit_agent_env_beats_host_provider_base_url(self, monkeypatch):
        """An explicit --agent-env override wins over the host value end-to-end."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-vllm-test")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "http://host/v1")

        result = resolve_agent_env(
            "codex-acp",
            "vllm/Qwen-test",
            {"BENCHFLOW_PROVIDER_BASE_URL": "http://explicit/v1"},
        )

        assert result["BENCHFLOW_PROVIDER_BASE_URL"] == "http://explicit/v1"

    def test_inherited_provider_base_url_does_not_shadow_registered_provider(
        self, monkeypatch
    ):
        """A global .env provider proxy must not override direct provider prefixes.

        Guards PR #587: a LiteLLM BENCHFLOW_PROVIDER_* default in .env broke
        direct Kimi/GLM/etc. runs by replacing the provider's own key and URL.
        """
        monkeypatch.setenv("KIMI_API_KEY", "sk-kimi")
        monkeypatch.setenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "http://host-proxy:9000/v1")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_API_KEY", "sk-host-proxy")

        result = resolve_agent_env("openhands", "kimi/kimi-k2.6", {})

        assert result["BENCHFLOW_PROVIDER_BASE_URL"] == "https://api.moonshot.ai/v1"
        assert result["BENCHFLOW_PROVIDER_API_KEY"] == "sk-kimi"
        assert result["LLM_BASE_URL"] == "https://api.moonshot.ai/v1"
        assert result["LLM_API_KEY"] == "sk-kimi"

    def test_inherited_provider_proxy_does_not_shadow_bare_gemini_model(
        self, monkeypatch
    ):
        """Guards this PR: a global LiteLLM proxy must not hijack Gemini direct runs."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
        monkeypatch.setenv(
            "BENCHFLOW_PROVIDER_BASE_URL", "https://llm-proxy.example.test"
        )
        monkeypatch.setenv("BENCHFLOW_PROVIDER_API_KEY", "sk-proxy")
        monkeypatch.setenv("LLM_BASE_URL", "https://llm-proxy.example.test")
        monkeypatch.setenv("LLM_API_KEY", "sk-proxy")

        result = resolve_agent_env("openhands", "gemini-3.5-flash", {})

        assert "BENCHFLOW_PROVIDER_BASE_URL" not in result
        assert "LLM_BASE_URL" not in result
        assert result["BENCHFLOW_PROVIDER_API_KEY"] == "test-gemini-key"
        assert result["LLM_API_KEY"] == "test-gemini-key"
        assert result["LLM_MODEL"] == "gemini/gemini-3.5-flash"

    def test_explicit_provider_base_url_can_override_registered_provider(self):
        """An explicit --agent-env generic endpoint remains a valid override."""
        result = resolve_agent_env(
            "openhands",
            "kimi/kimi-k2.6",
            {
                "KIMI_API_KEY": "sk-kimi",
                "KIMI_BASE_URL": "https://api.moonshot.ai/v1",
                "BENCHFLOW_PROVIDER_BASE_URL": "http://explicit-proxy:9000/v1",
                "BENCHFLOW_PROVIDER_API_KEY": "sk-explicit-proxy",
            },
        )

        assert result["BENCHFLOW_PROVIDER_BASE_URL"] == "http://explicit-proxy:9000/v1"
        assert result["BENCHFLOW_PROVIDER_API_KEY"] == "sk-explicit-proxy"
        assert result["LLM_BASE_URL"] == "http://explicit-proxy:9000/v1"
        assert result["LLM_API_KEY"] == "sk-explicit-proxy"

    def test_explicit_openhands_llm_env_can_override_registered_provider_without_native_key(
        self,
    ):
        """Guards PR #780: OpenHands proxy flags must not require native DeepSeek env."""
        result = resolve_agent_env(
            "openhands",
            "deepseek/deepseek-v4-flash",
            {
                "LLM_BASE_URL": "https://llm-proxy.example.test/v1",
                "LLM_API_KEY": "sk-explicit-proxy",
            },
        )

        assert (
            result["BENCHFLOW_PROVIDER_BASE_URL"] == "https://llm-proxy.example.test/v1"
        )
        assert result["BENCHFLOW_PROVIDER_API_KEY"] == "sk-explicit-proxy"
        assert result["LLM_BASE_URL"] == "https://llm-proxy.example.test/v1"
        assert result["LLM_API_KEY"] == "sk-explicit-proxy"
        assert result["LLM_MODEL"] == "openai/deepseek-v4-flash"
        assert "DEEPSEEK_API_KEY" not in result
        assert "DEEPSEEK_BASE_URL" not in result

    def test_explicit_generic_proxy_env_can_override_registered_provider_without_native_key(
        self,
    ):
        """Guards PR #780: generic provider proxy env is a valid explicit route."""
        result = resolve_agent_env(
            "openhands",
            "deepseek/deepseek-v4-flash",
            {
                "BENCHFLOW_PROVIDER_BASE_URL": "https://llm-proxy.example.test/v1",
                "BENCHFLOW_PROVIDER_API_KEY": "sk-explicit-proxy",
            },
        )

        assert (
            result["BENCHFLOW_PROVIDER_BASE_URL"] == "https://llm-proxy.example.test/v1"
        )
        assert result["BENCHFLOW_PROVIDER_API_KEY"] == "sk-explicit-proxy"
        assert result["LLM_BASE_URL"] == "https://llm-proxy.example.test/v1"
        assert result["LLM_API_KEY"] == "sk-explicit-proxy"
        assert "DEEPSEEK_API_KEY" not in result

    def test_no_host_override_keeps_resolved_provider_url(self, monkeypatch):
        """Sanity counterpart: without a host override zai's own endpoint is used."""
        monkeypatch.setenv("ZAI_API_KEY", "zk-test")

        result = resolve_agent_env("codex-acp", "zai/glm-5", {})

        assert result["BENCHFLOW_PROVIDER_BASE_URL"] == "https://api.z.ai/api/paas/v4"

    def test_empty_host_base_url_does_not_shadow_resolved_url(self, monkeypatch):
        """'export BENCHFLOW_PROVIDER_BASE_URL=' must not blank a real URL."""
        monkeypatch.setenv("ZAI_API_KEY", "zk-test")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "")

        result = resolve_agent_env("codex-acp", "zai/glm-5", {})

        assert result["BENCHFLOW_PROVIDER_BASE_URL"] == "https://api.z.ai/api/paas/v4"

    def test_provider_keys_inherited_from_dotenv_file(self, monkeypatch, tmp_path):
        """resolve_agent_env also inherits BENCHFLOW_PROVIDER_* from a .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "BENCHFLOW_PROVIDER_BASE_URL=http://dotenv-host:8000/v1\n"
            "BENCHFLOW_PROVIDER_API_KEY=sk-from-dotenv\n"
        )
        monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(env_file))

        result = resolve_agent_env("codex-acp", "vllm/Qwen-test", {})

        assert result["BENCHFLOW_PROVIDER_BASE_URL"] == "http://dotenv-host:8000/v1"
        assert result["OPENAI_BASE_URL"] == "http://dotenv-host:8000/v1"
        assert result["OPENAI_API_KEY"] == "sk-from-dotenv"

    def test_openhands_host_provider_keys_map_to_llm_vars(self, monkeypatch):
        """openhands maps BENCHFLOW_PROVIDER_{BASE_URL,API_KEY} → LLM_{BASE_URL,API_KEY}."""
        monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "http://my-vllm-host:8000/v1")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_API_KEY", "sk-provider-host")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-vllm")

        result = resolve_agent_env("openhands", "vllm/Qwen-test", {})

        assert result["LLM_BASE_URL"] == "http://my-vllm-host:8000/v1"
        assert result["LLM_API_KEY"] == "sk-provider-host"

    def test_whitespace_host_base_url_does_not_shadow_resolved_url(self, monkeypatch):
        """'export BENCHFLOW_PROVIDER_BASE_URL=" "' must not blank a real URL.

        A whitespace-only value is the same operator mistake as an empty one
        and must not shadow the provider's resolved endpoint.
        """
        monkeypatch.setenv("ZAI_API_KEY", "zk-test")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "   ")

        result = resolve_agent_env("codex-acp", "zai/glm-5", {})

        assert result["BENCHFLOW_PROVIDER_BASE_URL"] == "https://api.z.ai/api/paas/v4"

    def test_empty_host_provider_api_key_does_not_shadow_resolved_key(
        self, monkeypatch
    ):
        """'export BENCHFLOW_PROVIDER_API_KEY=' must not shadow the resolved key.

        The empty-string class of bug applies to the API key too: an empty host
        value must not block resolve_provider_env from filling it from the
        provider's own credential (ZAI_API_KEY here).
        """
        monkeypatch.setenv("ZAI_API_KEY", "zk-test")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_API_KEY", "")

        result = resolve_agent_env("codex-acp", "zai/glm-5", {})

        assert result["BENCHFLOW_PROVIDER_API_KEY"] == "zk-test"
        assert result["OPENAI_API_KEY"] == "zk-test"


class TestResolveAgentEnvCodexOpenAIPrefix:
    """Registering the first-party ``openai`` provider must not regress Codex auth.

    Codex-acp historically accepts four auth modes against api.openai.com:
    OPENAI_API_KEY / CODEX_API_KEY (alias) / CODEX_ACCESS_TOKEN /
    host ``~/.codex/auth.json`` subscription auth. The first three are also
    valid for bare ``gpt-*`` model IDs. After registering ``openai`` as a
    provider, ``find_provider("openai/...")`` returns a match and the native-
    OpenAI gate must still treat the canonical endpoint as native — otherwise
    the alias/access-token/subscription paths silently break for users who
    switch from ``gpt-5.4-mini`` to ``openai/gpt-5.4-mini``.

    Custom proxies (``vllm/``, ``us-openai/``, etc.) must keep requiring an
    explicit OPENAI_API_KEY — subscription/access-token auth does not apply.
    """

    def _patch_expanduser(self, monkeypatch, tmp_path):
        orig = Path.expanduser

        def fake(self):
            s = str(self)
            if s.startswith("~"):
                return tmp_path / s[2:]
            return orig(self)

        monkeypatch.setattr(Path, "expanduser", fake)

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for k in (
            "OPENAI_API_KEY",
            "CODEX_API_KEY",
            "CODEX_ACCESS_TOKEN",
            "CODEX_AUTH_JSON",
            "ANTHROPIC_API_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

    def test_codex_api_key_alias_works_for_openai_prefix(self, monkeypatch, tmp_path):
        self._patch_expanduser(monkeypatch, tmp_path)
        result = resolve_agent_env(
            "codex-acp",
            "openai/gpt-5.4-mini",
            {"CODEX_API_KEY": "codex-key"},
        )
        assert result["CODEX_API_KEY"] == "codex-key"
        assert result["OPENAI_API_KEY"] == "codex-key"
        assert json.loads(result[CODEX_DEFAULT_AUTH_REQUEST_ENV]) == {
            "methodId": "api-key",
            "_meta": {"api-key": {"apiKey": "codex-key"}},
        }

    def test_codex_access_token_works_for_openai_prefix(self, monkeypatch, tmp_path):
        self._patch_expanduser(monkeypatch, tmp_path)
        result = resolve_agent_env(
            "codex-acp",
            "openai/gpt-5.4-mini",
            {"CODEX_ACCESS_TOKEN": "access-token"},
        )
        assert result["CODEX_ACCESS_TOKEN"] == "access-token"

    def test_codex_auth_json_works_for_openai_prefix(self, monkeypatch, tmp_path):
        self._patch_expanduser(monkeypatch, tmp_path)
        result = resolve_agent_env(
            "codex-acp",
            "openai/gpt-5.4-mini",
            {"CODEX_AUTH_JSON": '{"tokens": {}}'},
        )
        assert result["CODEX_AUTH_JSON"] == '{"tokens": {}}'

    def test_codex_host_subscription_auth_works_for_openai_prefix(
        self, monkeypatch, tmp_path
    ):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text("{}")
        self._patch_expanduser(monkeypatch, tmp_path)

        result = resolve_agent_env("codex-acp", "openai/gpt-5.4-mini", {})

        assert result["_BENCHFLOW_SUBSCRIPTION_AUTH"] == "1"

    def test_us_openai_prefix_still_requires_explicit_key(self, monkeypatch, tmp_path):
        """Regional endpoint is not the canonical api.openai.com — host auth must not apply."""
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text("{}")
        self._patch_expanduser(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            resolve_agent_env("codex-acp", "us-openai/gpt-5.4-mini", {})

    def test_vllm_prefix_still_requires_explicit_key(self, monkeypatch, tmp_path):
        """Custom OpenAI-compatible endpoints keep rejecting subscription auth."""
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text("{}")
        self._patch_expanduser(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            resolve_agent_env("codex-acp", "vllm/Qwen-test", {})
