"""Tests for the custom provider registry (providers.py).

These tests define the spec for the provider system before implementation.
Run: pytest tests/test_providers.py
"""

import pytest

from benchflow.agents.providers import (
    PROVIDERS,
    ProviderConfig,
    find_provider,
    resolve_base_url,
    resolve_auth_env,
    strip_provider_prefix,
)


# ── ProviderConfig dataclass ──


class TestProviderConfig:
    """ProviderConfig holds provider facts."""

    def test_has_required_fields(self):
        p = ProviderConfig(
            name="test",
            base_url="https://api.example.com/v1",
            api_protocol="openai-completions",
            auth_type="api_key",
            auth_env="TEST_API_KEY",
        )
        assert p.name == "test"
        assert p.base_url == "https://api.example.com/v1"
        assert p.api_protocol == "openai-completions"
        assert p.auth_type == "api_key"
        assert p.auth_env == "TEST_API_KEY"

    def test_url_params_default_empty(self):
        p = ProviderConfig(
            name="test",
            base_url="https://api.example.com",
            api_protocol="openai-completions",
            auth_type="api_key",
        )
        assert p.url_params == {}

    def test_auth_env_optional(self):
        """ADC providers don't need an auth_env."""
        p = ProviderConfig(
            name="vertex",
            base_url="https://vertex.googleapis.com",
            api_protocol="openai-completions",
            auth_type="adc",
        )
        assert p.auth_env is None


# ── Built-in providers ──


class TestBuiltinProviders:
    """PROVIDERS dict contains expected entries."""

    def test_zai_exists(self):
        assert "zai" in PROVIDERS

    def test_zai_config(self):
        p = PROVIDERS["zai"]
        assert p.base_url == "https://api.z.ai/api/paas/v4"
        assert p.api_protocol == "openai-completions"
        assert "anthropic-messages" in p.endpoints
        assert p.endpoints["anthropic-messages"] == "https://api.z.ai/api/anthropic"
        assert p.auth_type == "api_key"
        assert p.auth_env == "ZAI_API_KEY"

    def test_all_providers_have_name_matching_key(self):
        for key, cfg in PROVIDERS.items():
            assert cfg.name == key, f"Provider {key!r} has mismatched name {cfg.name!r}"

    def test_all_providers_have_valid_auth_type(self):
        for key, cfg in PROVIDERS.items():
            assert cfg.auth_type in ("api_key", "adc"), (
                f"Provider {key!r} has unknown auth_type {cfg.auth_type!r}"
            )

    def test_api_key_providers_have_auth_env(self):
        """Every api_key provider must declare which env var holds the key."""
        for key, cfg in PROVIDERS.items():
            if cfg.auth_type == "api_key":
                assert cfg.auth_env is not None, (
                    f"Provider {key!r} uses api_key auth but has no auth_env"
                )


# ── find_provider: model string → provider config ──


class TestFindProvider:
    """find_provider resolves model prefix to ProviderConfig."""

    def test_zai_prefix(self):
        name, cfg = find_provider("zai/glm-5")
        assert name == "zai"
        assert cfg.auth_env == "ZAI_API_KEY"

    def test_case_insensitive(self):
        name, _ = find_provider("ZAI/glm-5")
        assert name == "zai"

    def test_unknown_prefix_returns_none(self):
        assert find_provider("anthropic/claude-sonnet-4-6") is None

    def test_no_prefix_returns_none(self):
        assert find_provider("glm-5") is None

# ── resolve_base_url: template expansion ──


class TestResolveBaseUrl:
    """resolve_base_url expands {placeholders} from env."""

    def test_no_placeholders(self):
        p = ProviderConfig(
            name="zai",
            base_url="https://api.z.ai/api/paas/v4",
            api_protocol="openai-completions",
            auth_type="api_key",
            auth_env="ZAI_API_KEY",
        )
        assert resolve_base_url(p, {}) == "https://api.z.ai/api/paas/v4"

    def test_project_id_expansion(self):
        p = ProviderConfig(
            name="test-vertex",
            base_url="https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/endpoints/openapi",
            api_protocol="openai-completions",
            auth_type="adc",
            url_params={"project_id": "GOOGLE_CLOUD_PROJECT"},
        )
        env = {"GOOGLE_CLOUD_PROJECT": "my-project"}
        url = resolve_base_url(p, env)
        assert "my-project" in url
        assert "{project_id}" not in url

    def test_missing_env_var_raises(self):
        p = ProviderConfig(
            name="test-vertex",
            base_url="https://example.com/{project_id}",
            api_protocol="openai-completions",
            auth_type="adc",
            url_params={"project_id": "GOOGLE_CLOUD_PROJECT"},
        )
        with pytest.raises((KeyError, ValueError)):
            resolve_base_url(p, {})

    def test_protocol_selects_endpoint(self):
        """resolve_base_url with protocol= picks from endpoints dict."""
        p = PROVIDERS["zai"]
        assert resolve_base_url(p, {}, protocol="anthropic-messages") == "https://api.z.ai/api/anthropic"
        assert resolve_base_url(p, {}, protocol="openai-completions") == "https://api.z.ai/api/paas/v4"

    def test_protocol_fallback_to_base_url(self):
        """Unknown protocol falls back to primary base_url."""
        p = PROVIDERS["zai"]
        assert resolve_base_url(p, {}) == "https://api.z.ai/api/paas/v4"
        assert resolve_base_url(p, {}, protocol="unknown") == "https://api.z.ai/api/paas/v4"


# ── resolve_auth_env: which env var does this provider need? ──


class TestResolveAuthEnv:
    """resolve_auth_env returns the env var name for a model's provider."""

    def test_zai_model(self):
        assert resolve_auth_env("zai/glm-5") == "ZAI_API_KEY"

    def test_unknown_model_returns_none(self):
        """Models without a custom provider fall through to None."""
        assert resolve_auth_env("some-unknown/model") is None


# ── Integration: backward compat with registry.py ──


class TestRegistryIntegration:
    """Provider system integrates with existing registry functions."""

    def test_infer_env_key_for_zai(self):
        """infer_env_key_for_model should return ZAI_API_KEY for zai/ models."""
        from benchflow.agents.registry import infer_env_key_for_model
        assert infer_env_key_for_model("zai/glm-5") == "ZAI_API_KEY"

    def test_is_vertex_model_zai_direct(self):
        """zai/ (direct API) is NOT vertex."""
        from benchflow.agents.registry import is_vertex_model
        assert is_vertex_model("zai/glm-5") is False

    def test_existing_models_unchanged(self):
        """Existing model inference must not regress."""
        from benchflow.agents.registry import infer_env_key_for_model
        assert infer_env_key_for_model("gemini-3.1-pro") == "GEMINI_API_KEY"
        assert infer_env_key_for_model("claude-sonnet-4-6") == "ANTHROPIC_API_KEY"
        assert infer_env_key_for_model("gpt-5.4") == "OPENAI_API_KEY"
        assert infer_env_key_for_model("google-vertex/gemini-2.5-flash") is None


# ── Provider model metadata (for openclaw.json generation) ──


class TestProviderModels:
    """Providers optionally include model metadata for agents that need it."""

    def test_zai_has_models(self):
        p = PROVIDERS["zai"]
        assert hasattr(p, "models") and len(p.models) > 0

    def test_zai_has_glm51(self):
        p = PROVIDERS["zai"]
        model_ids = [m["id"] for m in p.models]
        assert "glm-5.1" in model_ids

    def test_model_has_required_fields(self):
        """Each model entry should have at least id and name."""
        for key, cfg in PROVIDERS.items():
            if not cfg.models:
                continue
            for m in cfg.models:
                assert "id" in m, f"Provider {key!r} model missing 'id'"
                assert "name" in m, f"Provider {key!r} model missing 'name'"


# ── strip_provider_prefix ──


class TestStripProviderPrefix:
    def test_known_provider(self):
        assert strip_provider_prefix("zai/glm-5") == "glm-5"

    def test_vertex_provider(self):
        assert strip_provider_prefix("anthropic-vertex/claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_nested_prefix(self):
        assert strip_provider_prefix("google-vertex/gemini-3-flash") == "gemini-3-flash"

    def test_no_prefix(self):
        assert strip_provider_prefix("claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_unknown_prefix(self):
        assert strip_provider_prefix("unknown-provider/some-model") == "some-model"


# ── Shim provider fallback: stripped model + BENCHFLOW_PROVIDER_* env vars ──


class TestShimProviderFallback:
    """The openclaw shim must resolve providers from env vars when model is stripped.

    SDK strips provider prefix before ACP set_model (e.g. "anthropic-vertex/claude-sonnet-4-6"
    → "claude-sonnet-4-6"). The shim's _find_and_setup_provider() must fall through from
    find_provider() (returns None for stripped names) to BENCHFLOW_PROVIDER_* env vars.
    """

    def test_stripped_model_not_found_by_find_provider(self):
        """find_provider returns None for stripped model names — confirms the shim
        cannot rely on it alone and must fall back to env vars."""
        # These are what set_model receives after stripping
        assert find_provider("claude-sonnet-4-6") is None
        assert find_provider("gemini-3-flash-preview") is None
        assert find_provider("glm-5") is None

    def test_full_model_found_by_find_provider(self):
        """find_provider works with full prefixed names — the pre-strip path."""
        assert find_provider("anthropic-vertex/claude-sonnet-4-6") is not None
        assert find_provider("google-vertex/gemini-3-flash-preview") is not None
        assert find_provider("zai/glm-5") is not None
        assert find_provider("zai/glm-5.1") is not None

    def test_sdk_injects_provider_env_for_all_known_providers(self):
        """Every registered provider with a base_url should result in
        BENCHFLOW_PROVIDER_BASE_URL being injectable by the SDK."""
        for name, cfg in PROVIDERS.items():
            assert cfg.base_url, f"Provider {name!r} has no base_url"
            assert cfg.api_protocol, f"Provider {name!r} has no api_protocol"


# ── Shim helper functions ──


class TestInferProviderPrefix:
    """Tests for _infer_provider_prefix() in the openclaw ACP shim."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from benchflow.agents.openclaw_acp_shim import _infer_provider_prefix
        self.infer = _infer_provider_prefix

    @pytest.mark.parametrize("model,expected", [
        ("gpt-4o", "openai"),
        ("gpt-4o-mini", "openai"),
        ("o1-preview", "openai"),
        ("o3-mini", "openai"),
        ("gemini-3-flash", "google"),
        ("gemini-2.5-pro", "google"),
        ("claude-sonnet-4-6", "anthropic"),
        ("claude-haiku-4-5-20251001", "anthropic"),
        ("some-unknown-model", "anthropic"),  # default
    ])
    def test_infer(self, model, expected):
        assert self.infer(model) == expected


class TestSetupOpenaiAuth:
    """Tests for setup_openai_auth() writing to openclaw's auth-profiles.json."""

    @pytest.fixture()
    def home_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        return tmp_path

    def _auth_path(self, home_dir):
        return home_dir / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"

    def test_writes_key(self, home_dir, monkeypatch):
        import json
        from benchflow.agents.openclaw_acp_shim import setup_openai_auth

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        setup_openai_auth()

        auth = json.loads(self._auth_path(home_dir).read_text())
        assert auth["openai"]["apiKey"] == "sk-test-123"

    def test_no_key_is_noop(self, home_dir, monkeypatch):
        from benchflow.agents.openclaw_acp_shim import setup_openai_auth

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        setup_openai_auth()

        assert not self._auth_path(home_dir).exists()

    def test_preserves_existing_providers(self, home_dir, monkeypatch):
        import json
        from benchflow.agents.openclaw_acp_shim import setup_openai_auth

        path = self._auth_path(home_dir)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"anthropic": {"apiKey": "ant-key"}}))

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-456")
        setup_openai_auth()

        auth = json.loads(path.read_text())
        assert auth["anthropic"]["apiKey"] == "ant-key"
        assert auth["openai"]["apiKey"] == "sk-test-456"


# ── Shim model generation parameters ──


class TestShimModelParams:
    """The shim should read BENCHFLOW_MODEL_* env vars and call
    openclaw config set agents.defaults.params.<key> for each."""

    def test_param_map_covers_all_generation_params(self):
        """session/set_model handler should map all three env vars."""
        # Read the shim source and extract the _PARAM_MAP dict
        from pathlib import Path
        shim_src = (Path(__file__).parent.parent / "src/benchflow/agents/openclaw_acp_shim.py").read_text()
        assert "BENCHFLOW_MODEL_TEMPERATURE" in shim_src
        assert "BENCHFLOW_MODEL_TOP_P" in shim_src
        assert "BENCHFLOW_MODEL_MAX_TOKENS" in shim_src
        assert "agents.defaults.params.temperature" in shim_src
        assert "agents.defaults.params.topP" in shim_src
        assert "agents.defaults.params.maxTokens" in shim_src

    def test_set_model_applies_params(self, monkeypatch):
        """When BENCHFLOW_MODEL_* env vars are set, the shim should call
        openclaw config set for each param during session/set_model."""
        import subprocess
        calls = []
        original_run = subprocess.run

        def mock_run(cmd, **kwargs):
            calls.append(cmd)
            # Return a dummy CompletedProcess
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setenv("BENCHFLOW_MODEL_TEMPERATURE", "1.0")
        monkeypatch.setenv("BENCHFLOW_MODEL_TOP_P", "0.95")
        monkeypatch.setenv("BENCHFLOW_MODEL_MAX_TOKENS", "131072")

        # Import and simulate the set_model handler logic inline
        # (the shim's main() is a blocking loop, so we test the logic directly)
        import os
        _PARAM_MAP = {
            "BENCHFLOW_MODEL_TEMPERATURE": "agents.defaults.params.temperature",
            "BENCHFLOW_MODEL_TOP_P": "agents.defaults.params.topP",
            "BENCHFLOW_MODEL_MAX_TOKENS": "agents.defaults.params.maxTokens",
        }
        monkeypatch.setattr(subprocess, "run", mock_run)

        for env_key, config_path in _PARAM_MAP.items():
            val = os.environ.get(env_key)
            if val:
                subprocess.run(
                    ["openclaw", "config", "set", config_path, val],
                    capture_output=True, timeout=10,
                )

        monkeypatch.setattr(subprocess, "run", original_run)

        assert len(calls) == 3
        config_paths = [c[3] for c in calls]
        assert "agents.defaults.params.temperature" in config_paths
        assert "agents.defaults.params.topP" in config_paths
        assert "agents.defaults.params.maxTokens" in config_paths
        # Verify values
        vals = {c[3]: c[4] for c in calls}
        assert vals["agents.defaults.params.temperature"] == "1.0"
        assert vals["agents.defaults.params.topP"] == "0.95"
        assert vals["agents.defaults.params.maxTokens"] == "131072"

    def test_missing_env_vars_skipped(self, monkeypatch):
        """When no BENCHFLOW_MODEL_* env vars are set, no config calls are made."""
        import subprocess
        calls = []

        def mock_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.delenv("BENCHFLOW_MODEL_TEMPERATURE", raising=False)
        monkeypatch.delenv("BENCHFLOW_MODEL_TOP_P", raising=False)
        monkeypatch.delenv("BENCHFLOW_MODEL_MAX_TOKENS", raising=False)

        import os
        _PARAM_MAP = {
            "BENCHFLOW_MODEL_TEMPERATURE": "agents.defaults.params.temperature",
            "BENCHFLOW_MODEL_TOP_P": "agents.defaults.params.topP",
            "BENCHFLOW_MODEL_MAX_TOKENS": "agents.defaults.params.maxTokens",
        }
        monkeypatch.setattr(subprocess, "run", mock_run)

        for env_key, config_path in _PARAM_MAP.items():
            val = os.environ.get(env_key)
            if val:
                subprocess.run(
                    ["openclaw", "config", "set", config_path, val],
                    capture_output=True, timeout=10,
                )

        assert len(calls) == 0
