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

    def test_vertex_zai_exists(self):
        assert "vertex-zai" in PROVIDERS

    def test_zai_config(self):
        p = PROVIDERS["zai"]
        assert p.base_url == "https://api.z.ai/api/paas/v4"
        assert p.api_protocol == "openai-completions"
        assert p.auth_type == "api_key"
        assert p.auth_env == "ZAI_API_KEY"

    def test_vertex_zai_config(self):
        p = PROVIDERS["vertex-zai"]
        assert "aiplatform.googleapis.com" in p.base_url
        assert p.auth_type == "adc"
        assert p.auth_env is None
        assert "project_id" in p.url_params

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

    def test_vertex_zai_prefix(self):
        name, cfg = find_provider("vertex-zai/zai-org/glm-5-maas")
        assert name == "vertex-zai"
        assert cfg.auth_type == "adc"

    def test_case_insensitive(self):
        name, _ = find_provider("ZAI/glm-5")
        assert name == "zai"

    def test_unknown_prefix_returns_none(self):
        assert find_provider("anthropic/claude-sonnet-4-6") is None

    def test_no_prefix_returns_none(self):
        assert find_provider("glm-5") is None

    def test_longest_prefix_wins(self):
        """vertex-zai/ should match before a hypothetical 'vertex/' provider."""
        name, _ = find_provider("vertex-zai/zai-org/glm-5-maas")
        assert name == "vertex-zai"


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
            name="vertex-zai",
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
            name="vertex-zai",
            base_url="https://example.com/{project_id}",
            api_protocol="openai-completions",
            auth_type="adc",
            url_params={"project_id": "GOOGLE_CLOUD_PROJECT"},
        )
        with pytest.raises((KeyError, ValueError)):
            resolve_base_url(p, {})


# ── resolve_auth_env: which env var does this provider need? ──


class TestResolveAuthEnv:
    """resolve_auth_env returns the env var name for a model's provider."""

    def test_zai_model(self):
        assert resolve_auth_env("zai/glm-5") == "ZAI_API_KEY"

    def test_vertex_zai_returns_none(self):
        """Vertex models use ADC, no API key env var."""
        assert resolve_auth_env("vertex-zai/zai-org/glm-5-maas") is None

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

    def test_is_vertex_model_vertex_zai(self):
        """vertex-zai/ should still be recognized as vertex."""
        from benchflow.agents.registry import is_vertex_model
        assert is_vertex_model("vertex-zai/zai-org/glm-5-maas") is True

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

    def test_vertex_zai_has_models(self):
        p = PROVIDERS["vertex-zai"]
        assert hasattr(p, "models") and len(p.models) > 0

    def test_model_has_required_fields(self):
        """Each model entry should have at least id and name."""
        for key, cfg in PROVIDERS.items():
            if not cfg.models:
                continue
            for m in cfg.models:
                assert "id" in m, f"Provider {key!r} model missing 'id'"
                assert "name" in m, f"Provider {key!r} model missing 'name'"
