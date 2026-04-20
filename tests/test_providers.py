"""Tests for the custom provider registry (providers.py).

These tests define the spec for the provider system before implementation.
Run: pytest tests/test_providers.py
"""

import pytest

from benchflow.agents.providers import (
    PROVIDERS,
    ProviderConfig,
    find_provider,
    resolve_auth_env,
    resolve_base_url,
    strip_provider_prefix,
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
        assert (
            resolve_base_url(p, {}, protocol="anthropic-messages")
            == "https://api.z.ai/api/anthropic"
        )
        assert (
            resolve_base_url(p, {}, protocol="openai-completions")
            == "https://api.z.ai/api/paas/v4"
        )

    def test_protocol_fallback_to_base_url(self):
        """Unknown protocol falls back to primary base_url."""
        p = PROVIDERS["zai"]
        assert resolve_base_url(p, {}) == "https://api.z.ai/api/paas/v4"
        assert (
            resolve_base_url(p, {}, protocol="unknown")
            == "https://api.z.ai/api/paas/v4"
        )


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


# ── Provider model metadata (for openclaw.json generation) ──


class TestProviderModels:
    """Providers optionally include model metadata for agents that need it."""

    def test_model_has_required_fields(self):
        """Each model entry should have at least id and name."""
        for cfg in PROVIDERS.values():
            assert all("id" in m and "name" in m for m in cfg.models)


# ── strip_provider_prefix ──


class TestStripProviderPrefix:
    def test_known_provider(self):
        assert strip_provider_prefix("zai/glm-5") == "glm-5"

    def test_vertex_provider(self):
        assert (
            strip_provider_prefix("anthropic-vertex/claude-sonnet-4-6")
            == "claude-sonnet-4-6"
        )

    def test_nested_prefix(self):
        assert strip_provider_prefix("google-vertex/gemini-3-flash") == "gemini-3-flash"

    def test_no_prefix(self):
        assert strip_provider_prefix("claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_unknown_prefix_preserved(self):
        # Unregistered prefix passes through unchanged — avoids mangling
        # HuggingFace-style IDs whose org is not a registered provider.
        assert (
            strip_provider_prefix("unknown-provider/some-model")
            == "unknown-provider/some-model"
        )

    def test_bare_huggingface_id_preserved(self):
        # Regression: bare HF ID (no registered prefix) must keep org/model.
        assert strip_provider_prefix("Qwen/Qwen3-Coder") == "Qwen/Qwen3-Coder"
        assert strip_provider_prefix("Qwen/Qwen3.5-35B-A3B") == "Qwen/Qwen3.5-35B-A3B"

    def test_registered_prefix_with_huggingface_id(self):
        # Registered vllm/ prefix stripped; HF org/model kept intact.
        assert (
            strip_provider_prefix("vllm/Qwen/Qwen3.5-35B-A3B") == "Qwen/Qwen3.5-35B-A3B"
        )


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
        """Every registered provider with a fixed base_url should result in
        BENCHFLOW_PROVIDER_BASE_URL being injectable by the SDK.
        Providers with user-supplied URLs (e.g. vllm) are excluded."""
        for name, cfg in PROVIDERS.items():
            if cfg.auth_type == "none":
                continue  # user-supplied base_url (e.g. local inference servers)
            assert cfg.base_url, f"Provider {name!r} has no base_url"
            assert cfg.api_protocol, f"Provider {name!r} has no api_protocol"


# ── Shim helper functions ──


class TestInferProviderPrefix:
    """Tests for _infer_provider_prefix() in the openclaw ACP shim."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from benchflow.agents.openclaw_acp_shim import _infer_provider_prefix

        self.infer = _infer_provider_prefix

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gpt-4o", "openai"),
            ("gpt-4o-mini", "openai"),
            ("o1-preview", "openai"),
            ("o3-mini", "openai"),
            ("gemini-3-flash", "google"),
            ("gemini-2.5-pro", "google"),
            ("claude-sonnet-4-6", "anthropic"),
            ("claude-haiku-4-5-20251001", "anthropic"),
            ("some-unknown-model", "anthropic"),  # default
        ],
    )
    def test_infer(self, model, expected):
        assert self.infer(model) == expected


class TestSetupOpenaiAuth:
    """Tests for setup_openai_auth() writing to openclaw's auth-profiles.json."""

    @pytest.fixture()
    def home_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        return tmp_path

    def _auth_path(self, home_dir):
        return (
            home_dir / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"
        )

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
        """session/set_model handler should map all three env vars.

        Parses the AST of openclaw_acp_shim.py to find the _PARAM_MAP dict
        literal and assert its contents directly (not via source-text grep).
        """
        import ast
        from pathlib import Path

        shim_path = (
            Path(__file__).parent.parent / "src/benchflow/agents/openclaw_acp_shim.py"
        )
        tree = ast.parse(shim_path.read_text())

        param_map: dict[str, str] | None = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_PARAM_MAP"
                and isinstance(node.value, ast.Dict)
            ):
                param_map = {
                    ast.literal_eval(k): ast.literal_eval(v)
                    for k, v in zip(node.value.keys, node.value.values, strict=True)
                }
                break

        assert param_map is not None, "_PARAM_MAP not found in openclaw_acp_shim.py"
        assert param_map == {
            "BENCHFLOW_MODEL_TEMPERATURE": "agents.defaults.params.temperature",
            "BENCHFLOW_MODEL_TOP_P": "agents.defaults.params.topP",
            "BENCHFLOW_MODEL_MAX_TOKENS": "agents.defaults.params.maxTokens",
        }
