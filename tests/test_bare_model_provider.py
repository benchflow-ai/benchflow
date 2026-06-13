"""BF-4: bare (prefix-stripped) model ids resolve to the right provider.

Guards the fix from benchflow-ai/benchflow PR #670 (BF-4) against the
regression where bare custom-provider ids defaulted to anthropic.

``find_provider`` only matches an explicit ``provider/`` prefix, so after
``strip_provider_prefix`` runs, a bare id like ``deepseek-v4-flash`` no longer
resolves and the openclaw shim's ``_infer_provider_prefix`` historically
defaulted everything that was not gemini/gpt to ``anthropic`` — silently
running deepseek/glm/qwen/... as anthropic.

These tests pin the registry-driven bare-model routing:
  - ``find_provider_for_bare_model`` maps a bare id to its provider via each
    provider's declared ``model_prefixes`` (registry owns the knowledge).
  - ``_infer_provider_prefix`` consults that helper before its native
    gemini/gpt heuristics, and still falls back to anthropic.
  - ``_setup_bare_custom_provider`` (Codex P1 follow-up on PR #670) actually
    registers the resolved custom provider in openclaw.json before the bare id
    is prefixed, while openclaw-native and unknown ids trigger no setup.
  - ``_resolve_bare_model_prefix`` (Codex P2 follow-up on PR #670) falls back
    to the generic ``BENCHFLOW_PROVIDER_*`` env setup when the registry config
    is unresolvable, instead of prefixing an unconfigured provider.
"""

import pytest

from benchflow.agents.openclaw_acp_shim import (
    _infer_provider_prefix,
    _resolve_bare_model_prefix,
    _setup_bare_custom_provider,
)
from benchflow.agents.providers import (
    find_provider,
    find_provider_for_bare_model,
)


class TestFindProviderForBareModel:
    """Registry helper: bare model id -> (provider_name, config)."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("deepseek-v4-flash", "deepseek"),
            ("deepseek-v4-pro", "deepseek"),
            ("glm-4.6", "glm"),
            ("glm-5.1", "glm"),
            ("qwen3.6-max-preview", "qwen-dashscope"),  # version-suffixed, no hyphen
            ("qwen-max", "qwen-dashscope"),
            ("kimi-k2.6", "kimi"),
            ("moonshot-v1-8k", "kimi"),
            ("minimax-m2.7", "minimax"),
            ("mimo-v2.5-pro", "xiaomi"),
            ("hunyuan-turbo", "hunyuan"),
        ],
    )
    def test_known_families_resolve(self, model, expected):
        result = find_provider_for_bare_model(model)
        assert result is not None, f"{model!r} did not resolve"
        assert result[0] == expected

    def test_case_insensitive(self):
        assert find_provider_for_bare_model("DeepSeek-V4-Flash")[0] == "deepseek"

    def test_longest_token_wins_for_doubao(self):
        """doubao-seed-2-pro/-lite carry full family tokens; the longer wins."""
        assert (
            find_provider_for_bare_model("doubao-seed-2-pro-251015")[0]
            == "doubao-seed-2-pro"
        )
        assert (
            find_provider_for_bare_model("doubao-seed-2-lite-251015")[0]
            == "doubao-seed-2-lite"
        )

    def test_token_requires_family_boundary(self):
        """A different word that merely starts with a token must NOT match."""
        assert find_provider_for_bare_model("glmnext-9b") is None
        assert find_provider_for_bare_model("deepseekish-1b") is None

    def test_unknown_model_returns_none(self):
        assert find_provider_for_bare_model("whatever-7b") is None
        assert find_provider_for_bare_model("claude-sonnet-4-6") is None
        assert find_provider_for_bare_model("gpt-4o") is None
        assert find_provider_for_bare_model("gemini-3.1-flash-lite") is None

    def test_prefixed_input_defers_to_find_provider(self):
        """Inputs still carrying a registered provider/ prefix return None here."""
        assert find_provider_for_bare_model("deepseek/deepseek-v4-flash") is None
        assert find_provider_for_bare_model("zai/glm-5") is None
        # Sanity: those DO resolve via the prefix-based find_provider.
        assert find_provider("deepseek/deepseek-v4-flash")[0] == "deepseek"

    def test_empty_input_returns_none(self):
        assert find_provider_for_bare_model("") is None
        assert find_provider_for_bare_model("   ") is None


class TestInferProviderPrefixRegistry:
    """_infer_provider_prefix consults the registry, then heuristics, then anthropic."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            # BF-4 fix: bare custom-provider ids route via the registry.
            ("deepseek-v4-flash", "deepseek"),
            ("glm-4.6", "glm"),
            ("qwen3.6-max-preview", "qwen-dashscope"),
            ("minimax-m2.7", "minimax"),
            # Native heuristics unchanged.
            ("gemini-3.1-flash-lite", "google"),
            ("gemini-2.5-pro", "google"),
            ("gpt-4o", "openai"),
            ("o1-preview", "openai"),
            ("o3-mini", "openai"),
            # Anthropic stays the default for genuinely unknown ids.
            ("whatever-7b", "anthropic"),
            ("claude-sonnet-4-6", "anthropic"),
            ("claude-haiku-4-5-20251001", "anthropic"),
        ],
    )
    def test_infer(self, model, expected):
        assert _infer_provider_prefix(model) == expected


class TestSetupBareCustomProvider:
    """Codex P1 (PR #670): bare custom ids must REGISTER their provider.

    ``_infer_provider_prefix`` only names the prefix; for a registered custom
    provider the shim must also write the provider into ``openclaw.json``, or
    openclaw receives a ``deepseek/...`` id pointing at a provider it never
    learned about and the run fails. These tests spy on ``setup_custom_provider``
    (the openclaw.json writer) to prove the bare custom path triggers setup and
    that openclaw-native / unknown ids do NOT.
    """

    def test_bare_custom_model_registers_provider(self, monkeypatch):
        """deepseek-v4-flash → setup_custom_provider called with deepseek config."""
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.test/v1")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")

        calls = []
        monkeypatch.setattr(
            "benchflow.agents.openclaw_acp_shim.setup_custom_provider",
            lambda *a, **k: calls.append((a, k)),
        )

        provider = _setup_bare_custom_provider("deepseek-v4-flash")

        assert provider == "deepseek"
        assert len(calls) == 1
        args, _ = calls[0]
        # setup_custom_provider(provider_name, base_url, api_key, api_protocol, models)
        assert args[0] == "deepseek"
        assert args[1] == "https://api.deepseek.test/v1"
        assert args[2] == "sk-deepseek-test"

    def test_bare_custom_model_uses_registry_endpoint(self, monkeypatch):
        """glm-4.6 routes to the glm provider with its own env-supplied endpoint."""
        monkeypatch.setenv("GLM_BASE_URL", "https://glm.test/v1")
        monkeypatch.setenv("GLM_API_KEY", "glm-test-key")

        calls = []
        monkeypatch.setattr(
            "benchflow.agents.openclaw_acp_shim.setup_custom_provider",
            lambda *a, **k: calls.append(a),
        )

        assert _setup_bare_custom_provider("glm-4.6") == "glm"
        assert len(calls) == 1
        assert calls[0][0] == "glm"

    @pytest.mark.parametrize(
        "model",
        [
            "gemini-3.1-flash-lite",
            "gpt-4o",
            "o3-mini",
            "claude-sonnet-4-6",
            "whatever-7b",
        ],
    )
    def test_native_and_unknown_models_do_not_register(self, model, monkeypatch):
        """openclaw-native (gemini/gpt/claude) and unknown ids trigger NO setup."""
        calls = []
        monkeypatch.setattr(
            "benchflow.agents.openclaw_acp_shim.setup_custom_provider",
            lambda *a, **k: calls.append(a),
        )

        assert _setup_bare_custom_provider(model) is None
        assert calls == []

    def test_missing_config_env_does_not_register(self, monkeypatch):
        """A resolved custom provider with unset url_params/key registers nothing."""
        # Ensure the deepseek config env vars are absent.
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        calls = []
        monkeypatch.setattr(
            "benchflow.agents.openclaw_acp_shim.setup_custom_provider",
            lambda *a, **k: calls.append(a),
        )

        # Resolves to deepseek in the registry, but can't be configured → None.
        assert _setup_bare_custom_provider("deepseek-v4-flash") is None
        assert calls == []


class TestResolveBareModelPrefix:
    """Codex P2 (PR #670): generic env fallback before prefixing bare ids.

    Guards the fix from benchflow-ai/benchflow PR #670 against the regression
    where a bare custom id whose registry config could not resolve (e.g.
    DEEPSEEK_BASE_URL/DEEPSEEK_API_KEY unset) was prefixed via
    ``_infer_provider_prefix`` anyway — handing openclaw ``deepseek/<model>``
    for a provider never written to openclaw.json — even though the generic
    ``BENCHFLOW_PROVIDER_BASE_URL``/``BENCHFLOW_PROVIDER_API_KEY`` envs could
    have configured a working provider via ``_find_and_setup_provider``.
    """

    GENERIC_ENVS = (
        "BENCHFLOW_PROVIDER_BASE_URL",
        "BENCHFLOW_PROVIDER_API_KEY",
        "BENCHFLOW_PROVIDER_PROTOCOL",
        "BENCHFLOW_PROVIDER_MODELS",
    )

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Start from no provider config; individual tests opt back in."""
        for var in (*self.GENERIC_ENVS, "DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY"):
            monkeypatch.delenv(var, raising=False)

    @pytest.fixture
    def setup_spy(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "benchflow.agents.openclaw_acp_shim.setup_custom_provider",
            lambda *a, **k: calls.append(a),
        )
        return calls

    def test_generic_envs_used_when_registry_config_unresolvable(
        self, monkeypatch, setup_spy
    ):
        """Registry names deepseek but its envs are unset → generic setup runs."""
        monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "https://proxy.test/v1")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_API_KEY", "sk-generic-test")

        assert _resolve_bare_model_prefix("deepseek-v4-flash") == "custom"
        assert len(setup_spy) == 1
        # setup_custom_provider(provider_name, base_url, api_key, protocol, models)
        assert setup_spy[0][:3] == (
            "custom",
            "https://proxy.test/v1",
            "sk-generic-test",
        )

    def test_registry_config_wins_over_generic_envs(self, monkeypatch, setup_spy):
        """Provider-specific envs take precedence over the generic fallback."""
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.test/v1")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "https://proxy.test/v1")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_API_KEY", "sk-generic-test")

        assert _resolve_bare_model_prefix("deepseek-v4-flash") == "deepseek"
        assert len(setup_spy) == 1
        assert setup_spy[0][0] == "deepseek"

    def test_no_config_anywhere_prefixes_without_registration(self, setup_spy):
        """With neither registry nor generic envs, keep the inferred prefix
        and register nothing (the run cannot work without a key anyway)."""
        assert _resolve_bare_model_prefix("deepseek-v4-flash") == "deepseek"
        assert setup_spy == []

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gemini-3.1-flash-lite", "google"),
            ("gpt-4o", "openai"),
            ("o3-mini", "openai"),
            ("claude-sonnet-4-6", "anthropic"),
            ("whatever-7b", "anthropic"),
        ],
    )
    def test_native_and_unknown_ids_unchanged(self, model, expected, setup_spy):
        """Without generic envs, builtin/unknown ids resolve exactly as before."""
        assert _resolve_bare_model_prefix(model) == expected
        assert setup_spy == []
