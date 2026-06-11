"""BF-4: bare (prefix-stripped) model ids resolve to the right provider.

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
"""

import pytest

from benchflow.agents.openclaw_acp_shim import _infer_provider_prefix
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
