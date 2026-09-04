"""Bare model IDs resolve through generic provider-registry metadata."""

import pytest

from benchflow.agents.providers import find_provider, find_provider_for_bare_model


class TestFindProviderForBareModel:
    """Bare model id maps to provider without shim-owned behavior."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("deepseek-v4-flash", "deepseek"),
            ("deepseek-v4-pro", "deepseek"),
            ("glm-4.6", "glm"),
            ("glm-5.1", "glm"),
            ("qwen3.6-max-preview", "qwen-dashscope"),
            ("qwen-max", "qwen-dashscope"),
            ("kimi-k2.6", "kimi"),
            ("moonshot-v1-8k", "kimi"),
            ("minimax-m2.7", "minimax"),
            ("mimo-v2.5", "xiaomi"),
            ("hunyuan-turbo", "hunyuan"),
        ],
    )
    def test_known_families_resolve(self, model, expected):
        """Guards PR #670 provider routing after PR B for issue #1090."""
        result = find_provider_for_bare_model(model)
        assert result is not None
        assert result[0] == expected

    def test_case_insensitive(self):
        assert find_provider_for_bare_model("DeepSeek-V4-Flash")[0] == "deepseek"

    def test_longest_token_wins_for_doubao(self):
        assert find_provider_for_bare_model("doubao-seed-2-pro-251015")[0] == (
            "doubao-seed-2-pro"
        )
        assert find_provider_for_bare_model("doubao-seed-2-lite-251015")[0] == (
            "doubao-seed-2-lite"
        )

    def test_token_requires_family_boundary(self):
        assert find_provider_for_bare_model("glmnext-9b") is None
        assert find_provider_for_bare_model("deepseekish-1b") is None

    @pytest.mark.parametrize(
        "model",
        [
            "",
            "   ",
            "whatever-7b",
            "claude-sonnet-4-6",
            "gpt-4o",
            "gemini-3.1-flash-lite",
        ],
    )
    def test_unknown_or_empty_returns_none(self, model):
        assert find_provider_for_bare_model(model) is None

    def test_prefixed_input_defers_to_find_provider(self):
        for model in ("deepseek/deepseek-v4-flash", "zai/glm-5"):
            assert find_provider_for_bare_model(model) is None
        assert find_provider("deepseek/deepseek-v4-flash")[0] == "deepseek"
        assert find_provider("zai/glm-5")[0] == "zai"
