"""Core provider-registry coverage retained after OpenClaw shim extraction."""

import importlib.util
import os
from pathlib import Path

import pytest

from benchflow.agents.providers import find_provider, find_provider_for_bare_model


@pytest.fixture
def paired_shim():
    """Load PR A source for paired compatibility-seam tests when available."""
    path = (
        Path(os.environ.get("BENCHFLOW_AGENTS_SOURCE", ""))
        / "acp"
        / "openclaw"
        / "openclaw_acp_shim.py"
    )
    if not path.is_file():
        pytest.skip("paired local OpenClaw shim unavailable")
    spec = importlib.util.spec_from_file_location("paired_openclaw_shim", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFindProviderForBareModel:
    """Bare model id maps to provider without shim-owned behavior."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("deepseek-v4-flash", "deepseek"),
            ("glm-5.1", "glm"),
            ("qwen3.6-max-preview", "qwen-dashscope"),
            ("kimi-k2.6", "kimi"),
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

    def test_longest_token_wins(self):
        assert find_provider_for_bare_model("doubao-seed-2-pro-251015")[0] == (
            "doubao-seed-2-pro"
        )

    @pytest.mark.parametrize(
        "model", ["", "   ", "whatever-7b", "glmnext-9b", "claude-sonnet-4-6"]
    )
    def test_unknown_or_empty_returns_none(self, model):
        assert find_provider_for_bare_model(model) is None

    def test_prefixed_input_defers_to_find_provider(self):
        model = "deepseek/deepseek-v4-flash"
        assert find_provider_for_bare_model(model) is None
        assert find_provider(model)[0] == "deepseek"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("deepseek-v4-flash", "deepseek"),
        ("glm-5.1", "glm"),
        ("qwen3.6-max-preview", "qwen-dashscope"),
        ("gemini-3.1-flash-lite", "google"),
        ("gpt-4o", "openai"),
        ("claude-sonnet-4-6", "anthropic"),
    ],
)
def test_paired_shim_uses_core_provider_registry(paired_shim, model, expected):
    """Guards PR #670 compatibility seam after PR B for issue #1090."""
    assert paired_shim._infer_provider_prefix(model) == expected


def test_paired_shim_injects_core_provider_config(paired_shim, monkeypatch, tmp_path):
    """Guards PR #670 custom-provider setup after PR B for issue #1090."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://proxy.invalid/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    assert paired_shim._resolve_bare_model_prefix("deepseek-v4-flash") == "deepseek"
    assert (tmp_path / ".openclaw" / "openclaw.json").is_file()
