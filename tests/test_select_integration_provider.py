"""Unit tests for the integration-eval provider selector (ENG-265)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "select_integration_provider",
    Path(__file__).resolve().parents[1] / ".github" / "scripts" / "select_integration_provider.py",
)
sel = importlib.util.module_from_spec(_SPEC)
sys.modules["select_integration_provider"] = sel
assert _SPEC.loader is not None
_SPEC.loader.exec_module(sel)


def test_selects_first_credentialed_provider():
    choice = sel.select_provider({"DEEPSEEK_API_KEY": "sk-x"})
    assert choice.provider == "deepseek"
    assert choice.agent == "openhands"
    assert choice.model == "deepseek/deepseek-v4-flash"
    # judge reuses the provider's model so no extra credential is needed
    assert choice.judge_model == choice.model


def test_order_is_deterministic_first_wins():
    choice = sel.select_provider({"OPENAI_API_KEY": "sk-o", "DEEPSEEK_API_KEY": "sk-d"})
    assert choice.provider == "deepseek"  # deepseek precedes openai


def test_litellm_requires_base_url():
    # key alone is insufficient for litellm; base url also required
    with pytest.raises(sel.NoProviderAvailable):
        sel.select_provider({"LITELLM_API_KEY": "sk-l"})
    choice = sel.select_provider(
        {"LITELLM_API_KEY": "sk-l", "LITELLM_BASE_URL": "http://x/v1"}
    )
    assert choice.provider == "litellm"


def test_hard_fails_with_no_provider():
    with pytest.raises(sel.NoProviderAvailable, match="no integration provider"):
        sel.select_provider({})
