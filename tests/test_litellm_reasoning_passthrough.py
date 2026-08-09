"""Gateway must forward agent-sent reasoning params on completions routes.

Live capture (2026-08-08, proxy between the gateway and api.deepseek.com):
prime-agent sent ``thinking: {"type": "enabled"}`` + ``reasoning_effort: high``
and the upstream received neither — LiteLLM's global ``drop_params: True``
strips non-vanilla-OpenAI fields on ``openai/`` passthrough routes, so every
gateway-routed run silently used the provider's default thinking config while
a native run used the agent's. The route now allowlists the reasoning params;
an upstream that does not support them rejects them exactly as it would for a
native client, which is the parity-correct behavior.
"""

from __future__ import annotations

import pytest

from benchflow.providers.litellm_config import resolve_litellm_route


def test_deepseek_routes_via_native_litellm_provider():
    """deepseek/<model>, not openai/<model>: LiteLLM's deepseek integration
    declares thinking + reasoning_effort supported, so drop_params keeps them;
    the openai/ passthrough drops both (the pinned 1.89.0 has no
    allowed_openai_params escape)."""
    env = {
        "DEEPSEEK_API_KEY": "sk-test",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
    }
    route = resolve_litellm_route("deepseek/deepseek-v4-flash", env)
    assert route.upstream_model == "deepseek/deepseek-v4-flash"
    assert route.litellm_params["model"] == "deepseek/deepseek-v4-flash"


def test_deepseek_native_route_honors_base_url_override():
    """A custom DEEPSEEK_BASE_URL (mock/capture endpoints in parity runs) must
    survive the native-provider routing."""
    env = {
        "DEEPSEEK_API_KEY": "sk-test",
        "DEEPSEEK_BASE_URL": "http://127.0.0.1:11500/v1",
    }
    route = resolve_litellm_route("deepseek/deepseek-v4-flash", env)
    assert route.upstream_model == "deepseek/deepseek-v4-flash"
    assert route.litellm_params.get("api_base") == "http://127.0.0.1:11500/v1"


def test_pinned_litellm_deepseek_provider_supports_reasoning_params():
    """Guards the assumption the native-provider routing rests on: if a LiteLLM
    upgrade stops declaring these params supported for deepseek, this fails
    before a silent re-drop ships."""
    pytest.importorskip("litellm")
    from litellm.utils import get_supported_openai_params

    supported = (
        get_supported_openai_params(
            model="deepseek-chat", custom_llm_provider="deepseek"
        )
        or []
    )
    assert "thinking" in supported
    assert "reasoning_effort" in supported
