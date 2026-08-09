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

from benchflow.providers.litellm_config import resolve_litellm_route


def test_deepseek_route_allows_reasoning_params():
    env = {
        "DEEPSEEK_API_KEY": "sk-test",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
    }
    route = resolve_litellm_route("deepseek/deepseek-v4-flash", env)
    allowed = route.litellm_params.get("allowed_openai_params")
    assert allowed is not None
    assert "thinking" in allowed
    assert "reasoning_effort" in allowed
    assert route.upstream_model.startswith("openai/")


def test_anthropic_messages_route_does_not_get_openai_allowlist():
    env = {"ANTHROPIC_API_KEY": "sk-test"}
    route = resolve_litellm_route("anthropic/claude-sonnet-4-5", env)
    assert "allowed_openai_params" not in route.litellm_params
