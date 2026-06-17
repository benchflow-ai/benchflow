"""Regression: pi-acp (acp_model_format="registered-provider/model") must route
through the LiteLLM usage proxy in proxy mode.

In proxy mode BenchFlow serves the model under the alias `benchflow-<...>` and the
pi-acp launcher registers it under the `litellm` provider (BENCHFLOW_PROVIDER_NAME)
in `~/.pi/agent/models.json`. If `session/set_model` sends the *bare* alias (no
provider prefix), Pi cannot resolve it to that provider and the model calls
**bypass the proxy** — completing the task (tool calls happen) but capturing no
`trajectory/llm_trajectory.jsonl` (usage_source=unavailable). The fix sends the
registered-provider-qualified id `litellm/<alias>`.
"""

from benchflow.acp.runtime import _resolve_acp_model_input, _select_acp_model_id

_PROXY = {
    "BENCHFLOW_LITELLM_MODEL_ALIAS": "benchflow-deepseek-deepseek-v4-flash",
    "BENCHFLOW_PROVIDER_NAME": "litellm",
}


def _final_id(agent, model, env):
    return _select_acp_model_id(_resolve_acp_model_input(agent, model, env), agent, None)


def test_pi_acp_proxy_alias_is_litellm_provider_qualified():
    assert (
        _final_id("pi-acp", "deepseek/deepseek-v4-flash", _PROXY)
        == "litellm/benchflow-deepseek-deepseek-v4-flash"
    )


def test_pi_acp_non_proxy_keeps_registered_provider_behavior():
    # No alias → not proxy mode → must NOT be litellm-prefixed.
    out = _final_id("pi-acp", "anthropic/claude-sonnet-4-6", {})
    assert not out.startswith("litellm/")


def test_provider_model_agents_unaffected_by_pi_acp_fix():
    # opencode (provider/model) routes via the dedicated OpenCode gateway
    # provider id, NOT litellm/ (pi-acp) and NOT openai/ (Responses-API crash).
    out = _final_id("opencode", "deepseek/deepseek-v4-flash", _PROXY)
    assert out == "benchflow/benchflow-deepseek-deepseek-v4-flash"
    assert not out.startswith("litellm/")
    assert not out.startswith("openai/")
