"""Regression: OpenCode-family agents (opencode + its MiMo fork) support
LLM-proxy trajectory tracking.

These agents use acp_model_format="provider/model" and hard-code the OpenAI
Responses API for the built-in ``openai`` provider id, which the LiteLLM gateway
(chat-completions only) cannot serve — so the gateway alias never routes through
the usage proxy and no ``trajectory/llm_trajectory.jsonl`` is written (which
``benchflow-experiment-review`` requires). The fix installs a ``-proxy`` wrapper
under ``/opt/benchflow/bin`` that, in proxy mode, registers the gateway alias
under a dedicated ``@ai-sdk/openai-compatible`` provider id
(``OPENCODE_PROXY_PROVIDER_ID``) before exec'ing the isolated binary.
"""

import base64
import re

import pytest

from benchflow.agents.registry import AGENTS, OPENCODE_PROXY_PROVIDER_ID

# (agent name, proxy wrapper binary, agent config filename)
CASES = [
    ("opencode", "opencode-proxy", "opencode.json"),
    ("mimo", "mimo-proxy", "mimocode.json"),
]


def _wrapper_script(agent: str, wrapper_bin: str) -> str:
    ic = AGENTS[agent].install_cmd
    m = re.search(
        r"printf '%s' '([A-Za-z0-9+/=]+)' \| base64 -d > \S*/" + re.escape(wrapper_bin),
        ic,
    )
    assert m, f"{wrapper_bin} wrapper install snippet not found in {agent} install_cmd"
    return base64.b64decode(m.group(1)).decode()


@pytest.mark.parametrize("agent,wrapper_bin,cfg", CASES)
def test_proxy_wrapper_registers_gateway_alias_under_dedicated_provider(
    agent, wrapper_bin, cfg
):
    w = _wrapper_script(agent, wrapper_bin)
    assert "BENCHFLOW_LITELLM_MODEL_ALIAS" in w  # gated on proxy mode
    assert cfg in w  # writes the agent's config
    for token in ("provider", OPENCODE_PROXY_PROVIDER_ID, "models"):
        assert token in w, f"{agent} wrapper missing {token!r}"
    # Must NOT touch the built-in ``openai`` provider id — OpenCode hard-codes
    # the Responses API there, which the gateway cannot serve.
    assert '"openai"' not in w and "'openai'" not in w


@pytest.mark.parametrize("agent,wrapper_bin,cfg", CASES)
def test_proxy_wrapper_wires_gateway_base_url(agent, wrapper_bin, cfg):
    w = _wrapper_script(agent, wrapper_bin)
    assert "OPENAI_BASE_URL" in w
    assert "baseURL" in w


@pytest.mark.parametrize("agent,wrapper_bin,cfg", CASES)
def test_proxy_wrapper_forces_chat_completions_sdk(agent, wrapper_bin, cfg):
    """The dedicated provider must use ``@ai-sdk/openai-compatible`` (chat
    completions) so OpenCode does not call ``provider.responses()`` (which the
    gateway/DeepSeek do not serve), and forward the gateway master key as
    ``apiKey``."""
    w = _wrapper_script(agent, wrapper_bin)
    assert "@ai-sdk/openai-compatible" in w  # chat completions, not Responses API
    assert '"npm"' in w
    assert "OPENAI_API_KEY" in w and "apiKey" in w


def test_format_acp_model_routes_proxy_alias_to_dedicated_provider():
    """In proxy mode the gateway alias (``benchflow-…``) must be sent to
    set_model as ``<OPENCODE_PROXY_PROVIDER_ID>/<alias>`` for provider/model
    agents — NOT ``openai/<alias>`` (which crashes OpenCode's Responses path)."""
    from benchflow.acp.runtime import _format_acp_model

    alias = "benchflow-deepseek-deepseek-v4-flash"
    for agent in ("opencode", "mimo"):
        assert (
            _format_acp_model(alias, agent) == f"{OPENCODE_PROXY_PROVIDER_ID}/{alias}"
        )
        assert _format_acp_model(alias, agent).split("/")[0] != "openai"


@pytest.mark.parametrize("agent,wrapper_bin,cfg", CASES)
def test_proxy_wrapper_pins_small_model_to_gateway_alias(agent, wrapper_bin, cfg):
    """The title/summary helper must not fall back to the hard-coded
    ``gpt-5-nano`` (unservable by the gateway) — it is pinned to the alias."""
    w = _wrapper_script(agent, wrapper_bin)
    assert "small_model" in w


@pytest.mark.parametrize("agent,wrapper_bin,cfg", CASES)
def test_proxy_wrapper_is_conditional_and_execs_isolated_binary(
    agent, wrapper_bin, cfg
):
    w = _wrapper_script(agent, wrapper_bin)
    assert '[ -n "$BENCHFLOW_LITELLM_MODEL_ALIAS" ]' in w  # no-op without alias
    base = wrapper_bin[: -len("-proxy")]
    assert f"exec /opt/benchflow/bin/{base} " in w  # execs the isolated binary


@pytest.mark.parametrize("agent,wrapper_bin,cfg", CASES)
def test_launch_entrypoint_stays_under_isolated_bin(agent, wrapper_bin, cfg):
    lc = AGENTS[agent].launch_cmd
    # keeps the JS-ACP isolated-runtime invariant (launch entrypoint under
    # /opt/benchflow/bin)
    assert lc.split()[0] == f"/opt/benchflow/bin/{wrapper_bin}"
    assert "acp" in lc


def test_wrapper_runs_native_binary_directly_not_via_node():
    """opencode-ai ships its bin as a native ELF (bin/opencode.exe); the wrapper
    must detect non-shebang bins and exec them directly (running an ELF via
    `node` crashes with a SyntaxError at startup)."""
    w = _wrapper_script("opencode", "opencode-proxy")
    assert "head -c2" in w  # detect shebang vs native
    assert "/opt/benchflow/js-agents/bin/opencode" in w  # direct native-exec path
