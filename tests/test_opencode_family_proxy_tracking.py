"""Regression: OpenCode-family agents (opencode + its MiMo fork) support
LLM-proxy trajectory tracking.

These agents use acp_model_format="provider/model" and validate model ids
against the models.dev catalog, so they reject BenchFlow's synthetic gateway
model ``openai/benchflow-<alias>`` (ProviderModelNotFoundError) and never route
through the LiteLLM usage proxy — producing no ``trajectory/llm_trajectory.jsonl``
(which ``benchflow-experiment-review`` requires). The fix installs a ``-proxy``
wrapper under ``/opt/benchflow/bin`` that, in proxy mode, registers the gateway
alias under the agent's ``openai`` provider before exec'ing the isolated binary.
"""

import base64
import re

import pytest

from benchflow.agents.registry import AGENTS

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
def test_proxy_wrapper_registers_gateway_alias_under_openai_provider(
    agent, wrapper_bin, cfg
):
    w = _wrapper_script(agent, wrapper_bin)
    assert "BENCHFLOW_LITELLM_MODEL_ALIAS" in w  # gated on proxy mode
    assert cfg in w  # writes the agent's config
    for token in ("provider", "openai", "models"):
        assert token in w, f"{agent} wrapper missing {token!r}"


@pytest.mark.parametrize("agent,wrapper_bin,cfg", CASES)
def test_proxy_wrapper_wires_gateway_base_url(agent, wrapper_bin, cfg):
    w = _wrapper_script(agent, wrapper_bin)
    assert "OPENAI_BASE_URL" in w
    assert "baseURL" in w


@pytest.mark.parametrize("agent,wrapper_bin,cfg", CASES)
def test_proxy_wrapper_is_conditional_and_execs_isolated_binary(agent, wrapper_bin, cfg):
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
