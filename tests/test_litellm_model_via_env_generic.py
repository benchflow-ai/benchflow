"""The via-env model flag is derived from registration data, not agent names.

_wire_litellm_agent_env historically set BENCHFLOW_LITELLM_MODEL_VIA_ENV=1 only
in hardcoded per-agent branches (codex-acp / openhands / claude-agent-acp), so
generic manifest agents that deliver the model via env (supports_acp_set_model
= false + a BENCHFLOW_PROVIDER_MODEL env mapping) never got it — and benchflow
then drove their capability-advertised ``model`` config option, which agents
like qwen-code reject (-32603). These lock the generic derivation.
"""

from __future__ import annotations

import pytest

from benchflow.agents.registry import AGENT_INSTALLERS, AGENT_LAUNCH, AGENTS
from benchflow.agents.registry import AgentConfig as _AC
from benchflow.providers.litellm_config import (
    LITELLM_MODEL_VIA_ENV,
    resolve_litellm_route,
)
from benchflow.providers.litellm_runtime import _wire_litellm_agent_env


@pytest.fixture()
def _route():
    return resolve_litellm_route(
        "deepseek/deepseek-v4-flash", {"DEEPSEEK_API_KEY": "k"}
    )


def _register(name: str, **kw):
    AGENTS[name] = _AC(name=name, install_cmd="true", launch_cmd="true", **kw)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    for n in ("via-env-probe", "acp-model-probe", "no-mapping-probe"):
        AGENTS.pop(n, None)
        AGENT_INSTALLERS.pop(n, None)
        AGENT_LAUNCH.pop(n, None)


def test_env_owned_registration_gets_via_env_flag(_route):
    _register(
        "via-env-probe",
        supports_acp_set_model=False,
        env_mapping={"BENCHFLOW_PROVIDER_MODEL": "OPENAI_MODEL"},
    )
    updated = _wire_litellm_agent_env(
        agent="via-env-probe",
        agent_env={},
        route=_route,
        base_url="http://127.0.0.1:4000",
        master_key="sk-master",
    )
    assert updated[LITELLM_MODEL_VIA_ENV] == "1"


def test_acp_configured_agent_does_not_get_flag(_route):
    """supports_acp_set_model=True (default) keeps ACP-driven model config."""
    _register(
        "acp-model-probe",
        env_mapping={"BENCHFLOW_PROVIDER_MODEL": "OPENAI_MODEL"},
    )
    updated = _wire_litellm_agent_env(
        agent="acp-model-probe",
        agent_env={},
        route=_route,
        base_url="http://127.0.0.1:4000",
        master_key="sk-master",
    )
    assert LITELLM_MODEL_VIA_ENV not in updated


def test_no_model_mapping_does_not_get_flag(_route):
    """Without a model env mapping the agent cannot receive the model via env —
    ACP configuration must stay on (else it falls back to its own default)."""
    _register("no-mapping-probe", supports_acp_set_model=False)
    updated = _wire_litellm_agent_env(
        agent="no-mapping-probe",
        agent_env={},
        route=_route,
        base_url="http://127.0.0.1:4000",
        master_key="sk-master",
    )
    assert LITELLM_MODEL_VIA_ENV not in updated
