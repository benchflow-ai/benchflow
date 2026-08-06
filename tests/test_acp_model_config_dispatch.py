"""Capability-first ACP model/effort dispatch (``connect_acp``).

These cover the runtime behavior that lets the ``@agentclientprotocol`` family
migrate ``session/set_model`` → a ``"model"`` config option with no registry
change: the agent's advertised ``session/new`` config options drive the
dispatch, and the registry's ``acp_model_config_id`` is only an override/hint.

The broader connect_acp model-id formatting cases live in
``tests/test_acp.py::TestConnectAcpModelSelection``; the fail-closed lifecycle
cases live in ``tests/test_acp_setup_failure_propagation.py``.
"""

import contextlib
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from benchflow.acp.client import ACPClient
from benchflow.providers.litellm_config import (
    LITELLM_MODEL_ALIAS_ENV,
    LITELLM_MODEL_VIA_ENV,
)


def _make_mocks(config_options=None, model_state=None):
    mock_session = MagicMock()
    mock_session.session_id = "s1"
    mock_session.config_options = [] if config_options is None else config_options
    mock_session.model_state = model_state
    mock_init = MagicMock()
    mock_init.agent_info = None

    mock_acp = AsyncMock(spec=ACPClient)
    mock_acp.connect = AsyncMock()
    mock_acp.initialize = AsyncMock(return_value=mock_init)
    mock_acp.session_new = AsyncMock(return_value=mock_session)
    mock_acp.set_model = AsyncMock()
    mock_acp.set_config_option = AsyncMock()
    mock_acp.close = AsyncMock()
    return mock_acp


@contextlib.contextmanager
def _runtime_patches(mock_acp):
    with (
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
    ):
        yield


async def _connect(
    mock_acp, *, agent, model, tmp_path, agent_env=None, reasoning_effort=None
):
    from benchflow.acp.runtime import connect_acp

    with _runtime_patches(mock_acp):
        await connect_acp(
            env=AsyncMock(),
            agent=agent,
            agent_launch=agent,
            agent_env={} if agent_env is None else agent_env,
            sandbox_user=None,
            model=model,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
            reasoning_effort=reasoning_effort,
        )


@pytest.mark.asyncio
async def test_codex_with_only_fastmode_option_uses_set_model(tmp_path):
    """codex-acp@0.0.45 advertises only 'fast-mode' (no 'model'), so dispatch
    must use session/set_model — capability-first must NOT regress it."""
    mock_acp = _make_mocks(config_options=[{"id": "fast-mode"}])
    await _connect(mock_acp, agent="codex-acp", model="gpt-5.5", tmp_path=tmp_path)

    mock_acp.set_model.assert_awaited_once_with("gpt-5.5")
    mock_acp.set_config_option.assert_not_awaited()


@pytest.mark.asyncio
async def test_codex_litellm_alias_uses_bare_model_for_set_model(tmp_path):
    """Codex validates set_model against its own model catalog, not proxy aliases.

    This guards against a false-green CI path where BenchFlow recorded the
    requested model but codex-acp fell back to its own default at request time.
    """
    mock_acp = _make_mocks(
        config_options=[{"id": "fast-mode"}],
        model_state={
            "availableModels": [
                {"modelId": "gpt-5.4-mini[low]"},
                {"modelId": "gpt-5.4-mini[medium]"},
            ],
            "currentModelId": "gpt-5.5[medium]",
        },
    )
    await _connect(
        mock_acp,
        agent="codex-acp",
        model="openai/gpt-5.4-mini",
        tmp_path=tmp_path,
        agent_env={
            "BENCHFLOW_PROVIDER_MODEL": "benchflow-openai-gpt-5.4-mini",
            LITELLM_MODEL_ALIAS_ENV: "benchflow-openai-gpt-5.4-mini",
            LITELLM_MODEL_VIA_ENV: "1",
        },
    )

    mock_acp.set_model.assert_awaited_once_with("gpt-5.4-mini[medium]")
    mock_acp.set_config_option.assert_not_awaited()


@pytest.mark.asyncio
async def test_codex_with_model_option_uses_config_option(tmp_path):
    """When a future codex-acp advertises a 'model' config option (as Claude
    already does), capability-first routes through it — no registry change."""
    mock_acp = _make_mocks(config_options=[{"id": "model"}])
    await _connect(mock_acp, agent="codex-acp", model="gpt-5.5", tmp_path=tmp_path)

    mock_acp.set_config_option.assert_awaited_once_with("model", "gpt-5.5")
    mock_acp.set_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_codex_model_option_does_not_receive_legacy_effort_suffix(tmp_path):
    """Guards ACP capability mapping against sending ``model[effort]`` to a
    current Codex ACP ``model`` config option."""
    mock_acp = _make_mocks(
        config_options=[{"id": "model"}],
        model_state={"currentModelId": "gpt-5.6-sol[medium]"},
    )
    await _connect(mock_acp, agent="codex-acp", model="gpt-5.6-sol", tmp_path=tmp_path)

    mock_acp.set_config_option.assert_awaited_once_with("model", "gpt-5.6-sol")
    mock_acp.set_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_codex_current_acp_uses_dedicated_effort_option(tmp_path):
    """Guards ACP capability mapping against dropping Codex's requested effort."""
    mock_acp = _make_mocks(config_options=[{"id": "model"}, {"id": "reasoning_effort"}])
    await _connect(
        mock_acp,
        agent="codex-acp",
        model="gpt-5.6-sol",
        tmp_path=tmp_path,
        reasoning_effort="high",
    )

    mock_acp.set_config_option.assert_has_awaits(
        [
            call("model", "gpt-5.6-sol"),
            call("reasoning_effort", "high"),
        ]
    )
    mock_acp.set_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_codex_legacy_acp_encodes_effort_in_model_id(tmp_path):
    """Guards ACP capability mapping for legacy Codex ``model[effort]`` IDs."""
    mock_acp = _make_mocks(
        config_options=[{"id": "fast-mode"}],
        model_state={
            "availableModels": [
                {"modelId": "gpt-5.5[medium]"},
                {"modelId": "gpt-5.5[high]"},
            ],
            "currentModelId": "gpt-5.5[medium]",
        },
    )
    await _connect(
        mock_acp,
        agent="codex-acp",
        model="gpt-5.5",
        tmp_path=tmp_path,
        reasoning_effort="high",
    )

    mock_acp.set_model.assert_awaited_once_with("gpt-5.5[high]")
    mock_acp.set_config_option.assert_not_awaited()


@pytest.mark.asyncio
async def test_codex_legacy_acp_rejects_unadvertised_effort(tmp_path):
    """Guards the ACP capability mapping fix against silently using a default."""
    mock_acp = _make_mocks(
        config_options=[{"id": "fast-mode"}],
        model_state={
            "availableModels": [{"modelId": "gpt-5.5[medium]"}],
            "currentModelId": "gpt-5.5[medium]",
        },
    )

    with pytest.raises(RuntimeError, match="does not advertise reasoning effort"):
        await _connect(
            mock_acp,
            agent="codex-acp",
            model="gpt-5.5",
            tmp_path=tmp_path,
            reasoning_effort="high",
        )

    mock_acp.set_model.assert_not_awaited()
    mock_acp.set_config_option.assert_not_awaited()


@pytest.mark.asyncio
async def test_pi_acp_uses_thought_level_and_maps_none_to_off(tmp_path):
    """Guards Pi ACP's distinct effort option and its ``off`` spelling."""
    mock_acp = _make_mocks(config_options=[{"id": "model"}, {"id": "thought_level"}])
    await _connect(
        mock_acp,
        agent="pi-acp",
        model="openai/gpt-5.6-sol",
        tmp_path=tmp_path,
        reasoning_effort="none",
    )

    mock_acp.set_config_option.assert_has_awaits(
        [
            call("model", "openai/gpt-5.6-sol"),
            call("thought_level", "off"),
        ]
    )
    mock_acp.set_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_openhands_effort_is_owned_by_its_launch_environment(tmp_path):
    """OpenHands consumes the effort before ACP starts, not via session config."""
    mock_acp = _make_mocks()
    await _connect(
        mock_acp,
        agent="openhands",
        model="gpt-5.6-sol",
        tmp_path=tmp_path,
        agent_env={"LLM_REASONING_EFFORT": "high"},
        reasoning_effort="high",
    )

    mock_acp.set_model.assert_not_awaited()
    mock_acp.set_config_option.assert_not_awaited()


@pytest.mark.asyncio
async def test_unregistered_agent_with_model_option_uses_config_option(tmp_path):
    """An agent not in the registry that advertises a 'model' option is still
    handled via the config option — discovery is from the session, not the
    registry."""
    mock_acp = _make_mocks(config_options=[{"id": "model"}])
    await _connect(
        mock_acp, agent="brand-new-acp", model="some-model", tmp_path=tmp_path
    )

    mock_acp.set_config_option.assert_awaited_once_with("model", "some-model")
    mock_acp.set_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_registry_hint_overrides_when_session_advertises_nothing(tmp_path):
    """The registry's acp_model_config_id is honored as an override even when
    session/new echoes no config options (thin transports), preserving the
    claude-agent-acp config-option path."""
    mock_acp = _make_mocks(config_options=[])
    await _connect(
        mock_acp, agent="claude-agent-acp", model="claude-opus-4-8", tmp_path=tmp_path
    )

    mock_acp.set_config_option.assert_awaited_once_with("model", "claude-opus-4-8")
    mock_acp.set_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_effort_without_advertised_option_fails_closed(tmp_path):
    """An ACP agent without an effort option must fail rather than drop it."""
    mock_acp = _make_mocks(config_options=[])
    with pytest.raises(RuntimeError, match="does not expose an effort"):
        await _connect(
            mock_acp,
            agent="test-agent",
            model=None,
            tmp_path=tmp_path,
            reasoning_effort="max",
        )

    mock_acp.close.assert_awaited()


@pytest.mark.asyncio
async def test_env_owned_model_skips_advertised_model_option(tmp_path):
    """A manifest-shaped agent (supports_acp_set_model=False + a
    BENCHFLOW_PROVIDER_MODEL env mapping) with the via-env flag set must get NO
    ACP model configuration — several registry agents (qwen-code, kilo,
    dimcode) advertise a ``model`` config option but validate values against
    their own catalog and reject the gateway alias with -32603."""
    from benchflow.agents.registry import AGENT_INSTALLERS, AGENT_LAUNCH, AGENTS
    from benchflow.agents.registry import AgentConfig as _AC

    AGENTS["env-owned-probe"] = _AC(
        name="env-owned-probe",
        install_cmd="true",
        launch_cmd="true",
        supports_acp_set_model=False,
        env_mapping={"BENCHFLOW_PROVIDER_MODEL": "OPENAI_MODEL"},
    )
    try:
        opt = MagicMock()
        opt.id = "model"
        mock_acp = _make_mocks(config_options=[opt])
        await _connect(
            mock_acp,
            agent="env-owned-probe",
            model="deepseek/deepseek-v4-flash",
            tmp_path=tmp_path,
            agent_env={
                LITELLM_MODEL_VIA_ENV: "1",
                LITELLM_MODEL_ALIAS_ENV: "benchflow-deepseek-deepseek-v4-flash",
                "OPENAI_MODEL": "benchflow-deepseek-deepseek-v4-flash",
            },
        )
        mock_acp.set_config_option.assert_not_awaited()
        mock_acp.set_model.assert_not_awaited()
    finally:
        AGENTS.pop("env-owned-probe", None)
        AGENT_INSTALLERS.pop("env-owned-probe", None)
        AGENT_LAUNCH.pop("env-owned-probe", None)
