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
from unittest.mock import AsyncMock, MagicMock, patch

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
        patch(
            "benchflow.acp.runtime.DockerProcess.from_sandbox_env",
            return_value=MagicMock(),
        ),
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
async def test_effort_without_effort_config_id_fails_closed(tmp_path):
    """reasoning_effort requested for an agent that declares no effort config
    option must fail closed rather than silently drop the effort."""
    mock_acp = _make_mocks(config_options=[])
    with pytest.raises(RuntimeError, match="does not declare an ACP effort"):
        await _connect(
            mock_acp,
            agent="test-agent",
            model=None,
            tmp_path=tmp_path,
            reasoning_effort="max",
        )

    mock_acp.close.assert_awaited()
