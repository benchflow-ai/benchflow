"""Regression tests for #365: ``connect_acp`` must fail closed when
``session/set_model`` errors out.

Before the fix, a failed ``set_model`` was caught and logged as a warning;
the rollout then continued on the agent's default/previous model while
result metadata still claimed the requested model. That silently
mis-attributes the entire trajectory.

The fix raises ``RuntimeError`` (after closing the half-built client) so the
caller aborts before prompting.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.acp.client import ACPClient


def _stock_acp_mock() -> AsyncMock:
    """An ACPClient mock whose handshake succeeds — only ``set_model`` varies."""
    mock_session = MagicMock()
    mock_session.session_id = "s1"
    mock_init = MagicMock()
    mock_init.agent_info = None

    mock_acp = AsyncMock(spec=ACPClient)
    mock_acp.connect = AsyncMock()
    mock_acp.initialize = AsyncMock(return_value=mock_init)
    mock_acp.session_new = AsyncMock(return_value=mock_session)
    mock_acp.close = AsyncMock()
    return mock_acp


async def test_set_model_failure_aborts_rollout(tmp_path) -> None:
    """If ``session/set_model`` raises, ``connect_acp`` must propagate the
    failure — not log-and-continue with a corrupt session.
    """
    from benchflow.acp.runtime import connect_acp

    mock_acp = _stock_acp_mock()
    mock_acp.set_model = AsyncMock(side_effect=RuntimeError("unsupported model"))
    mock_env = AsyncMock()

    with (
        patch(
            "benchflow.acp.runtime.DockerProcess.from_sandbox_env",
            return_value=MagicMock(),
        ),
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        pytest.raises(RuntimeError, match="Failed to set model"),
    ):
        await connect_acp(
            env=mock_env,
            agent="test-agent",
            agent_launch="test-agent",
            agent_env={},
            sandbox_user=None,
            model="claude-sonnet-4-6",
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )

    # The half-built client must be closed so the agent subprocess does not
    # leak when the rollout aborts.
    mock_acp.close.assert_awaited()


async def test_set_model_timeout_aborts_rollout(tmp_path) -> None:
    """A ``set_model`` timeout (TimeoutError) must also fail closed — not
    silently leave the run on the previous model.
    """
    from benchflow.acp.runtime import connect_acp

    mock_acp = _stock_acp_mock()
    mock_acp.set_model = AsyncMock(side_effect=TimeoutError())
    mock_env = AsyncMock()

    with (
        patch(
            "benchflow.acp.runtime.DockerProcess.from_sandbox_env",
            return_value=MagicMock(),
        ),
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        pytest.raises(RuntimeError, match="Failed to set model"),
    ):
        await connect_acp(
            env=mock_env,
            agent="test-agent",
            agent_launch="test-agent",
            agent_env={},
            sandbox_user=None,
            model="claude-sonnet-4-6",
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )

    mock_acp.close.assert_awaited()


async def test_set_model_success_still_returns_session(tmp_path) -> None:
    """Happy path stays happy — only failures must abort."""
    from benchflow.acp.runtime import connect_acp

    mock_acp = _stock_acp_mock()
    mock_acp.set_model = AsyncMock()  # succeeds
    mock_env = AsyncMock()

    with (
        patch(
            "benchflow.acp.runtime.DockerProcess.from_sandbox_env",
            return_value=MagicMock(),
        ),
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
    ):
        client, session, _adapter, agent_name = await connect_acp(
            env=mock_env,
            agent="test-agent",
            agent_launch="test-agent",
            agent_env={},
            sandbox_user=None,
            model="claude-sonnet-4-6",
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )

    mock_acp.set_model.assert_awaited_once()
    mock_acp.close.assert_not_awaited()
    assert client is mock_acp
    assert session.session_id == "s1"
    assert agent_name == "test-agent"


async def test_no_model_does_not_call_set_model(tmp_path) -> None:
    """``model=None`` is a legitimate flow (model comes from agent env) — the
    fail-closed branch must not trigger when set_model is intentionally
    skipped.
    """
    from benchflow.acp.runtime import connect_acp

    mock_acp = _stock_acp_mock()
    mock_acp.set_model = AsyncMock()
    mock_env = AsyncMock()

    with (
        patch(
            "benchflow.acp.runtime.DockerProcess.from_sandbox_env",
            return_value=MagicMock(),
        ),
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
    ):
        await connect_acp(
            env=mock_env,
            agent="test-agent",
            agent_launch="test-agent",
            agent_env={},
            sandbox_user=None,
            model=None,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )

    mock_acp.set_model.assert_not_awaited()
    mock_acp.close.assert_not_awaited()
