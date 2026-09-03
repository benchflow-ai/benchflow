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

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.acp.client import ACPClient
from benchflow.acp.session import ACPSession
from benchflow.acp.types import StopReason
from benchflow.diagnostics import TransportClosedError
from benchflow.rollout import Rollout
from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    redact_trajectory_obj_with_exact_values,
)


def _stock_acp_mock() -> AsyncMock:
    """An ACPClient mock whose handshake succeeds — only model config varies."""
    mock_session = MagicMock()
    mock_session.session_id = "s1"
    mock_init = MagicMock()
    mock_init.agent_info = None

    mock_acp = AsyncMock(spec=ACPClient)
    mock_acp.connect = AsyncMock()
    mock_acp.initialize = AsyncMock(return_value=mock_init)
    mock_acp.session_new = AsyncMock(return_value=mock_session)
    mock_acp.set_config_option = AsyncMock()
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


async def test_config_option_failure_aborts_rollout(tmp_path) -> None:
    """Claude ACP config-option failures must fail closed like set_model."""
    from benchflow.acp.runtime import connect_acp

    mock_acp = _stock_acp_mock()
    mock_acp.session_new.return_value.config_options = [{"id": "model"}]
    mock_acp.set_config_option = AsyncMock(side_effect=RuntimeError("bad option"))
    mock_env = AsyncMock()

    with (
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        pytest.raises(RuntimeError, match="Failed to set ACP model config option"),
    ):
        await connect_acp(
            env=mock_env,
            agent="claude-agent-acp",
            agent_launch="claude-agent-acp",
            agent_env={},
            sandbox_user=None,
            model="claude-opus-4-8",
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )

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


async def test_initialize_timeout_is_transport_failure_not_agent_timeout(
    tmp_path,
) -> None:
    """Guards PR #921 against classifying ACP bootstrap stalls as task timeout."""
    from benchflow.acp.runtime import connect_acp

    mock_acp = _stock_acp_mock()
    mock_acp.initialize = AsyncMock(side_effect=TimeoutError())
    mock_env = AsyncMock()

    with (
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        patch("benchflow.acp.runtime.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(TransportClosedError) as exc_info,
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

    assert exc_info.value.diagnostic.transport_diagnosis == "acp_initialize_timeout"
    assert mock_acp.initialize.await_count == 4
    mock_acp.close.assert_awaited()


async def test_live_process_connection_error_retries_before_transport_exists(
    tmp_path,
) -> None:
    from benchflow.acp.runtime import connect_acp

    mock_acp = _stock_acp_mock()
    mock_env = AsyncMock()
    mock_env.live_process = AsyncMock(
        side_effect=[ConnectionError("live process unavailable"), MagicMock()]
    )

    with (
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        patch("benchflow.acp.runtime.asyncio.sleep", new_callable=AsyncMock),
    ):
        client, session, _adapter, agent_name = await connect_acp(
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

    assert mock_env.live_process.await_count == 2
    assert client is mock_acp
    assert session.session_id == "s1"
    assert agent_name == "test-agent"


async def test_no_web_firewall_runs_after_session_new_before_return(tmp_path) -> None:
    """Guards PR #921: bootstrap first, then fail-closed egress isolation."""
    from benchflow.acp.runtime import connect_acp

    events: list[str] = []
    mock_acp = _stock_acp_mock()

    async def session_new(*args, **kwargs):
        events.append("session_new")
        return MagicMock(session_id="s1")

    async def enforce(*args, **kwargs):
        events.append("firewall")

    mock_acp.session_new = AsyncMock(side_effect=session_new)
    mock_env = AsyncMock()

    with (
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        patch(
            "benchflow.acp.runtime.enforce_agent_egress_firewall",
            new_callable=AsyncMock,
            side_effect=enforce,
        ) as mock_firewall,
    ):
        await connect_acp(
            env=mock_env,
            agent="openhands",
            agent_launch="openhands acp",
            agent_env={
                "BENCHFLOW_DISALLOW_WEB_TOOLS": "1",
                "LLM_BASE_URL": "http://127.0.0.1:1234",
            },
            sandbox_user="agent",
            model=None,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )

    assert events == ["session_new", "firewall"]
    mock_firewall.assert_awaited_once_with(
        mock_env,
        "agent",
        {
            "BENCHFLOW_DISALLOW_WEB_TOOLS": "1",
            "LLM_BASE_URL": "http://127.0.0.1:1234",
        },
    )


def test_acp_observation_uses_only_live_session_identity(tmp_path) -> None:
    rollout = Rollout.__new__(Rollout)
    rollout._config = SimpleNamespace(primary_agent="configured-default")
    rollout._session = SimpleNamespace(
        agent_info=None,
        model_state={"currentModelId": "wire-model"},
        mode_state={"currentModeId": "wire-mode"},
        stop_reason=StopReason.END_TURN,
    )
    rollout._rollout_dir = tmp_path
    rollout._acp_session_observation = None

    assert rollout.acp_session_observation is None


def test_acp_observation_records_live_results_and_trajectory_digest(tmp_path) -> None:
    trajectory = tmp_path / "trajectory" / "acp_trajectory.jsonl"
    trajectory.parent.mkdir()
    payload = b'{"type":"message"}\n'
    trajectory.write_bytes(payload)
    rollout = Rollout.__new__(Rollout)
    rollout._session = SimpleNamespace(
        agent_info=SimpleNamespace(name="wire-agent"),
        model_state={"currentModelId": "wire-model"},
        mode_state={"currentModeId": "wire-mode"},
        stop_reason=StopReason.END_TURN,
    )
    rollout._rollout_dir = tmp_path
    rollout._acp_session_observation = None

    observation = rollout.acp_session_observation

    assert observation is not None
    assert observation.agent_name == "wire-agent"
    assert observation.model_id == "wire-model"
    assert observation.mode_id == "wire-mode"
    assert observation.stop_reason == "end_turn"
    assert observation.trajectory_path == str(trajectory)
    assert observation.trajectory_digest == (
        "sha256:" + hashlib.sha256(payload).hexdigest()
    )


def test_acp_observation_prefers_acknowledged_config_values(tmp_path) -> None:
    rollout = Rollout.__new__(Rollout)
    rollout._session = SimpleNamespace(
        agent_info=SimpleNamespace(name="wire-agent"),
        model_state={"currentModelId": "session-new-model"},
        mode_state={"currentModeId": "session-new-mode"},
        config_options=[
            {
                "id": "model",
                "category": "model",
                "currentValue": "acknowledged-model",
            },
            {
                "id": "mode",
                "category": "mode",
                "currentValue": "acknowledged-mode",
            },
        ],
        stop_reason=None,
    )
    rollout._rollout_dir = tmp_path
    rollout._acp_session_observation = None

    observation = rollout.acp_session_observation

    assert observation is not None
    assert observation.model_id == "acknowledged-model"
    assert observation.mode_id == "acknowledged-mode"


def test_acp_observation_exactly_redacts_protocol_metadata(tmp_path) -> None:
    credential = "opaque-observation-license-123"
    session = ACPSession(
        "session",
        redact_protocol_value=lambda value: redact_trajectory_obj_with_exact_values(
            value, [credential]
        ),
    )
    session.agent_info = SimpleNamespace(name=f"agent-{credential}")
    session.model_state = {"currentModelId": f"model-{credential}"}
    session.mode_state = {"currentModeId": f"mode-{credential}"}
    session.stop_reason = f"stop-{credential}"

    rollout = Rollout.__new__(Rollout)
    rollout._session = session
    rollout._rollout_dir = tmp_path
    rollout._acp_session_observation = None

    observation = rollout.acp_session_observation

    assert observation is not None
    persisted_values = (
        observation.agent_name,
        observation.model_id,
        observation.mode_id,
        observation.stop_reason,
    )
    assert credential not in " ".join(value for value in persisted_values if value)
    assert all("***REDACTED***" in value for value in persisted_values if value)
    assert credential in session.agent_info.name
    assert credential in session.model_state["currentModelId"]
    assert credential in session.mode_state["currentModeId"]
    assert credential in session.stop_reason


def test_reconciled_trajectory_refreshes_cached_observation_digest(tmp_path) -> None:
    trajectory = tmp_path / "trajectory" / "acp_trajectory.jsonl"
    trajectory.parent.mkdir()
    acp_event = {
        "type": "tool_call",
        "tool_call_id": "call-1",
        "title": "cat input.txt",
        "status": "completed",
        "content": [],
    }
    trajectory.write_text(json.dumps(acp_event) + "\n")
    rollout = Rollout.__new__(Rollout)
    rollout._session = SimpleNamespace(
        agent_info=SimpleNamespace(name="wire-agent"),
        model_state={"currentModelId": "wire-model"},
        mode_state={"currentModeId": "wire-mode"},
        stop_reason=StopReason.END_TURN,
    )
    rollout._rollout_dir = tmp_path
    rollout._trajectory = [acp_event]
    rollout._acp_session_observation = None
    before = rollout.acp_session_observation
    assert before is not None
    rollout._session = None

    provider_payload = json.dumps(
        {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "id": "call-1",
                                "name": "run_shell_command",
                                "args": {"command": "cat input.txt"},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "id": "call-1",
                                "name": "run_shell_command",
                                "response": {"output": "returned input"},
                            }
                        }
                    ],
                },
            ]
        }
    )
    exchange = LLMExchange(
        request=LLMRequest(
            body={"messages": [{"role": "user", "content": provider_payload}]}
        ),
        response=LLMResponse(body={}),
    )
    usage_runtime = SimpleNamespace(
        server=SimpleNamespace(
            trajectory=SimpleNamespace(exchanges=[exchange]),
        )
    )

    rollout._reconcile_acp_tool_evidence(usage_runtime)

    final_bytes = trajectory.read_bytes()
    observation = rollout.acp_session_observation
    assert observation is not None
    assert observation.agent_name == before.agent_name
    assert observation.model_id == before.model_id
    assert observation.mode_id == before.mode_id
    assert observation.stop_reason == before.stop_reason
    assert observation.trajectory_path == str(trajectory)
    assert observation.trajectory_digest == (
        "sha256:" + hashlib.sha256(final_bytes).hexdigest()
    )
    assert observation.trajectory_digest != before.trajectory_digest
