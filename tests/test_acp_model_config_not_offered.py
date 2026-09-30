"""A Claude model the adapter refuses fails once, naming what it offers.

Guards the dx/errors fix for dx/first-run's finding: `--agent claude --model
claude-sonnet-9` made three attempts over 203.6 s, each ending
``ACP session/set_config_option failed for agent=claude-agent-acp
config=model value=claude-sonnet-9: ACP error -32603: Internal error``
(acp_error, retried), although the adapter's session/new lists its model
values. It is now the same unscored, never retried ``agent_model``
integration failure codex-acp's set_model path raises.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow._utils.scoring import AGENT_INTEGRATION, classify_error
from benchflow.acp.client import ACPClient, ACPError, acp_error
from benchflow.diagnostics import AgentModelNotOfferedError
from benchflow.evaluation import RetryConfig

MODEL_OPTION = {
    "id": "model",
    "name": "Model",
    "type": "select",
    "currentValue": "default",
    "options": [
        {"value": "default", "name": "Default (recommended)"},
        {"value": "sonnet", "name": "Sonnet"},
        {
            "group": "more",
            "name": "More",
            "options": [{"value": "opus"}, {"value": "haiku"}],
        },
    ],
}


def _acp(set_error: BaseException) -> AsyncMock:
    session = MagicMock()
    session.session_id = "s1"
    session.config_options = [MODEL_OPTION]
    session.model_state = None
    init = MagicMock()
    init.agent_info = None
    acp = AsyncMock(spec=ACPClient)
    acp.initialize = AsyncMock(return_value=init)
    acp.session_new = AsyncMock(return_value=session)
    acp.set_config_option = AsyncMock(side_effect=set_error)
    return acp


async def _connect(tmp_path, acp, model: str) -> None:
    from benchflow.acp.runtime import connect_acp

    with (
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=acp),
    ):
        await connect_acp(
            env=AsyncMock(),
            agent="claude-agent-acp",
            agent_launch="claude-agent-acp",
            agent_env={},
            sandbox_user=None,
            model=model,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )


# claude-agent-acp throws `Error("Invalid value for config option model: <v>")`;
# the ACP TypeScript SDK (1.5.0, jsonrpc.ts errorToResult) answers it as
# -32603 "Internal error" with data {"details": <the message>}.
REFUSED = {"details": "Invalid value for config option model: claude-sonnet-9"}


async def test_a_refused_model_names_the_offered_ones_and_is_not_retried(tmp_path):
    acp = _acp(acp_error(-32603, "Internal error", REFUSED))
    with pytest.raises(AgentModelNotOfferedError) as caught:
        await _connect(tmp_path, acp, "claude-sonnet-9")
    message = str(caught.value)
    assert message == (
        "agent integration failure [agent_model]: claude-agent-acp does not offer "
        "model 'claude-sonnet-9' for this login; it offers: default, sonnet, opus, haiku"
    )
    assert classify_error(message) == AGENT_INTEGRATION
    assert RetryConfig().should_retry(message) is False


async def test_a_timeout_is_not_called_a_refusal(tmp_path):
    acp = _acp(TimeoutError())
    with pytest.raises(RuntimeError) as caught:
        await _connect(tmp_path, acp, "claude-sonnet-9")
    assert not isinstance(caught.value, AgentModelNotOfferedError)
    assert "Failed to set ACP model config option" in str(caught.value)


@pytest.mark.parametrize(
    ("error", "model"),
    [
        # An opaque -32603 with nothing saying the value was refused.
        (ACPError(-32603, "Internal error"), "claude-sonnet-9"),
        # A listed value that fails is not a value the agent does not offer.
        (ACPError(-32603, "Internal error", REFUSED), "sonnet"),
        # Second review finding: "not available" read as a refusal, so a
        # provider that is briefly down made the model permanently wrong.
        (
            ACPError(-32603, "Internal error", {"details": "model not available"}),
            "claude-sonnet-9",
        ),
        # A refusal that names a different value is not this value's refusal.
        (
            ACPError(
                -32603,
                "Internal error",
                {"details": "Invalid value for config option model: other-model"},
            ),
            "claude-sonnet-9",
        ),
    ],
)
async def test_other_config_errors_stay_retryable(tmp_path, error, model):
    """Review finding: every -32603 on the model option became a permanent,
    never retried agent_model failure that also trips the circuit breaker."""
    acp = _acp(error)
    with pytest.raises(RuntimeError) as caught:
        await _connect(tmp_path, acp, model)
    assert not isinstance(caught.value, AgentModelNotOfferedError)
    assert RetryConfig().should_retry(str(caught.value)) is True
