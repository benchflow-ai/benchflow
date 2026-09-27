"""A model the agent does not offer fails fast, names the offered models, and
is not retried.

When codex-acp was asked for a model its session did not advertise,
BenchFlow sent the bare id to ``session/set_model``, the agent answered
``ACP error -32603: Internal error``, and a batch retried that deterministic
failure with new sandboxes for every task, with no hint of which model ids
the agent would accept.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow._utils.scoring import AGENT_INTEGRATION, classify_error
from benchflow.acp.client import ACPClient
from benchflow.evaluation import RetryConfig

ADVERTISED = {
    "currentModelId": "gpt-5.6-sol[medium]",
    "availableModels": [
        {"modelId": "gpt-5.6-sol[medium]", "name": "gpt-5.6-sol (medium)"},
        {"modelId": "gpt-5.6-sol[high]", "name": "gpt-5.6-sol (high)"},
        {"modelId": "gpt-5.4-mini[medium]", "name": "gpt-5.4-mini (medium)"},
    ],
}


def _acp_mock() -> AsyncMock:
    session = MagicMock()
    session.session_id = "s1"
    session.config_options = [{"id": "model"}]
    session.model_state = ADVERTISED
    init = MagicMock()
    init.agent_info = None
    acp = AsyncMock(spec=ACPClient)
    acp.connect = AsyncMock()
    acp.initialize = AsyncMock(return_value=init)
    acp.session_new = AsyncMock(return_value=session)
    acp.set_model = AsyncMock(
        side_effect=RuntimeError("ACP error -32603: Internal error")
    )
    acp.set_config_option = AsyncMock()
    acp.close = AsyncMock()
    return acp


async def _connect(tmp_path, acp, model: str) -> None:
    from benchflow.acp.runtime import connect_acp

    with (
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=acp),
    ):
        await connect_acp(
            env=AsyncMock(),
            agent="codex-acp",
            agent_launch="codex-acp",
            agent_env={},
            sandbox_user=None,
            model=model,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )


async def test_unoffered_codex_model_fails_before_set_model(tmp_path) -> None:
    acp = _acp_mock()

    with pytest.raises(RuntimeError) as excinfo:
        await _connect(tmp_path, acp, "gpt-9-unknown")

    message = str(excinfo.value)
    assert message.startswith("agent integration failure [agent_model]")
    assert "gpt-9-unknown" in message
    assert "gpt-5.6-sol" in message and "gpt-5.4-mini" in message
    acp.set_model.assert_not_awaited()
    acp.close.assert_awaited()
    category = classify_error(message)
    assert category == AGENT_INTEGRATION
    assert not RetryConfig().should_retry(message, category=category)


async def test_offered_codex_model_is_still_selected(tmp_path) -> None:
    acp = _acp_mock()
    acp.set_model = AsyncMock()

    await _connect(tmp_path, acp, "gpt-5.6-sol")

    acp.set_model.assert_awaited_once_with("gpt-5.6-sol[medium]")


def test_rollout_records_the_integration_error_text_verbatim() -> None:
    """The rollout keeps the message as the trial error (no exception-class
    prefix), so the category and the batch circuit breaker both see it."""
    from benchflow.diagnostics import AgentModelNotOfferedError
    from benchflow.evaluation import ApiErrorCircuitBreaker
    from benchflow.integration_health import PERMANENT_CAUSES

    assert "agent_model" in PERMANENT_CAUSES
    err = AgentModelNotOfferedError("codex-acp", "gpt-9-unknown", ["gpt-5.6-sol"])
    result = MagicMock(error=str(err), error_category=None)
    assert ApiErrorCircuitBreaker._fingerprint_of(result) == "integration:agent_model"


async def test_rollout_marks_the_trial_unscored_with_the_agent_model_cause(
    tmp_path,
) -> None:
    import json
    from datetime import datetime

    from benchflow.diagnostics import AgentModelNotOfferedError
    from benchflow.rollout import Rollout, RolloutConfig

    task = tmp_path / "t"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n[environment]\n')
    (task / "instruction.md").write_text("Write hello.txt.\n")
    rollout = Rollout(
        RolloutConfig(task_path=task, agent="codex-acp", model="gpt-9-unknown")
    )
    trial = tmp_path / "trial"
    trial.mkdir()
    rollout._rollout_dir = trial
    rollout._rollout_name = "t__codex"
    rollout._started_at = datetime.now()
    rollout.setup = AsyncMock()
    rollout.start = AsyncMock()
    rollout.install_agent = AsyncMock(
        side_effect=AgentModelNotOfferedError(
            "codex-acp", "gpt-9-unknown", ["gpt-5.6-sol"]
        )
    )
    rollout._env = AsyncMock()
    rollout._agent_cwd = "/app"

    async def cleanup():
        rollout._phase = "cleaned"

    rollout.cleanup = AsyncMock(side_effect=cleanup)

    await rollout.run()

    saved = json.loads((trial / "result.json").read_text())
    assert saved["error_category"] == AGENT_INTEGRATION
    assert saved["error"].startswith("agent integration failure [agent_model]")
    assert saved["integration_failure_info"]["cause"] == "agent_model"
    assert saved["rewards"] is None
