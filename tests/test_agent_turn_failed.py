"""A session-factory agent's failed turn is an agent error, not a timeout.

Guards the audit fix on branch ``audit-agent-turn-failed``
(finding LIFE-2, the lead's ruling of 2026-11-13). Any failure of
``session.prompt()`` that was not a timeout (a provider 5xx, a one-shot CLI's
non-zero exit, an exception) was wrapped in ``AgentPromptTimeoutError`` to
keep the partial trajectory. The trial was then recorded as the agent running
out of time: verified, given reward 0.0 when the verifier gave none, category
``timeout``, never retried.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.scoring import (
    ACP_ERROR,
    INFRA_ERROR,
    PROVIDER_AUTH,
    TIMED_OUT,
    classify_error,
)
from benchflow.acp.types import StopReason
from benchflow.agents.errors import (
    SESSION_FACTORY_ERROR_HEADER,
    AgentProtocolError,
    AgentTurnFailedError,
)
from benchflow.evaluation import RetryConfig
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.rollout.session_factory_runtime import execute_prompts_session_factory


class _Session:
    """A session-factory session whose prompt appends a step, then fails."""

    def __init__(self, failure: BaseException) -> None:
        self.steps: list[dict] = []
        self._failure = failure

    async def prompt(self, text: str) -> StopReason:
        self.steps.append({"type": "user_message", "text": text})
        raise self._failure


def _failed_turn(failure: BaseException) -> AgentTurnFailedError:
    with pytest.raises(AgentTurnFailedError) as caught:
        asyncio.run(
            execute_prompts_session_factory(_Session(failure), ["go"], timeout=30)
        )
    return caught.value


def test_a_provider_503_is_the_providers_error_and_is_retried():
    error = _failed_turn(RuntimeError("provider returned HTTP 503 Service Unavailable"))

    assert isinstance(error, AgentProtocolError)
    assert not isinstance(error, TimeoutError)
    text = str(error)
    assert text.startswith(f"{SESSION_FACTORY_ERROR_HEADER}: ")
    assert classify_error(text) == INFRA_ERROR
    assert RetryConfig().should_retry(text)
    assert error.trajectory == [{"type": "user_message", "text": "go"}]
    assert error.executed_prompts == ["go"]


def test_a_crash_is_an_agent_error_and_is_retried_like_one():
    text = str(_failed_turn(RuntimeError("omnigent exited with status 2")))

    assert classify_error(text) == ACP_ERROR
    assert classify_error(text) != TIMED_OUT
    assert RetryConfig().should_retry(text)


def test_the_status_the_model_proxy_saw_decides_the_category():
    """As for an ACP error: the lifecycle appends the provider's status."""
    text = str(_failed_turn(RuntimeError("request failed")))

    refused = f"{text} | provider auth failed (HTTP 401)"
    assert classify_error(refused) == PROVIDER_AUTH
    assert not RetryConfig().should_retry(refused)


def test_a_real_timeout_is_still_the_agents_timeout():
    from benchflow.acp.runtime import AgentPromptTimeoutError

    class _Slow:
        def __init__(self) -> None:
            self.steps: list[dict] = []

        async def prompt(self, text: str) -> StopReason:
            await asyncio.sleep(30)
            return StopReason.END_TURN

    with pytest.raises(AgentPromptTimeoutError):
        asyncio.run(execute_prompts_session_factory(_Slow(), ["go"], timeout=0.05))


def test_the_trial_is_not_graded_as_a_timeout(tmp_path: Path):
    """Through the lifecycle: no timeout record, no fallback reward of 0, no
    verifier run on a workspace the agent never finished."""
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n[agent]\ntimeout_sec = 900\n')
    (task / "instruction.md").write_text("Do the thing.\n")
    rollout = Rollout(RolloutConfig(task_path=task, agent="claude-agent-acp"))
    trial = tmp_path / "trial"
    trial.mkdir()
    rollout._rollout_dir = trial
    rollout._rollout_name = "task__1"
    rollout._started_at = datetime.now()
    for name in ("setup", "start", "install_agent", "cleanup", "verify"):
        setattr(rollout, name, AsyncMock())

    async def agent(_steps):
        raise AgentTurnFailedError(
            f"{SESSION_FACTORY_ERROR_HEADER}: prompt failed before completion: "
            "provider returned HTTP 503 Service Unavailable",
            trajectory=[{"type": "user_message", "text": "go"}],
            executed_prompts=["go"],
        )

    rollout._run_steps = agent

    result = asyncio.run(rollout.run())

    saved = json.loads((trial / "result.json").read_text())
    assert result.error.startswith(f"{SESSION_FACTORY_ERROR_HEADER}: ")
    assert result.error_category == saved["error_category"] == INFRA_ERROR
    assert saved["agent_timeout_info"] is None
    assert result.rewards is None
    rollout.verify.assert_not_awaited()
    assert RetryConfig().should_retry(result.error, category=result.error_category)
