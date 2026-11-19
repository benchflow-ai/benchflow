"""A requested reasoning effort the agent does not offer fails the trial; the
effort that ran is recorded beside the one asked for.

Guards the silent fallback in ``_codex_session_model_id``: when the codex-acp
session offered no ``model[effort]`` variant for the requested effort,
BenchFlow selected the model's ``medium`` variant, and the trial recorded
only the effort it had asked for (a published run's xhigh rows).
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow._utils.scoring import AGENT_INTEGRATION, classify_error
from benchflow.acp.client import ACPClient
from benchflow.acp.session import ACPSession
from benchflow.diagnostics import AgentEffortNotOfferedError
from benchflow.evaluation import ApiErrorCircuitBreaker, RetryConfig
from benchflow.rollout._results import _build_rollout_result, _record_effort_in_config
from benchflow.trajectories.model_calls import (
    effort_outcome,
    effort_warning,
    write_model_calls,
)

ASTRA = {
    "currentModelId": "gpt-6-astra[low]",
    "availableModels": [
        {"modelId": "gpt-6-astra[low]"},
        {"modelId": "gpt-6-astra[medium]"},
        {"modelId": "gpt-6-astra[high]"},
    ],
}


def _acp(model_state=None, config_options=None):
    session = ACPSession("s1")
    # A copy: set_model makes the session's current model the one it took.
    session.model_state = copy.deepcopy(model_state)
    session.config_options = list(config_options or [])
    init = MagicMock()
    init.agent_info = None
    acp = AsyncMock(spec=ACPClient)
    acp.initialize = AsyncMock(return_value=init)
    acp.session_new = AsyncMock(return_value=session)
    acp.set_model = AsyncMock()
    acp.set_config_option = AsyncMock()
    return acp, session


async def _connect(tmp_path, acp, *, agent, model, effort):
    from benchflow.acp.runtime import connect_acp

    with (
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=acp),
    ):
        return await connect_acp(
            env=AsyncMock(),
            agent=agent,
            agent_launch=agent,
            agent_env={},
            sandbox_user=None,
            model=model,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
            reasoning_effort=effort,
        )


async def test_an_effort_codex_does_not_offer_fails_before_the_model_is_set(
    tmp_path,
):
    acp, _ = _acp(ASTRA)

    with pytest.raises(AgentEffortNotOfferedError) as caught:
        await _connect(
            tmp_path, acp, agent="codex-acp", model="gpt-6-astra", effort="xhigh"
        )

    message = str(caught.value)
    assert message == (
        "agent integration failure [agent_effort]: codex-acp does not offer "
        "reasoning effort 'xhigh' for model 'gpt-6-astra'; it offers: low, "
        "medium, high"
    )
    acp.set_model.assert_not_awaited()
    acp.close.assert_awaited()
    # Not retried, and a batch stops after five in a row, as for a model the
    # agent does not offer.
    assert classify_error(message) == AGENT_INTEGRATION
    assert not RetryConfig().should_retry(message)
    result = MagicMock(error=message, error_category=None)
    assert ApiErrorCircuitBreaker._fingerprint_of(result) == "integration:agent_effort"


@pytest.mark.parametrize(
    "effort,model_id,applied",
    [("high", "gpt-6-astra[high]", "high"), (None, "gpt-6-astra[low]", "low")],
)
async def test_the_session_records_the_effort_its_model_id_carries(
    tmp_path, effort, model_id, applied
):
    """With no effort asked for, the recorded one is the session's own: the
    variant it was already on (this model's default is low)."""
    acp, session = _acp(ASTRA)

    await _connect(tmp_path, acp, agent="codex-acp", model="gpt-6-astra", effort=effort)

    acp.set_model.assert_awaited_once_with(model_id)
    assert session.effort_applied == applied


async def test_an_adapter_that_maps_an_effort_records_its_answer(tmp_path):
    """The Antigravity shim takes max and answers that its thinking option is
    high (agy has three levels): the session records high, not max."""
    thinking = {
        "id": "thinking",
        "currentValue": "high",
        "options": [{"value": v} for v in ("low", "medium", "high")],
    }
    acp, session = _acp(config_options=[{**thinking, "currentValue": "medium"}])

    async def answer(config_id, value):
        session.config_options = [thinking]
        return {"configOptions": [thinking]}

    acp.set_config_option = AsyncMock(side_effect=answer)
    await _connect(
        tmp_path, acp, agent="antigravity", model="gemini-3.1-pro", effort="max"
    )

    acp.set_config_option.assert_any_await("thinking", "max")
    assert session.effort_applied == "high"


def _summary(source="proxy", agent_effort=None, calls=3):
    return {
        "calls": calls,
        "source": source,
        "agent_effort": agent_effort or {},
    }


def test_the_calls_say_which_effort_was_sent():
    effort = effort_outcome(
        _summary(agent_effort={"high": 884}), requested="max", applied="max"
    )
    assert effort == {
        "requested": "max",
        "requested_from": None,
        "applied": "max",
        "route": None,
        "sent": {"high": 884},
        "sent_source": "proxy",
        "calls_with_effort": 884,
        "calls_with_budget": 0,
        "calls_without_effort": None,
        "matches": False,
        "reason": "mismatch",
    }
    assert effort_warning(effort) == (
        "reasoning effort max was asked for, but the agent's model calls "
        "carried high (884 calls)"
    )


def test_the_proxy_seeing_no_effort_is_evidence_a_session_log_is_not():
    # The proxy saw calls and none carried an effort: unmeasured, warned, not
    # a mismatch (ruling: a route BenchFlow sets is tested apart).
    dropped = effort_outcome(_summary(), requested="xhigh", applied="xhigh")
    assert (dropped["sent"], dropped["matches"]) == ({}, None)
    assert "none of the agent's model calls carried an effort" in (
        effort_warning(dropped) or ""
    )
    # Claude Code's session log records no request settings: with no calls
    # to read, the session's word alone confirms nothing.
    unknown = effort_outcome(
        _summary(source="claude-code-session-log"), requested="max", applied="max"
    )
    assert (unknown["sent"], unknown["matches"]) == (None, None)
    assert effort_warning(unknown) is None
    assert (
        effort_outcome(_summary(), requested=None, applied="medium")["matches"] is None
    )


def test_a_route_or_an_adapter_that_lowers_the_effort_is_named():
    effort = effort_outcome(
        _summary(agent_effort={"high": 4}),
        requested="max",
        applied="high",
        route="high",
    )
    assert effort["matches"] is False
    assert effort_warning(effort) == (
        "reasoning effort max was asked for, but the agent applied high; the "
        "agent's model calls carried high (4 calls)"
    )


@pytest.mark.parametrize("label", ["default", "auto"])
def test_a_session_label_that_is_no_level_is_nothing_applied(label):
    """R383 M1: with no -e, claude-agent-acp answers its effort option with
    ``default``; its calls carried Claude Code's own level, which differs from
    nothing that was asked or applied. The label is kept as said."""
    effort = effort_outcome(
        _summary(agent_effort={"medium": 2}), requested=None, applied=label
    )
    assert effort["applied"] == label
    assert (effort["matches"], effort["reason"]) == (None, "nothing_to_compare")
    assert effort_warning(effort) is None
    asked = effort_outcome(
        _summary(agent_effort={"max": 2}), requested="max", applied=label
    )
    assert (asked["matches"], asked["reason"]) == (True, None)


def test_a_routes_pinned_effort_decides_nothing():
    """R383 L1: the calls say what was sent; a route's pin is kept beside."""
    effort = effort_outcome(
        _summary(agent_effort={"low": 3}), requested="low", applied=None, route="high"
    )
    assert (effort["matches"], effort["route"]) == (True, "high")
    assert effort_warning(effort) is None


def test_a_messages_call_records_its_own_effort_not_the_routes_pin(tmp_path):
    """R383 L1: a Bedrock Claude route pins reasoning_effort, which the
    proxy's log copies into each call, but an Anthropic Messages call goes
    upstream with its own output_config.effort."""
    trial = tmp_path / "task__one"
    (trial / "trajectory").mkdir(parents=True)
    call = _call("high")
    call["request"]["body"]["output_config"] = {"effort": "medium"}
    (trial / "trajectory/llm_trajectory.jsonl").write_text(json.dumps(call) + "\n")
    summary = write_model_calls(trial)
    assert summary is not None
    assert summary["agent_effort"] == {"medium": 1}


def _call(effort, purpose="agent"):
    return {
        "request": {"body": {"model": "m", "reasoning_effort": effort}},
        "response": {
            "status_code": 200,
            "body": {"choices": [{"finish_reason": "stop"}]},
        },
        "metadata": {"call_purpose": purpose},
    }


def test_result_and_config_record_the_effort_that_ran(tmp_path):
    """A side call's effort (a judge's here) is not the run's."""
    trial = tmp_path / "task__one"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "trajectory/llm_trajectory.jsonl").write_text(
        "".join(
            json.dumps(c) + "\n"
            for c in (_call("high"), _call("high"), _call("low", purpose="judge"))
        )
    )
    (trial / "config.json").write_text(json.dumps({"reasoning_effort": "max"}))
    _record_effort_in_config(trial, "max", None)
    write_model_calls(trial)
    _build_rollout_result(
        trial,
        task_name="task",
        rollout_name=trial.name,
        agent="test",
        agent_name="test",
        model="m",
        n_tool_calls=1,
        prompts=["task"],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        reasoning_effort="max",
        effort_applied="max",
        rewards={"reward": 0.0},
        started_at=datetime.now(),
        timing={},
    )

    config = json.loads((trial / "config.json").read_text())
    assert (config["reasoning_effort"], config["reasoning_effort_applied"]) == (
        "max",
        "max",
    )
    assert "reasoning_effort_route" not in config
    result = json.loads((trial / "result.json").read_text())
    assert result["effort"]["sent"] == {"high": 2}
    assert result["effort"]["matches"] is False
    assert result["warnings"] == [
        "reasoning effort max was asked for, but the agent's model calls "
        "carried high (2 calls)"
    ]


async def test_the_rollout_marks_the_trial_unscored_with_the_agent_effort_cause(
    tmp_path,
):
    from benchflow.rollout import Rollout, RolloutConfig

    task = tmp_path / "t"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n[environment]\n')
    (task / "instruction.md").write_text("Write hello.txt.\n")
    rollout = Rollout(
        RolloutConfig(task_path=task, agent="codex-acp", model="gpt-6-astra")
    )
    trial = tmp_path / "trial"
    trial.mkdir()
    rollout._rollout_dir = trial
    rollout._rollout_name = "t__codex"
    rollout._started_at = datetime.now()
    rollout.setup = AsyncMock()
    rollout.start = AsyncMock()
    rollout.install_agent = AsyncMock(
        side_effect=AgentEffortNotOfferedError(
            "codex-acp", "gpt-6-astra", "xhigh", ["low", "medium", "high"]
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
    assert saved["error"].startswith("agent integration failure [agent_effort]")
    assert saved["integration_failure_info"]["cause"] == "agent_effort"
    assert saved["rewards"] is None


def test_a_connection_keeps_the_sessions_effort_and_the_routes(tmp_path):
    """connect() and connect_as() for the run's own agent record what the
    session applied and what the proxy's route pins (Bedrock's ceiling)."""
    from types import SimpleNamespace

    from benchflow.rollout import Rollout

    (tmp_path / "config.json").write_text(json.dumps({"reasoning_effort": "max"}))
    rollout = Rollout.__new__(Rollout)
    rollout._config = SimpleNamespace(primary_agent="claude-agent-acp")
    rollout._agent_env = {}
    rollout._session = SimpleNamespace(effort_applied="max")
    rollout._usage_runtime = SimpleNamespace(
        server=SimpleNamespace(
            route=SimpleNamespace(
                litellm_params={"model": "bedrock/x", "reasoning_effort": "high"}
            )
        )
    )

    rollout._record_effort_applied(tmp_path)

    assert (rollout._effort_applied, rollout._effort_route) == ("max", "high")
    config = json.loads((tmp_path / "config.json").read_text())
    assert (config["reasoning_effort_applied"], config["reasoning_effort_route"]) == (
        "max",
        "high",
    )


@pytest.mark.parametrize(("env", "applied"), [("high", None), ("auto", "max")])
def test_claude_codes_effort_variable_is_the_effort_it_applies(tmp_path, env, applied):
    """With --effort max and CLAUDE_CODE_EFFORT_LEVEL=high in the agent's env,
    Claude Code sends high: the record does not say "applied max", nor echo
    the variable back as applied; the variable is the request, and
    the calls say what ran."""
    from types import SimpleNamespace

    from benchflow.rollout import Rollout

    (tmp_path / "config.json").write_text(json.dumps({"reasoning_effort": "max"}))
    rollout = Rollout.__new__(Rollout)
    rollout._config = SimpleNamespace(primary_agent="claude-agent-acp")
    rollout._agent_env = {"CLAUDE_CODE_EFFORT_LEVEL": env}
    rollout._session = SimpleNamespace(effort_applied="max")
    rollout._usage_runtime = SimpleNamespace(server=None)
    rollout._record_effort_applied(tmp_path)
    assert rollout._effort_applied == applied
