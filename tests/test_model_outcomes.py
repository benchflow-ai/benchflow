"""agent_outcome: how the agent's model ended, beside the verifier's score."""

import json
import threading
from datetime import datetime
from pathlib import Path

import pytest

from benchflow.acp.runtime import _agent_prompt_timeout_error
from benchflow.acp.session import ACPSession
from benchflow.acp.types import StopReason
from benchflow.diagnostics import AgentPromptTimeoutError
from benchflow.report import build_report
from benchflow.rollout._results import _build_rollout_result
from benchflow.trajectories._capture import _capture_session_trajectory
from benchflow.trajectories.model_calls import write_model_calls


def _call(status=200, reason="stop", purpose="agent"):
    return {
        "request": {
            "body": {"model": "test", "messages": [{"role": "user", "content": "task"}]}
        },
        "response": {
            "status_code": status,
            "body": {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": reason,
                    }
                ]
            },
        },
        "metadata": {"call_purpose": purpose},
    }


def _responses_tool_call():
    """A Responses API call that asked for a tool (a Codex tool turn): it
    ends ``completed`` all the same."""
    return {
        "request": {"body": {"model": "gpt-5.5", "input": "task"}},
        "response": {
            "status_code": 200,
            "body": {
                "object": "response",
                "status": "completed",
                "output": [
                    {"type": "function_call", "name": "shell", "arguments": "{}"}
                ],
            },
        },
        "metadata": {"call_purpose": "agent"},
    }


def _session(text="answer", timed_out=False):
    session = ACPSession("trial")
    session.record_user_prompt("task")
    session.handle_update(
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": text},
        }
    )
    session.mark_prompt_end()
    if timed_out:
        session.record_agent_timeout(
            timeout_sec=600, pending_tool_call_ids=[], terminal_trajectory_complete=True
        )
    return session


def _result(path, calls=(), stop=None, text="answer", error=None, timed_out=False):
    """result.json of a trial whose proxy recorded *calls* and whose last
    prompt ended with *stop*: teardown writes the model calls, then the
    result is built, as a rollout does."""
    path.mkdir(parents=True, exist_ok=True)
    if calls:
        (path / "trajectory").mkdir(parents=True)
        (path / "trajectory/llm_trajectory.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in calls)
        )
        write_model_calls(path)
    _build_rollout_result(
        path,
        task_name="task",
        rollout_name=path.name,
        agent="test",
        agent_name="test",
        model="test",
        n_tool_calls=3,
        prompts=["task"],
        error=error,
        verifier_error=None,
        trajectory=_capture_session_trajectory(_session(text, timed_out)),
        partial_trajectory=False,
        # Passed only when the prompt ended, so the tests without one also
        # run against a build that has no stop_reason argument.
        **({} if stop is None else {"stop_reason": stop.value}),
        rewards={"reward": 0.0},
        started_at=datetime.now(),
        timing={},
        total_tokens=5000,
    )
    return json.loads((path / "result.json").read_text())


@pytest.mark.parametrize(
    "statuses,terminal",
    [([200, 503], True), ([503, 200], False), ([200, 503, 200], False)],
)
def test_http_errors_are_kept_and_terminal_failure_is_distinct(
    tmp_path, statuses, terminal
):
    result = _result(tmp_path / "task__one", [_call(code) for code in statuses])
    outcome = result["agent_outcome"]
    assert outcome["http_errors"] == 1
    assert outcome["terminal_provider_error"] is terminal
    assert outcome["status"] == ("provider_error" if terminal else "completed")
    assert result["rewards"] == {"reward": 0.0}
    assert (
        json.loads((tmp_path / "task__one/model-calls.json").read_text())["http_errors"]
        == 1
    )


@pytest.mark.parametrize("statuses,terminal", [([200, 429], True), ([429, 200], False)])
def test_a_rate_limit_on_the_last_call_is_a_provider_error(
    tmp_path, statuses, terminal
):
    outcome = _result(tmp_path, [_call(code) for code in statuses])["agent_outcome"]
    assert outcome["terminal_provider_error"] is terminal
    assert outcome["status"] == ("provider_error" if terminal else "completed")
    assert outcome["last_http_status"] == statuses[-1]


def test_a_timed_out_trial_is_never_completed(tmp_path):
    """A Codex trial through the proxy that ran out of time: its last call
    (a tool request) ended ``completed`` with HTTP 200."""
    result = _result(
        tmp_path,
        [_responses_tool_call()],
        error="Agent timed out after 600s",
        timed_out=True,
    )
    assert result["agent_outcome"]["status"] == "unknown"
    assert result["agent_outcome"]["stop_reason"] is None
    # A timeout recorded only as the trajectory's last event counts too.
    marker_only = _result(tmp_path / "marker", [_call()], timed_out=True)
    assert marker_only["agent_outcome"]["status"] == "unknown"


def test_an_agent_that_fails_after_a_successful_call_is_not_completed(tmp_path):
    result = _result(tmp_path, [_call()], error="ACP connection lost")
    assert result["agent_outcome"]["status"] == "unknown"
    assert result["agent_outcome"]["last_http_status"] == 200


def test_a_responses_call_that_completed_is_not_a_finished_turn(tmp_path):
    asked = _result(tmp_path / "asked", [_responses_tool_call()])
    assert asked["agent_outcome"]["status"] == "unknown"
    ended = _result(
        tmp_path / "ended", [_responses_tool_call()], stop=StopReason.END_TURN
    )
    assert ended["agent_outcome"]["status"] == "completed"


def test_a_reviewer_error_does_not_end_the_agent_turn(tmp_path):
    result = _result(tmp_path, [_call(), _call(503, purpose="reviewer")])
    assert result["agent_outcome"]["http_errors"] == 1
    assert result["agent_outcome"]["terminal_provider_error"] is False


def test_an_acp_refusal_reaches_the_result(tmp_path):
    result = _result(tmp_path, stop=StopReason.REFUSAL)
    assert result["agent_outcome"]["status"] == "refusal"
    assert result["agent_outcome"]["stop_reason"] == "refusal"
    assert result["rewards"] == {"reward": 0.0}
    assert result["error"] is None


@pytest.mark.parametrize("reason", ["refusal", "content_filter"])
def test_provider_refusals_are_structured_outcomes(tmp_path, reason):
    result = _result(tmp_path, [_call(reason=reason)])
    assert result["agent_outcome"]["status"] == "refusal"
    assert result["agent_outcome"]["refused_calls"] == 1


def test_a_recovered_refusal_is_counted_but_not_terminal(tmp_path):
    result = _result(tmp_path, [_call(reason="content_filter"), _call()])
    assert result["agent_outcome"]["status"] == "completed"
    assert result["agent_outcome"]["refused_calls"] == 1


def test_refusal_is_not_inferred_from_prose(tmp_path):
    result = _result(
        tmp_path, stop=StopReason.END_TURN, text="I cannot assist with this request."
    )
    assert result["agent_outcome"]["status"] == "completed"
    assert result["agent_outcome"]["refused_calls"] is None


def test_report_shows_outcomes_and_counts_beside_scores(tmp_path):
    _result(tmp_path / "task__one", [_call(), _call(503)])
    _result(tmp_path / "other__two", stop=StopReason.REFUSAL)
    report = build_report([tmp_path], "test")
    assert "HTTP errors" in report and "Refused calls" in report
    assert "provider_error" in report and "refusal" in report
    assert "Verifier reward" in report


@pytest.mark.parametrize(
    "body",
    [
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "refusal": "declined"},
                }
            ]
        },
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "refusal", "refusal": "declined"}],
                }
            ],
        },
    ],
)
def test_explicit_provider_refusal_is_not_hidden_by_normal_finish(tmp_path, body):
    call = _call()
    call["response"]["body"] = body
    result = _result(tmp_path, [call])
    assert result["agent_outcome"]["status"] == "refusal"
    assert result["agent_outcome"]["refused_calls"] == 1


def test_a_new_prompt_clears_the_previous_stop_reason():
    session = ACPSession("trial")
    session.stop_reason = StopReason.REFUSAL
    session.record_user_prompt("try a different task")
    assert session.stop_reason is None


def test_a_claude_code_subagent_refusal_is_counted_not_the_outcome(tmp_path):
    """Claude Code's session log marks a subagent's calls (isSidechain): they
    count in the totals but are never the agent's last call."""
    logs = tmp_path / "agent" / "claude-sessions"
    logs.mkdir(parents=True)
    rows = [
        {"type": "user", "timestamp": "2026-11-05T00:00:00Z"},
        {
            "type": "assistant",
            "timestamp": "2026-11-05T00:00:01Z",
            "message": {"id": "parent", "stop_reason": "end_turn", "usage": {}},
        },
        {
            "type": "assistant",
            "isSidechain": True,
            "timestamp": "2026-11-05T00:00:02Z",
            "message": {"id": "child", "stop_reason": "refusal", "usage": {}},
        },
    ]
    (logs / "s.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = write_model_calls(tmp_path)
    assert summary is not None
    assert summary["last_agent_call"]["finish_reason"] == "end_turn"
    result = _result(tmp_path, stop=StopReason.END_TURN)
    assert result["agent_outcome"]["refused_calls"] == 1
    assert result["agent_outcome"]["status"] == "completed"


def test_an_unfinished_last_call_is_not_a_known_terminal_5xx(tmp_path):
    unfinished = _call()
    unfinished["response"] = None
    result = _result(tmp_path, [_call(503), unfinished])
    assert result["agent_outcome"]["http_errors"] == 1
    assert result["agent_outcome"]["status"] == "unknown"
    assert result["agent_outcome"]["terminal_provider_error"] is False


def test_building_a_result_does_not_read_the_proxy_log_again(tmp_path, monkeypatch):
    """Teardown writes the model calls once; a result build reads the totals
    it wrote and never parses the proxy's log (tens of MB in a long run)."""
    from benchflow.trajectories import model_calls as mc

    def no_reparse(*_args, **_kwargs):
        raise AssertionError("the proxy log was read again")

    (tmp_path / "trajectory").mkdir()
    (tmp_path / "trajectory/llm_trajectory.jsonl").write_text(
        json.dumps(_call(503)) + "\n"
    )
    write_model_calls(tmp_path)
    written = (tmp_path / "model-calls.json").read_text()
    monkeypatch.setattr(mc, "model_calls", no_reparse)
    monkeypatch.setattr(mc, "write_model_calls", no_reparse)
    result = _result(tmp_path, stop=StopReason.END_TURN)
    assert result["agent_outcome"]["status"] == "provider_error"
    assert (tmp_path / "model-calls.json").read_text() == written
    # Before teardown wrote them, a result has no counts, and writes none.
    early = _result(tmp_path / "early", stop=StopReason.END_TURN)
    assert early["agent_outcome"]["http_errors"] is None
    assert not (tmp_path / "early" / "model-calls.json").exists()


async def test_teardown_writes_the_model_calls_off_the_event_loop(
    tmp_path, monkeypatch
):
    import benchflow.rollout as rollout_module
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.task.paths import RolloutPaths

    async def nothing(_rollout):
        return None

    seen = []
    monkeypatch.setattr(rollout_module, "price_from_session_log", nothing)
    monkeypatch.setattr(rollout_module, "keep_session_logs", nothing)
    monkeypatch.setattr(
        rollout_module,
        "write_model_calls",
        lambda trial: seen.append((trial, threading.current_thread())),
    )
    (tmp_path / "task").mkdir()
    rollout = Rollout(RolloutConfig(task_path=tmp_path / "task"))
    rollout._rollout_paths = RolloutPaths(tmp_path / "trial")
    await rollout._teardown_price()
    assert [trial for trial, _ in seen] == [tmp_path / "trial"]
    assert seen[0][1] is not threading.current_thread()


def _rollout(tmp_path: Path):
    """A rollout wired enough to drive execute() and build its result."""
    from benchflow.rollout import Rollout, RolloutConfig

    task_path = tmp_path / "task"
    task_path.mkdir()
    (task_path / "instruction.md").write_text("task")
    rollout = Rollout(RolloutConfig(task_path=task_path))
    rollout._rollout_dir = tmp_path / "trial"
    rollout._rollout_dir.mkdir()
    rollout._rollout_name = "trial"
    rollout._started_at = datetime.now()
    rollout._timeout = 60
    rollout._resolved_prompts = ["task"]
    rollout._acp_client = object()
    rollout._session = ACPSession("trial")
    return rollout


async def test_the_stop_reason_reaches_the_result_not_the_trajectory(
    tmp_path, monkeypatch
):
    """The last prompt's stop reason is kept beside the trajectory, so
    acp_trajectory.jsonl and its step count are what they were."""
    rollout = _rollout(tmp_path)

    async def prompts_end(_client, session, prompts, *_args, **_kwargs):
        for prompt in prompts:
            session.record_user_prompt(prompt)
            session.handle_update(
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "done"},
                }
            )
            session.stop_reason = StopReason.END_TURN
            session.mark_prompt_end()
        return _capture_session_trajectory(session), 0

    monkeypatch.setattr(rollout._planes, "execute_prompts", prompts_end)
    trajectory, _ = await rollout.execute(prompts=["task"])
    assert [event["type"] for event in trajectory] == ["user_message", "agent_message"]
    rollout._build_result()
    trial = rollout._rollout_dir
    result = json.loads((trial / "result.json").read_text())
    assert result["agent_outcome"]["stop_reason"] == "end_turn"
    assert result["agent_outcome"]["status"] == "completed"
    assert result["trajectory_summary"]["steps"] == 2
    written = (trial / "trajectory" / "acp_trajectory.jsonl").read_text()
    assert [json.loads(line)["type"] for line in written.splitlines()] == [
        "user_message",
        "agent_message",
    ]


async def test_a_prompt_that_times_out_forgets_the_last_stop_reason(
    tmp_path, monkeypatch
):
    """A first prompt ended end_turn, the next one timed out: the trial did
    not complete."""
    rollout = _rollout(tmp_path)

    async def ends_then_times_out(_client, session, prompts, *_args, **_kwargs):
        session.record_user_prompt(prompts[0])
        if prompts[0] == "first":
            session.stop_reason = StopReason.END_TURN
            session.mark_prompt_end()
            return _capture_session_trajectory(session), 0
        raise _agent_prompt_timeout_error(session, 60)

    monkeypatch.setattr(rollout._planes, "execute_prompts", ends_then_times_out)
    await rollout.execute(prompts=["first"])
    assert rollout._agent_stop_reason == "end_turn"
    with pytest.raises(AgentPromptTimeoutError):
        await rollout.execute(prompts=["second"])
    assert rollout._agent_stop_reason is None
    rollout._error = "Agent timed out after 60s"
    rollout._build_result()
    result = json.loads((rollout._rollout_dir / "result.json").read_text())
    assert result["agent_outcome"]["status"] == "unknown"
    assert result["agent_outcome"]["stop_reason"] is None
