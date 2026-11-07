"""Provider errors and harness failures are their own causes, never the
model's failure, and every trial counts its failed calls and retries.

Guards what a published run showed: provider 5xx and 429 answers the agent
survived were invisible (a trial's failed-call counts existed only when every
call failed), a turn that ended on a failed call was scored as the model's
answer, a 5xx behind the agent's ACP error read as the agent's own error, and
an agent process that died was labelled a memory kill with no evidence.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.evaluation_results import model_call_summary
from benchflow._utils.scoring import (
    API_ERROR,
    INFRA_ERROR,
    PIPE_CLOSED,
    classify_error,
    classify_score_outcome,
)
from benchflow.diagnostics import (
    ProviderApiErrorDiagnostic,
    RolloutDiagnostics,
    TransportClosedDiagnostic,
)
from benchflow.evaluation import RetryConfig
from benchflow.failures import cause_of
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.rollout._results import _build_rollout_result
from benchflow.rollout._usage import _provider_failure_from_status
from benchflow.trajectories.model_calls import write_model_calls


def _call(status, purpose="agent"):
    answer = (
        {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
        if status == 200
        else {"error": {"message": "upstream"}}
    )
    return {
        "request": {"body": {"model": "m", "messages": []}},
        "response": {"status_code": status, "body": answer},
        "metadata": {"call_purpose": purpose},
    }


def _trial(tmp_path, statuses, attempts=None):
    trial = tmp_path / "task__1"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "trajectory/llm_trajectory.jsonl").write_text(
        "".join(json.dumps(_call(s)) + "\n" for s in statuses)
    )
    if attempts is not None:
        (trial / "trajectory/provider_requests.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in attempts)
        )
    return trial


def _attempt(request, status):
    return [
        {
            "event": "upstream_start",
            "request_id": request,
            "attempt_id": f"{request}{status}",
        },
        {"event": "upstream_end", "request_id": request, "status": status},
    ]


def test_a_trial_counts_its_failed_calls_and_retries(tmp_path):
    """Two 500s and a 429 the agent survived: the agent tried again after
    each; the proxy itself sent one request three times upstream."""
    trial = _trial(
        tmp_path,
        [200, 500, 500, 200, 429, 200],
        attempts=[
            *_attempt("a", 200),
            *_attempt("b", 502),
            *_attempt("b", None),
            *_attempt("b", 200),
        ],
    )
    summary = write_model_calls(trial)
    assert summary is not None
    assert summary["http_status_counts"] == {"200": 3, "500": 2, "429": 1}
    assert summary["http_errors"] == 3
    assert summary["agent_retries"] == 3
    assert (
        summary["upstream_attempts"],
        summary["upstream_retries"],
        summary["upstream_errors"],
    ) == (4, 2, 2)

    _build_rollout_result(
        trial,
        task_name="task",
        rollout_name=trial.name,
        agent="test",
        agent_name="test",
        model="m",
        n_tool_calls=3,
        prompts=["task"],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
    )
    outcome = json.loads((trial / "result.json").read_text())["agent_outcome"]
    assert outcome["status"] == "completed"
    assert (
        outcome["http_errors"],
        outcome["agent_retries"],
        outcome["upstream_retries"],
    ) == (3, 3, 2)
    assert outcome["http_status_counts"] == {"200": 3, "500": 2, "429": 1}


def test_two_calls_upstream_for_one_request_are_not_a_retry(tmp_path):
    """The proxy's first request of a trial makes a lookup upstream before
    the model call (seen in every Claude Code trial of the deterministic
    tier): both answer 200, and nothing was retried."""
    trial = _trial(
        tmp_path,
        [200],
        attempts=[*_attempt("a", 200), *_attempt("a", 200), *_attempt("b", 200)],
    )
    summary = write_model_calls(trial)
    assert summary is not None
    assert (summary["upstream_attempts"], summary["upstream_retries"]) == (3, 0)


def test_without_the_proxys_records_the_counts_are_unknown(tmp_path):
    summary = write_model_calls(_trial(tmp_path, [200, 200]))
    assert summary is not None
    assert summary["agent_retries"] == 0
    assert summary["upstream_attempts"] is None


def _rollout(trial, rewards):
    rollout = Rollout(
        RolloutConfig(task_path=Path("task"), agent="claude-agent-acp", model="m")
    )
    rollout._rollout_dir = trial
    rollout._executed_prompts = ["task"]
    rollout._agent_env = {"BENCHFLOW_PROVIDER_NAME": "litellm"}
    rollout._usage_metrics = {"total_tokens": 4000, "n_output_tokens": 200}
    rollout._n_tool_calls = 4
    rollout._rewards = rewards
    rollout._diagnostics = RolloutDiagnostics()
    return rollout


@pytest.mark.parametrize(
    "statuses,kind",
    [([200, 500, 200, 500], "provider_error"), ([200, 503, 429], "rate_limit")],
)
def test_a_turn_that_ended_on_a_failed_call_is_the_providers_error(
    tmp_path, statuses, kind
):
    """The verifier scored the workspace 0, but the model never answered the
    agent's last call: an api_error, unscored and retried as transient."""
    trial = _trial(tmp_path, statuses)
    write_model_calls(trial)
    rollout = _rollout(trial, {"reward": 0.0})

    rollout._maybe_classify_api_error()

    last = statuses[-1]
    failed = sum(s >= 400 for s in statuses)
    assert rollout._error == (
        f"provider api error [{kind}/transient] HTTP {last} (the agent's last "
        f"model call) on {failed}/{len(statuses)} requests"
    )
    assert rollout._rewards is None
    info = rollout._diagnostics.get(ProviderApiErrorDiagnostic.field)
    assert info.reward_withheld == {"reward": 0.0}
    assert info.status_counts == {
        str(s): statuses.count(s) for s in set(statuses) if s >= 400
    }
    assert classify_error(rollout._error) == API_ERROR
    assert RetryConfig().should_retry(rollout._error)
    record = {"error": rollout._error, "error_category": API_ERROR, "rewards": None}
    assert classify_score_outcome(record) == "errored"
    cause = cause_of(record)
    assert cause is not None and cause.fault == "infrastructure"


def test_failures_the_agent_survived_leave_the_score_alone(tmp_path):
    trial = _trial(tmp_path, [500, 500, 200])
    write_model_calls(trial)
    rollout = _rollout(trial, {"reward": 0.0})

    rollout._maybe_classify_api_error()

    assert rollout._error is None and rollout._rewards == {"reward": 0.0}


def test_an_error_already_recorded_is_not_replaced(tmp_path):
    trial = _trial(tmp_path, [200, 500])
    write_model_calls(trial)
    rollout = _rollout(trial, None)
    rollout._error = "Agent idle for 600s with no new tool call, message, or thought"

    rollout._maybe_classify_api_error()

    assert rollout._error.startswith("Agent idle for 600s")


@pytest.mark.parametrize("status", [500, 502, 503, 504, 529])
def test_a_server_error_behind_the_agents_acp_error_is_the_providers(status):
    """A 500 behind "ACP error -32603" read as acp_error, the agent's own
    error; as a 503 always did, it is the provider's now."""
    failure = _provider_failure_from_status(status)
    assert failure is not None
    error = f"ACP error -32603: Internal error | {failure.error_suffix}"
    assert error.endswith(f"provider unavailable (HTTP {status})")
    assert classify_error(error) == INFRA_ERROR
    assert RetryConfig().should_retry(error)
    cause = cause_of({"error": error, "error_category": INFRA_ERROR, "rewards": None})
    assert cause is not None
    assert (cause.key, cause.label, cause.fault) == (
        "provider_unavailable",
        f"the model provider failed with HTTP {status}",
        "infrastructure",
    )


def test_a_lost_pty_websocket_is_the_transports_not_the_models():
    error = "PTY closed by the peer: No PONG received after 60.0 seconds"
    assert classify_error(error) == PIPE_CLOSED
    assert RetryConfig().should_retry(error)
    cause = cause_of({"error": error, "error_category": PIPE_CLOSED, "rewards": None})
    assert cause is not None
    assert (cause.key, cause.fault) == ("pipe_closed", "infrastructure")


@pytest.mark.parametrize(
    "kills,label",
    [
        (None, "the agent process exited with code 137"),
        (
            0,
            "the agent process exited with code 137, and the sandbox's cgroup "
            "counted no OOM kill",
        ),
        (
            2,
            "the agent process exited with code 137, and the sandbox's cgroup "
            "counted 2 OOM kills",
        ),
    ],
)
def test_an_exit_is_a_memory_kill_only_with_the_cgroups_word(kills, label):
    info = TransportClosedDiagnostic(process_exit_code=137, oom_kills=kills)
    record = {
        "error": "PTY closed by the peer: agent process exited with code 137",
        "error_category": PIPE_CLOSED,
        "rewards": None,
        "transport_error_info": info.to_dict(),
    }
    cause = cause_of(record)
    assert cause is not None and cause.label == label
    assert ("oom_kills" in info.to_dict()) is (kills is not None)


async def test_the_health_probe_reads_the_cgroups_oom_kills():
    exec_ = AsyncMock(
        return_value=SimpleNamespace(
            stdout="__BENCHFLOW_HEALTH_OK__\nlow 0\nhigh 0\nmax 4\noom 2\n"
            "oom_kill 2\noom_group_kill 0\n",
            stderr="",
            return_code=0,
        )
    )
    rollout = SimpleNamespace(
        _env=SimpleNamespace(exec=exec_), _diagnostics=RolloutDiagnostics()
    )
    rollout._diagnostics.set(TransportClosedDiagnostic(process_exit_code=137))

    await Rollout._probe_sandbox_health(rollout)  # type: ignore[arg-type]

    diag = rollout._diagnostics.transport_closed
    assert diag is not None and diag.oom_kills == 2 and diag.sandbox_reachable


def test_the_job_summary_totals_the_calls_and_retries():
    def outcome(errors, counts, retries=0, upstream=0, terminal=False):
        return {
            "agent_outcome": {
                "http_errors": errors,
                "http_status_counts": counts,
                "agent_retries": retries,
                "upstream_retries": upstream,
                "terminal_provider_error": terminal,
            }
        }

    block = model_call_summary(
        {
            "a": outcome(11, {"200": 528, "500": 11}, retries=11),
            "b": outcome(0, {"200": 40}),
            "c": outcome(1, {"200": 9, "503": 1}, upstream=2, terminal=True),
            "subscription": {"agent_outcome": {"http_errors": None}},
        }
    )["model_calls"]

    assert block["trials_with_status"] == 3
    assert (block["http_errors"], block["trials_with_http_errors"]) == (12, 2)
    assert block["http_status_counts"] == {"200": 577, "500": 11, "503": 1}
    assert (block["agent_retries"], block["upstream_retries"]) == (11, 2)
    assert block["terminal_provider_errors"] == 1


def test_an_acp_error_long_after_a_survived_500_stays_the_agents():
    """The agent got past a 500 on its third call, made 40 more calls that
    answered, then Claude Code ended the turn on its own output cap (an ACP
    -32603). Only the last answered call says whether the provider stopped
    it: here none did, so the error is the agent's, not 'provider
    unavailable'."""
    from types import SimpleNamespace

    from benchflow._utils.scoring import ACP_ERROR, classify_error
    from benchflow.acp.client import ACPError
    from benchflow.rollout import Rollout
    from benchflow.rollout._usage import _provider_failure_from_runtime

    def runtime(statuses):
        exchanges = [
            SimpleNamespace(response=SimpleNamespace(status_code=s)) for s in statuses
        ]
        return SimpleNamespace(
            server=SimpleNamespace(trajectory=SimpleNamespace(exchanges=exchanges))
        )

    assert _provider_failure_from_runtime(runtime([200, 200, 500] + [200] * 40)) is None
    last = _provider_failure_from_runtime(runtime([200, 200, 500, None]))
    assert last is not None and last.status == 500
    rollout = Rollout.__new__(Rollout)
    rollout._provider_failure_cached = None
    rollout._provider_auth_status_cached = None
    rollout._config = SimpleNamespace(
        primary_agent="claude-agent-acp", primary_model=None
    )
    error = rollout._classify_acp_error(
        ACPError(
            -32603,
            "Internal error: API Error: Claude's response exceeded the 32000 "
            "output token maximum.",
        )
    )
    assert classify_error(error) == ACP_ERROR
