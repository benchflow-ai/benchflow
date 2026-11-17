"""A step that is not the agent's turn and runs out of time is not the agent's timeout.

Guards the audit fix on branch ``audit-phase-timeout`` (finding
LIFE-3, the lead's ruling of 2026-11-13). One handler caught every
``TimeoutError`` of the trial and recorded it as the agent's: a bare timeout
in setup became "Agent timed out after 0s elapsed in rollout (budget 0s)",
category ``timeout``, which is the agent's fault and is never retried.

The step is now decided from what raised the timeout and where the lifecycle
stood, and recorded as ``phase_timeout``, retried like an infrastructure
failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import benchflow.rollout as module
import benchflow.rollout._results as results_module
import benchflow.rollout._verifier_recovery as recovery
from benchflow._utils.attempt_decision import RETRIED, record_decision
from benchflow._utils.result_paths import load_task_results
from benchflow._utils.scoring import (
    AGENT_INTEGRATION,
    API_ERROR,
    ENDING_KEEPS_REWARD,
    ERROR_CATEGORIES,
    INFRA_ERROR,
    PHASE_TIMEOUT,
    PHASE_TIMEOUT_HEADER,
    TIMED_OUT,
    TRIAL_BUDGET,
    classify_error,
)
from benchflow.agents.errors import AgentProtocolError
from benchflow.diagnostics import (
    AgentPromptTimeoutDiagnostic,
    AgentPromptTimeoutError,
    IdleTimeoutDiagnostic,
    IdleTimeoutError,
)
from benchflow.evaluation import Evaluation, RetryConfig
from benchflow.failures import describe
from benchflow.models import RolloutResult
from benchflow.rollout import Rollout, RolloutConfig, _deadline
from benchflow.rollout._deadline import HARD_DEADLINE_ENV
from benchflow.rollout._setup import OracleTimeoutError
from benchflow.sandbox.protocol import ExecTimeoutError
from benchflow.trial_budget import TrialBudgetSpentError


def _rollout(tmp_path: Path, agent: str = "claude-agent-acp") -> Rollout:
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n[agent]\ntimeout_sec = 900\n')
    (task / "instruction.md").write_text("Do the thing.\n")
    rollout = Rollout(RolloutConfig(task_path=task, agent=agent))
    trial = tmp_path / "trial"
    trial.mkdir()
    rollout._rollout_dir = trial
    rollout._rollout_name = "task__1"
    rollout._started_at = datetime.now()
    for name in ("setup", "start", "install_agent", "cleanup"):
        setattr(rollout, name, AsyncMock())
    return rollout


def _saved(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "trial" / "result.json").read_text())


@pytest.mark.parametrize(
    "failing,phase_before,step",
    [
        ("setup", "created", "setup"),
        ("start", "setup", "start"),
        ("install_agent", "started", "install"),
    ],
)
def test_a_timeout_before_the_agent_runs_names_its_step(
    tmp_path, failing, phase_before, step
):
    rollout = _rollout(tmp_path)

    async def before():
        rollout._phase = phase_before
        raise TimeoutError()

    setattr(rollout, failing, AsyncMock(side_effect=before))

    result = asyncio.run(rollout.run())

    assert result.error.startswith(f"{PHASE_TIMEOUT_HEADER} [{step}] after ")
    assert "Agent timed out" not in result.error
    assert result.error_category == PHASE_TIMEOUT
    assert _saved(tmp_path)["error_category"] == PHASE_TIMEOUT
    assert result.rewards is None
    # Retried like an infrastructure failure, by the stored category and by
    # the text alone (a reader that has only the error).
    assert RetryConfig().should_retry(result.error, category=result.error_category)
    assert RetryConfig().should_retry(result.error)
    assert describe(PHASE_TIMEOUT).fault == "infrastructure"


def test_the_detail_of_the_timeout_is_kept_and_cannot_change_the_category(tmp_path):
    rollout = _rollout(tmp_path)

    async def start():
        rollout._phase = "setup"
        raise TimeoutError("Agent idle for 600s; install failed; timed out")

    rollout.start = AsyncMock(side_effect=start)

    result = asyncio.run(rollout.run())

    assert result.error.endswith(": Agent idle for 600s; install failed; timed out")
    assert classify_error(result.error) == PHASE_TIMEOUT


def test_the_agents_own_timeouts_are_still_the_agents(tmp_path):
    rollout = _rollout(tmp_path)
    rollout._phase = "started"

    idle = IdleTimeoutError("Agent idle for 600s", IdleTimeoutDiagnostic())
    assert rollout._step_that_timed_out(idle) is None
    oracle = OracleTimeoutError("oracle/solve.sh", 5)
    assert rollout._step_that_timed_out(oracle) is None
    assert classify_error(str(oracle)) == TIMED_OUT
    # While the agent takes its turn, a timeout is not a step's.
    rollout._phase = "connected"
    assert rollout._step_that_timed_out(TimeoutError()) is None


# Review of PR #141, 2026-11-13. The lifecycle below moves through its phases
# as the real setup(), start() and install_agent() do, with nothing in a
# sandbox, and runs the real verify() up to the step that times out.


def _entering(rollout: Rollout, phase: str):
    async def step(*_args, **_kwargs):
        rollout._phase = phase

    return step


def _lifecycle(tmp_path: Path, agent: str = "claude-agent-acp") -> Rollout:
    rollout = _rollout(tmp_path, agent)
    for name, phase in (
        ("setup", "setup"),
        ("start", "started"),
        ("install_agent", "installed"),
    ):
        setattr(rollout, name, AsyncMock(side_effect=_entering(rollout, phase)))
    return rollout


def _script_lifecycle(tmp_path: Path, monkeypatch, *, timed_out: bool) -> Rollout:
    script = tmp_path / "solve.sh"
    script.write_text("sleep 600\n")
    rollout = _lifecycle(tmp_path, agent=f"script:{script}")
    rollout._timeout = 20
    rollout._agent_cwd = "/app"
    rollout._env = SimpleNamespace()
    rollout._egress_denylist = None
    monkeypatch.setattr(
        rollout._planes, "enforce_agent_egress_firewall", AsyncMock(), raising=False
    )

    async def run(env, data, label, *args, **kwargs):
        event = {"type": "script", "return_code": 124, "timed_out": timed_out}
        return [event], label

    monkeypatch.setattr(module, "_run_script", run, raising=False)
    return rollout


def _verify_times_out(monkeypatch, *, where: str = "capture") -> None:
    """The real verify() raises a bare TimeoutError: in the post-turn integrity
    probe (still phase "installed") or in the workspace capture after it marked
    "verifying"."""

    async def probe(_rollout, phase):
        if where == "probe" and phase == "post":
            raise TimeoutError()

    async def capture(_rollout):
        raise TimeoutError()

    monkeypatch.setattr(module, "_probe_integrity_access", probe)
    monkeypatch.setattr(module, "capture_terminal_workspace", capture)


_DEADLINE = "Agent timed out: still running at the host hard deadline"
# What the ACP session records of a turn that did nothing: the prompt alone.
# The zero-activity judgements of an agent timeout apply to it.
_PROMPTED = [{"type": "user_message", "text": "Do the thing."}]
_BUDGET = {"budget": "tokens", "limits": {"tokens": 5}}


def _ending(name: str) -> BaseException:
    return {
        "prompt timeout": lambda: AgentPromptTimeoutError(
            "Agent prompt timed out after 900s",
            trajectory=[],
            diagnostic=AgentPromptTimeoutDiagnostic(timeout_sec=900.0),
        ),
        "bare timeout": TimeoutError,
        "budget": lambda: TrialBudgetSpentError(_BUDGET),
        "refusal": lambda: AgentProtocolError("ACP error -32603: Internal error"),
        "output cap": lambda: AgentProtocolError(
            "ACP error: the response exceeded the 32,000 output token maximum."
        ),
    }[name]()


@pytest.mark.parametrize(
    "ending",
    ["deadline", "prompt timeout", "bare timeout", "budget", "refusal", "output cap"],
)
def test_a_verify_step_timeout_keeps_the_ending_the_agents_turn_recorded(
    tmp_path, monkeypatch, ending
):
    """Review of PR #141, 2026-11-13 (S2): the agent's turn ended the trial
    (the host deadline stopped it, its own prompt timeout, a bare timeout, the
    trial budget, a model refusal, an output-cap ending), then verify() raised
    a bare TimeoutError. The outer handler wrote "trial step timed out
    [verify]", category phase_timeout, retried: the agent would run and be paid
    for again. The agent's error and category stay.

    The turn did work (tool calls): the third review's ruling (A) makes the
    zero-activity judgement of a timeout part of its ending, and
    test_a_later_failure_changes_nothing_the_turns_ending_decides covers a
    turn that did nothing."""
    rollout = _lifecycle(tmp_path)
    if ending == "deadline":
        monkeypatch.setenv(HARD_DEADLINE_ENV, "0.5")

        async def turn(_steps):
            rollout._trajectory = list(_PROMPTED)
            rollout._n_tool_calls = 2
            await asyncio.sleep(30)

    else:

        async def turn(_steps):
            rollout._trajectory = list(_PROMPTED)
            rollout._n_tool_calls = 2
            raise _ending(ending)

    if ending == "refusal":
        rollout._model_refused_so_far = AsyncMock(return_value=True)
    rollout._run_steps = turn
    _verify_times_out(monkeypatch)

    result = asyncio.run(asyncio.wait_for(rollout.run(), timeout=60))

    expected = {
        "deadline": _DEADLINE,
        "prompt timeout": "Agent prompt timed out after 900s",
        "bare timeout": "Agent timed out after ",
        "budget": "trial budget reached: ",
        "refusal": "ACP error -32603: Internal error",
        "output cap": "ACP error: the response exceeded the 32,000 output",
    }[ending]
    assert result.error.startswith(expected), result.error
    assert PHASE_TIMEOUT_HEADER not in result.error
    assert result.error_category == classify_error(result.error)
    assert result.error_category != PHASE_TIMEOUT
    saved = _saved(tmp_path)
    assert saved["error"] == result.error
    assert saved["error_category"] == result.error_category
    assert saved["phase_timeout_info"] is None
    if ending in ("deadline", "prompt timeout", "bare timeout", "budget"):
        assert not RetryConfig().should_retry(
            result.error, category=result.error_category
        )


def test_a_verify_step_timeout_after_a_script_ran_to_its_limit_keeps_its_timeout(
    tmp_path, monkeypatch
):
    """Review of PR #141, 2026-11-13 (S2): a script stopped at its time limit
    records no error (its workspace is to be graded as it stands), only its
    timeout. A verify step that then timed out made it "trial step timed out
    [verify]"; it stays the script's timeout, never retried, with no score."""
    rollout = _script_lifecycle(tmp_path, monkeypatch, timed_out=True)
    _verify_times_out(monkeypatch)

    result = asyncio.run(rollout.run())

    assert result.error is None
    assert result.error_category == TIMED_OUT
    assert result.rewards is None
    saved = _saved(tmp_path)
    assert saved["agent_timeout_info"]["timeout_sec"] == 20.0
    assert saved["phase_timeout_info"] is None
    assert not RetryConfig().should_retry(result.error, category=result.error_category)


@pytest.mark.parametrize(
    "agent,where,step",
    [
        ("oracle", "install", "install"),
        ("oracle", "workspace trust", "agent set-up"),
        ("script", "egress firewall", "agent set-up"),
        ("oracle", "post-turn probe", "verify"),
        ("claude-agent-acp", "post-turn probe", "verify"),
    ],
)
def test_each_step_outside_the_agents_turn_is_named_by_what_it_is(
    tmp_path, monkeypatch, agent, where, step
):
    """Review of PR #141, 2026-11-13 (S3.1, S3.2): a scripted agent's install
    and the steps BenchFlow runs around its turn were the agent's timeout, and
    a timeout in verify()'s integrity probe after a model agent's turn (phase
    "installed" again after disconnect) was labelled [connect]. Each is now
    the infrastructure's, named by what it is."""
    if agent == "script":
        rollout = _script_lifecycle(tmp_path, monkeypatch, timed_out=False)
    else:
        rollout = _lifecycle(tmp_path, agent=agent)
    monkeypatch.setattr(module, "trust_workspace_repos", AsyncMock())
    monkeypatch.setattr(
        module,
        "_run_oracle",
        AsyncMock(return_value=([{"type": "oracle", "return_code": 0}], "oracle")),
    )
    rollout._run_steps = AsyncMock()
    if where == "install":

        async def install():
            raise TimeoutError()

        rollout.install_agent = AsyncMock(side_effect=install)
    elif where == "workspace trust":
        monkeypatch.setattr(
            module, "trust_workspace_repos", AsyncMock(side_effect=TimeoutError())
        )
    elif where == "egress firewall":
        monkeypatch.setattr(
            rollout._planes,
            "enforce_agent_egress_firewall",
            AsyncMock(side_effect=TimeoutError()),
            raising=False,
        )
    else:
        _verify_times_out(monkeypatch, where="probe")

    result = asyncio.run(rollout.run())

    assert result.error.startswith(f"{PHASE_TIMEOUT_HEADER} [{step}] after "), (
        result.error
    )
    assert "Agent timed out" not in result.error
    assert result.error_category == PHASE_TIMEOUT
    assert _saved(tmp_path)["error_category"] == PHASE_TIMEOUT
    assert RetryConfig().should_retry(result.error, category=result.error_category)


def test_a_scripted_agents_own_time_limit_is_still_the_agents(tmp_path, monkeypatch):
    """Review of PR #141, 2026-11-13 (S3.1): with the exemption for scripted
    agents gone, the oracle's own limit is still the agent's timeout: solve.sh
    out of time raises the typed OracleTimeoutError (a script's limit comes
    back in its trajectory, tests/test_script_agent_timeout_record.py)."""
    rollout = _lifecycle(tmp_path, agent="oracle")
    monkeypatch.setattr(module, "trust_workspace_repos", AsyncMock())
    monkeypatch.setattr(
        module,
        "_run_oracle",
        AsyncMock(side_effect=OracleTimeoutError("oracle/solve.sh", 5)),
    )

    result = asyncio.run(rollout.run())

    assert (
        result.error == "oracle/solve.sh timed out after 5s, the task's agent timeout"
    )
    assert result.error_category == TIMED_OUT
    assert not RetryConfig().should_retry(result.error, category=result.error_category)


def test_a_step_that_timed_out_before_the_trial_clock_started_says_so(tmp_path):
    """Review of PR #141, 2026-11-13 (S3.3): with no trial clock yet the error
    said "after 0s of the trial"; it says the time is unknown."""
    rollout = _rollout(tmp_path)
    rollout._rollout_dir = None
    rollout._started_at = None
    rollout.setup = AsyncMock(side_effect=TimeoutError())

    result = asyncio.run(rollout.run())

    assert result.error.startswith(
        f"{PHASE_TIMEOUT_HEADER} [setup] after an unknown time"
    ), result.error
    assert "after 0s" not in result.error
    assert classify_error(result.error) == PHASE_TIMEOUT
    assert rollout._diagnostics.get("phase_timeout_info").elapsed_sec is None


class _TimeoutWithLimit(TimeoutError):
    timeout_sec = 120


@pytest.mark.parametrize(
    "raised,limit", [(TimeoutError, None), (_TimeoutWithLimit, 120)]
)
def test_a_phase_timeout_is_recorded_in_phase_timeout_info(tmp_path, raised, limit):
    """Review of PR #141, 2026-11-13 (S3.4): a phase timeout had no
    ``*_info`` record in result.json, as the agent's, idle and verifier
    timeouts have: phase_timeout_info names the step, the limit when the
    timeout carries one, and the trial's elapsed seconds."""
    rollout = _lifecycle(tmp_path)

    async def install():
        raise raised()

    rollout.install_agent = AsyncMock(side_effect=install)

    result = asyncio.run(rollout.run())

    info = _saved(tmp_path)["phase_timeout_info"]
    assert info["step"] == "install"
    assert info["timeout_sec"] == limit
    assert isinstance(info["elapsed_sec"], float) and info["elapsed_sec"] >= 0
    assert result.error_category == PHASE_TIMEOUT


@pytest.mark.parametrize("tool_calls", [3, 0])
def test_a_hard_deadline_kill_after_a_prompt_timeout_keeps_the_agents_timeout(
    tmp_path, monkeypatch, tool_calls
):
    """Review of PR #141, 2026-11-13 (F3, the lead's ruling): the agent's
    prompt timed out, then verification wedged and the host hard deadline
    killed the lifecycle. The first review made it infra_failure, retried, the
    agent run and paid for again. Once the agent's own timeout ended the trial
    a later failure keeps that ending: category timeout, not retried, the
    deadline recorded beside it, unscored.

    Third review, 2026-11-13 (A, the lead's ruling): the zero-activity
    judgement is part of that ending. With no tool calls the prompt timeout is
    the agent integration failure it is when nothing fails after it; the
    kill had dropped that judgement and left a plain timeout."""
    monkeypatch.setenv(HARD_DEADLINE_ENV, "0.5")
    rollout = _lifecycle(tmp_path)

    async def turn(_steps):
        rollout._trajectory = list(_PROMPTED)
        rollout._n_tool_calls = tool_calls
        raise _ending("prompt timeout")

    async def verify():
        await asyncio.sleep(30)

    rollout._run_steps = turn
    rollout.verify = verify

    result = asyncio.run(asyncio.wait_for(rollout.run(), timeout=60))

    expected = TIMED_OUT if tool_calls else AGENT_INTEGRATION
    if tool_calls:
        assert result.error == "Agent prompt timed out after 900s", result.error
    else:
        assert result.error.startswith("agent integration failure ["), result.error
    assert result.error_category == expected
    assert result.rewards is None
    saved = _saved(tmp_path)
    assert saved["error_category"] == expected
    assert saved["agent_timeout_info"]["timeout_sec"] == 900.0
    assert (saved["integration_failure_info"] is None) == bool(tool_calls)
    later = saved["post_turn_failure_info"]
    assert (later["step"], later["kind"]) == ("verify", "host_deadline")
    assert later["detail"].startswith("Rollout exceeded host hard deadline")
    assert not RetryConfig().should_retry(result.error, category=result.error_category)
    assert not RetryConfig().should_retry_verifier_error(result.verifier_error)


def test_a_hard_deadline_kill_with_no_ending_before_it_is_an_infrastructure_failure(
    tmp_path, monkeypatch
):
    """Review of PR #141, 2026-11-13 (F3): the ruling's other half. The
    agent's turn ended by itself, then verification wedged and the host hard
    deadline killed the lifecycle: the kill is the trial's ending,
    infra_failure, retried, as before."""
    monkeypatch.setenv(HARD_DEADLINE_ENV, "0.5")
    rollout = _lifecycle(tmp_path)

    async def turn(_steps):
        rollout._trajectory = list(_PROMPTED)
        rollout._n_tool_calls = 3

    async def verify():
        await asyncio.sleep(30)

    rollout._run_steps = turn
    rollout.verify = verify

    result = asyncio.run(asyncio.wait_for(rollout.run(), timeout=60))

    assert result.error.startswith("Rollout exceeded host hard deadline"), result.error
    assert result.error_category == INFRA_ERROR
    saved = _saved(tmp_path)
    assert saved["error_category"] == INFRA_ERROR
    assert saved["post_turn_failure_info"] is None
    assert RetryConfig().should_retry(result.error, category=result.error_category)


# Review of PR #141, 2026-11-13, second round (F1 and the lead's ruling F3):
# once the agent's turn is over and recorded the trial's ending, nothing that
# fails later replaces that ending or its category.

# How each later failure reaches the lifecycle, the kind it is recorded as,
# and its text. On Docker an exec past its limit raises ExecTimeoutError, a
# RuntimeError (the lifecycle's ``except Exception``). verify() itself records
# a connection lost while it captures the workspace as a verifier crash,
# which the verifier retry rule retried; one lost while it disconnects first
# reaches the lifecycle's ``except ConnectionError``.
_LATER = {
    "exec timeout": (
        lambda: ExecTimeoutError(600),
        "capture",
        "exec_timeout",
        "ExecTimeoutError: Command timed out after 600 seconds",
    ),
    "timeout": (
        TimeoutError,
        "capture",
        "timeout",
        "TimeoutError (no message)",
    ),
    "connection lost in capture": (
        lambda: ConnectionError("sandbox connection lost"),
        "capture",
        "verifier_crash",
        "verifier crashed: preparation failed: ConnectionError: sandbox connection lost",
    ),
    "connection lost": (
        lambda: ConnectionError("sandbox connection lost"),
        "disconnect",
        "connection_lost",
        "ConnectionError: sandbox connection lost",
    ),
}

_ENDINGS = {
    "prompt timeout": ("Agent prompt timed out after 900s", TIMED_OUT),
    "bare timeout": ("Agent timed out after ", TIMED_OUT),
    "deadline": (_DEADLINE, TIMED_OUT),
    "budget": ("trial budget reached: ", TRIAL_BUDGET),
}


def _ending_turn(rollout: Rollout, monkeypatch, ending: str):
    if ending == "deadline":
        monkeypatch.setenv(HARD_DEADLINE_ENV, "0.5")

        async def turn(_steps):
            rollout._trajectory = list(_PROMPTED)
            rollout._n_tool_calls = 2
            await asyncio.sleep(30)

    else:

        async def turn(_steps):
            rollout._trajectory = list(_PROMPTED)
            rollout._n_tool_calls = 2
            raise _ending(ending)

    rollout._run_steps = turn


def _verify_fails(rollout: Rollout, monkeypatch, raised, where: str) -> None:
    async def probe(_rollout, _phase):
        return None

    async def capture(_rollout):
        raise raised()

    monkeypatch.setattr(module, "_probe_integrity_access", probe)
    if where == "capture":
        monkeypatch.setattr(module, "capture_terminal_workspace", capture)
    else:
        rollout._muse_credentials_staged = True
        rollout.disconnect = AsyncMock(side_effect=raised())


@pytest.mark.parametrize("later", list(_LATER))
@pytest.mark.parametrize("ending", list(_ENDINGS))
def test_a_later_verify_failure_keeps_the_ending_and_is_recorded_beside_it(
    tmp_path, monkeypatch, ending, later
):
    """Review of PR #141, 2026-11-13 (F1, F3): after the agent's prompt
    timeout, a bare timeout, the host deadline that stopped it or the trial
    budget, the verify step raised ExecTimeoutError (Docker's exec past its
    limit) and the lifecycle's ``except Exception`` replaced the ending,
    infra_failure, retried; a connection lost in the verify step was a
    verifier crash the verifier retry rule retried, or replaced the ending in
    ``except ConnectionError``. Each keeps the ending's error and category, is
    not retried, is recorded in post_turn_failure_info, and leaves the trial
    unscored: no reward was recorded, and the 0 a verifier that ran and gave
    nothing gets is not given."""
    raised, where, kind, detail = _LATER[later]
    prefix, category = _ENDINGS[ending]
    rollout = _lifecycle(tmp_path)
    _ending_turn(rollout, monkeypatch, ending)
    _verify_fails(rollout, monkeypatch, raised, where)

    result = asyncio.run(asyncio.wait_for(rollout.run(), timeout=60))

    assert result.error.startswith(prefix), result.error
    assert result.error_category == category
    assert not RetryConfig().should_retry(result.error, category=result.error_category)
    assert not RetryConfig().should_retry_verifier_error(result.verifier_error)
    assert result.rewards is None
    saved = _saved(tmp_path)
    assert (saved["error"], saved["error_category"]) == (result.error, category)
    assert saved["rewards"] is None
    assert saved["post_turn_failure_info"] == {
        "step": "verify",
        "kind": kind,
        "detail": detail,
    }
    # Nothing that would decide another category is recorded for it.
    assert saved["phase_timeout_info"] is None
    assert saved["transport_error_info"] is None


def test_a_verifier_that_ran_and_gave_nothing_still_scores_a_timed_out_agent_0(
    tmp_path,
):
    """Review of PR #141, 2026-11-13 (F3): the 0 stays for the case it is
    for, a verifier that ran and returned no reward after the agent's prompt
    timeout; nothing is recorded beside the ending."""
    rollout = _lifecycle(tmp_path)

    async def turn(_steps):
        raise _ending("prompt timeout")

    async def verify():
        rollout._rewards = None

    rollout._run_steps = turn
    rollout.verify = verify

    result = asyncio.run(rollout.run())

    assert result.error_category == TIMED_OUT
    assert result.rewards == {"reward": 0.0}
    assert _saved(tmp_path)["post_turn_failure_info"] is None


def _verifier_scores(monkeypatch, reward: float) -> None:
    async def score(rollout):
        rollout._rewards = {"reward": reward}
        rollout._phase = "verified"
        return rollout._rewards

    monkeypatch.setattr(Rollout, "_score_finished_solver", score)


@pytest.mark.parametrize("cleanup_fails", ["raises", "wedges"])
def test_a_reward_recorded_before_a_later_failure_stays(
    tmp_path, monkeypatch, cleanup_fails
):
    """Review of PR #141, 2026-11-13 (F3): the verifier scored the workspace
    the agent's prompt timeout left, then cleanup lost its connection (the
    lifecycle raised out of run()) or wedged until the host hard deadline
    killed it (infra_failure, the reward dropped, retried). The ending, its
    category and the recorded reward stay; the cleanup's failure is recorded
    beside them."""
    rollout = _lifecycle(tmp_path)

    async def turn(_steps):
        rollout._trajectory = list(_PROMPTED)
        rollout._n_tool_calls = 2
        raise _ending("prompt timeout")

    rollout._run_steps = turn
    _verifier_scores(monkeypatch, 0.25)
    if cleanup_fails == "raises":
        rollout.cleanup = AsyncMock(
            side_effect=ConnectionError("teardown transport lost")
        )
        expected = ("cleanup", "connection_lost")
    else:
        monkeypatch.setenv(HARD_DEADLINE_ENV, "0.5")

        async def wedged():
            await asyncio.sleep(30)

        rollout.cleanup = wedged
        expected = ("cleanup", "host_deadline")

    result = asyncio.run(asyncio.wait_for(rollout.run(), timeout=200))

    assert result.error == "Agent prompt timed out after 900s", result.error
    assert result.error_category == TIMED_OUT
    assert result.rewards == {"reward": 0.25}
    saved = _saved(tmp_path)
    assert saved["rewards"] == {"reward": 0.25}
    later = saved["post_turn_failure_info"]
    assert (later["step"], later["kind"]) == expected
    assert not RetryConfig().should_retry(result.error, category=result.error_category)


def test_the_later_failures_text_is_masked_as_the_error_is(tmp_path, monkeypatch):
    """Review of PR #141, 2026-11-13 (F3): post_turn_failure_info keeps the
    later failure's text through the masking result.json's error text gets
    (made-up value)."""
    fake = "sk-proj-" + "FAKE" * 8 + "1234"
    rollout = _lifecycle(tmp_path)
    _ending_turn(rollout, monkeypatch, "prompt timeout")
    _verify_fails(
        rollout,
        monkeypatch,
        lambda: ExecTimeoutError(
            600, f"Command timed out after 600s: curl -H 'Authorization: Bearer {fake}'"
        ),
        "capture",
    )

    asyncio.run(rollout.run())

    raw = (tmp_path / "trial" / "result.json").read_text()
    assert fake not in raw
    detail = _saved(tmp_path)["post_turn_failure_info"]["detail"]
    assert detail.startswith("ExecTimeoutError: Command timed out after 600s: curl")


@pytest.mark.parametrize("timed_out", [True, False])
def test_a_scripts_tear_down_after_its_turn_is_named_and_keeps_its_timeout(
    tmp_path, monkeypatch, timed_out
):
    """Review of PR #141, 2026-11-13 (F4, F1): a timeout while a script's
    egress policy was torn down after it ran was labelled [agent set-up]. It
    is [agent tear-down]; after a script stopped at its limit it keeps that
    timeout (no error text, category timeout, not retried) and is recorded
    beside it."""
    rollout = _script_lifecycle(tmp_path, monkeypatch, timed_out=timed_out)
    rollout._stop_active_egress = AsyncMock(side_effect=TimeoutError())
    rollout.verify = AsyncMock()

    result = asyncio.run(rollout.run())

    saved = _saved(tmp_path)
    if timed_out:
        assert result.error is None
        assert result.error_category == TIMED_OUT
        assert saved["post_turn_failure_info"] == {
            "step": "agent tear-down",
            "kind": "timeout",
            "detail": "TimeoutError (no message)",
        }
        assert not RetryConfig().should_retry(
            result.error, category=result.error_category
        )
    else:
        assert result.error.startswith(f"{PHASE_TIMEOUT_HEADER} [agent tear-down] ")
        assert result.error_category == PHASE_TIMEOUT
        assert saved["phase_timeout_info"]["step"] == "agent tear-down"


class _TwoSteps:
    """A two-step task as the steps user reads it (task/steps.py)."""

    def __init__(self, solution_dir: Path) -> None:
        self.steps = [SimpleNamespace(name="one"), SimpleNamespace(name="two")]
        self.soft_rewards: list = []
        self.stopped_after = None
        self._solution_dir = solution_dir

    def task_for_round(self, index):
        return SimpleNamespace(
            paths=SimpleNamespace(
                solution_dir=self._solution_dir, uses_native_oracle_dir=False
            ),
            config=SimpleNamespace(agent=SimpleNamespace(timeout_sec=5)),
        )

    def record(self, index, rewards):
        return True


@pytest.mark.parametrize(
    "agent,where,step",
    [
        ("oracle", "solution upload", "step set-up"),
        ("oracle", "soft verify", "soft verify"),
        ("nop", "soft verify", "soft verify"),
    ],
)
def test_a_multi_step_tasks_own_steps_are_named(
    tmp_path, monkeypatch, agent, where, step
):
    """Review of PR #141, 2026-11-13 (F4): on a multi-step task a timeout
    while the oracle's next step's solution was uploaded, or while the oracle
    or nop ran a step's soft verify, was labelled [agent set-up]. Each is
    named by what it is."""
    rollout = _lifecycle(tmp_path, agent=agent)
    solution = tmp_path / "solution"
    if where == "solution upload":
        solution.mkdir()
    monkeypatch.setattr(module, "has_steps", lambda _task: True)
    monkeypatch.setattr(module, "StepsUser", lambda _task: _TwoSteps(solution))
    monkeypatch.setattr(module, "trust_workspace_repos", AsyncMock())
    monkeypatch.setattr(
        module,
        "_run_oracle",
        AsyncMock(return_value=([{"type": "oracle", "return_code": 0}], "oracle")),
    )
    rollout._env = SimpleNamespace(upload_dir=AsyncMock(side_effect=TimeoutError()))
    rollout.soft_verify = AsyncMock(side_effect=TimeoutError())

    result = asyncio.run(rollout.run())

    assert result.error.startswith(f"{PHASE_TIMEOUT_HEADER} [{step}] after "), (
        result.error
    )
    assert result.error_category == PHASE_TIMEOUT
    assert _saved(tmp_path)["phase_timeout_info"]["step"] == step


# Third review of PR #141, 2026-11-13 (A, the lead's ruling): the zero-activity
# judgement of an agent timeout is part of the turn's ending. With and without
# a later failure the trial's error, category, retry decision and reward
# handling are the same; the one difference is that a verify step that itself
# failed recorded no reward.

_TURNS = {
    "prompt timeout after activity": ("prompt timeout", 3),
    "bare timeout, no activity": ("bare timeout", 0),
    "prompt timeout, no activity": ("prompt timeout", 0),
    "deadline stop, no activity": ("deadline", 0),
}
_CLEANUP_FAILS = ("cleanup raises", "cleanup wedges")


def _run_turn(path: Path, monkeypatch, turn: str, later: str | None):
    """The turn, then the verify step scoring 0.7 (*later* None), or failing
    (a ``_LATER`` kind), or scoring 0.7 before cleanup fails."""
    path.mkdir()
    ending, tool_calls = _TURNS[turn]
    with monkeypatch.context() as patch:
        rollout = _lifecycle(path)
        if ending == "deadline" or later == "cleanup wedges":
            patch.setenv(HARD_DEADLINE_ENV, "0.5")
            patch.setattr(_deadline, "post_agent_grace_sec", lambda _config: 1.0)

        async def run_turn(_steps):
            rollout._trajectory = list(_PROMPTED)
            rollout._n_tool_calls = tool_calls
            if ending == "deadline":
                await asyncio.sleep(30)
            raise _ending(ending)

        rollout._run_steps = run_turn
        if later in _LATER:
            raised, where, _kind, _detail = _LATER[later]
            _verify_fails(rollout, patch, raised, where)
        else:
            _verifier_scores(patch, 0.7)
        if later == "cleanup raises":
            rollout.cleanup = AsyncMock(
                side_effect=ConnectionError("teardown transport lost")
            )
        elif later == "cleanup wedges":

            async def wedged():
                await asyncio.sleep(30)

            rollout.cleanup = wedged
        result = asyncio.run(asyncio.wait_for(rollout.run(), timeout=60))
    return result, json.loads((path / "trial" / "result.json").read_text())


def _norm(text: str | None) -> str | None:
    """An error with its seconds left out (a bare timeout says its elapsed
    time)."""
    return re.sub(r"\\d+", "N", text) if text is not None else None


@pytest.mark.parametrize("later", [*_LATER, *_CLEANUP_FAILS])
@pytest.mark.parametrize("turn", list(_TURNS))
def test_a_later_failure_changes_nothing_the_turns_ending_decides(
    tmp_path, monkeypatch, turn, later
):
    """Third review of PR #141, 2026-11-13 (A): after a zero-activity agent
    timeout a later failure cleared the zero-activity judgement, so a trial
    that is an agent integration failure (reward withheld) became a plain
    timeout, and a later cleanup failure even kept the verifier's 0.7. The
    trial now comes out as the same turn does with nothing failing after it:
    error, category, retry decision and reward handling, except that a verify
    step that failed recorded no reward."""
    base, base_saved = _run_turn(tmp_path / "base", monkeypatch, turn, None)
    result, saved = _run_turn(tmp_path / "later", monkeypatch, turn, later)

    if turn == "prompt timeout after activity":
        assert (base.error_category, base.rewards) == (TIMED_OUT, {"reward": 0.7})
    elif turn != "deadline stop, no activity":
        # The zero-activity judgement: unscored, its reward withheld.
        assert (base.error_category, base.rewards) == (AGENT_INTEGRATION, None)
        assert base_saved["integration_failure_info"]["reward_withheld"] == {
            "reward": 0.7
        }
    assert _norm(result.error) == _norm(base.error), result.error
    assert result.error_category == base.error_category
    retry = RetryConfig()
    assert retry.should_retry(
        result.error, category=result.error_category
    ) == retry.should_retry(base.error, category=base.error_category)
    assert not retry.should_retry_verifier_error(result.verifier_error)
    assert not retry.reruns_verifier_error_on_resume(result.verifier_error)
    if later in _CLEANUP_FAILS:
        assert result.rewards == base.rewards
        withheld = (base_saved["integration_failure_info"] or {}).get("reward_withheld")
        step = "cleanup"
    else:
        assert result.rewards is None
        withheld = None
        step = "verify"
    assert saved["post_turn_failure_info"]["step"] == step
    if base_saved["integration_failure_info"] is not None:
        assert (
            saved["integration_failure_info"]["cause"]
            == (base_saved["integration_failure_info"]["cause"])
        )
        assert saved["integration_failure_info"]["reward_withheld"] == withheld
    else:
        assert saved["integration_failure_info"] is None


@pytest.mark.parametrize(
    "turn", ["bare timeout, no activity", "deadline stop, no activity"]
)
def test_a_hard_deadline_kill_keeps_the_zero_activity_judgement_of_every_ending(
    tmp_path, monkeypatch, turn
):
    """Third review of PR #141, 2026-11-13 (A): the matching cases of the
    prompt-timeout deadline test for the other endings. Verification wedged
    after the turn and the host hard deadline killed the lifecycle: the trial
    is what the same turn is when verification gives nothing (a bare timeout
    with no activity, an agent integration failure; a deadline stop with no
    activity, what that ending is on this branch), unscored, with the kill
    recorded beside it."""
    ending, tool_calls = _TURNS[turn]
    outcomes = {}
    for case in ("base", "killed"):
        path = tmp_path / case
        path.mkdir()
        with monkeypatch.context() as patch:
            patch.setenv(HARD_DEADLINE_ENV, "0.5")
            patch.setattr(_deadline, "post_agent_grace_sec", lambda _config: 1.0)
            rollout = _lifecycle(path)

            async def run_turn(_steps, rollout=rollout):
                rollout._trajectory = list(_PROMPTED)
                rollout._n_tool_calls = tool_calls
                if ending == "deadline":
                    await asyncio.sleep(30)
                raise _ending(ending)

            async def verify(case=case):
                if case == "killed":
                    await asyncio.sleep(30)

            rollout._run_steps = run_turn
            rollout.verify = verify
            result = asyncio.run(asyncio.wait_for(rollout.run(), timeout=60))
        outcomes[case] = (
            result,
            json.loads((path / "trial" / "result.json").read_text()),
        )

    (base, _), (result, saved) = outcomes["base"], outcomes["killed"]
    if turn == "bare timeout, no activity":
        assert base.error_category == AGENT_INTEGRATION
    assert _norm(result.error) == _norm(base.error), result.error
    assert result.error_category == base.error_category
    assert result.rewards is None
    later = saved["post_turn_failure_info"]
    assert (later["step"], later["kind"]) == ("verify", "host_deadline")
    retry = RetryConfig()
    assert retry.should_retry(
        result.error, category=result.error_category
    ) == retry.should_retry(base.error, category=base.error_category)


# Third review of PR #141, 2026-11-13 (B): a task with a verifier-only
# recovery contract.


@pytest.mark.parametrize("cleanup", ["raises", "wedges"])
def test_a_verifier_crash_after_an_ending_never_runs_the_agent_again(
    tmp_path, monkeypatch, cleanup
):
    """Third review of PR #141, 2026-11-13 (B): on a task with a verifier-only
    recovery contract, the agent's prompt timed out, the verify step lost its
    connection while it captured the workspace, then cleanup raised or wedged
    until the host hard deadline. The verifier crash stayed as it was, which
    the verifier retry rule and a resume answer by running the whole trial,
    agent and all, again. It is recorded beside the ending and kept only in
    its [solver-preserved] form: the retry rule and a resume leave the trial
    as it is, and only a verifier-only re-run reads it."""
    monkeypatch.setattr(recovery, "recovery_ineligible_reason", lambda *a, **k: None)
    rollout = _lifecycle(tmp_path)
    _ending_turn(rollout, monkeypatch, "prompt timeout")
    _verify_fails(
        rollout, monkeypatch, _LATER["connection lost in capture"][0], "capture"
    )
    if cleanup == "raises":
        rollout.cleanup = AsyncMock(
            side_effect=ConnectionError("teardown transport lost")
        )
    else:
        monkeypatch.setenv(HARD_DEADLINE_ENV, "0.5")

        async def wedged():
            await asyncio.sleep(30)

        rollout.cleanup = wedged

    result = asyncio.run(asyncio.wait_for(rollout.run(), timeout=60))

    assert result.error == "Agent prompt timed out after 900s"
    assert result.error_category == TIMED_OUT
    assert result.rewards is None
    crash = (
        "verifier crashed: preparation failed: ConnectionError: sandbox connection lost"
    )
    assert result.verifier_error == f"{recovery.PRESERVED_SOLVER} {crash}"
    retry = RetryConfig()
    assert not retry.should_retry(result.error, category=result.error_category)
    assert not retry.should_retry_verifier_error(result.verifier_error)
    assert not retry.reruns_verifier_error_on_resume(result.verifier_error)
    # What bench eval score's verifier-only re-run reads to run the verifier
    # again without the agent.
    assert recovery.needs_verifier_recovery(result.verifier_error)
    saved = _saved(tmp_path)
    assert saved["post_turn_failure_info"] == {
        "step": "verify",
        "kind": "verifier_crash",
        "detail": crash,
    }
    # A resume loads it as the task's result, which it reuses.
    assert (
        load_task_results(tmp_path)["task"]["verifier_error"] == result.verifier_error
    )
    assert (tmp_path / "trial" / "solver.json").is_file() == (cleanup == "raises")


# Third review of PR #141, 2026-11-13 (LIFE-6): a cleanup that fails with no
# ending recorded before it.


@pytest.mark.parametrize(
    "state", ["setup failed", "agent error", "clean turn", "cancelled"]
)
def test_a_cleanup_failure_leaves_a_result_and_keeps_the_retry(
    tmp_path, monkeypatch, state
):
    """Third review of PR #141, 2026-11-13 (LIFE-6): with no ending before it,
    an exception from cleanup() raised out of run() and the trial left no
    result.json. A trial whose setup or agent had already failed keeps that
    error, its category and its retry decision; a trial with nothing recorded
    is an infrastructure failure, retried, and the reward its verify step
    recorded is withheld, as at the hard deadline's kill. Cancellation still
    cancels."""
    rollout = _lifecycle(tmp_path)
    if state == "setup failed":

        async def start():
            rollout._phase = "setup"
            raise TimeoutError()

        rollout.start = AsyncMock(side_effect=start)
    elif state == "agent error":

        async def turn(_steps):
            raise AgentProtocolError("ACP error -32000: the agent's session broke")

        rollout._run_steps = turn
    else:
        rollout._run_steps = AsyncMock()
        _verifier_scores(monkeypatch, 0.7)
    rollout.cleanup = AsyncMock(
        side_effect=asyncio.CancelledError()
        if state == "cancelled"
        else ConnectionError("teardown transport lost")
    )

    if state == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(rollout.run())
        return
    result = asyncio.run(rollout.run())

    saved = _saved(tmp_path)
    assert saved["error"] == result.error
    assert saved["error_category"] == result.error_category
    assert (
        saved["export_error"]
        == "Post-solver cleanup did not finalize telemetry/evidence"
    )
    retry = RetryConfig()
    if state == "setup failed":
        assert result.error.startswith(f"{PHASE_TIMEOUT_HEADER} [start] ")
        assert result.error_category == PHASE_TIMEOUT
    elif state == "agent error":
        assert result.error.startswith("ACP error -32000"), result.error
        assert result.error_category == classify_error(result.error)
    else:
        assert result.error == (
            "trial cleanup failed: ConnectionError: teardown transport lost"
        )
        assert result.error_category == INFRA_ERROR
        assert result.rewards is None and saved["rewards"] is None
    assert retry.should_retry(result.error, category=result.error_category)


@pytest.mark.asyncio
async def test_the_job_summary_warns_of_trials_with_a_later_failure(
    job_factory, caplog
):
    """Third review of PR #141, 2026-11-13 (C): post_turn_failure_info decides
    no category, so the per-diagnostic summary warnings, which count
    categories, never named it. The summary counts the trials that have one,
    by kind, read from each trial's result.json."""
    job, _ = job_factory(n_tasks=2)

    async def run_one(task_dir, _cfg):
        name = task_dir.name
        trial = job._jobs_dir / job._job_name / f"{name}__1"
        trial.mkdir(parents=True, exist_ok=True)
        later = (
            {"step": "verify", "kind": "exec_timeout", "detail": "ExecTimeoutError"}
            if name == "task-0"
            else None
        )
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": name,
                    "rollout_name": trial.name,
                    "rewards": None,
                    "error": "Agent prompt timed out after 900s",
                    "error_category": TIMED_OUT,
                    "post_turn_failure_info": later,
                }
            )
        )
        # The in-memory result carries no diagnostics: the count must come
        # from the file.
        return RolloutResult(
            task_name=name,
            rollout_name=trial.name,
            error="Agent prompt timed out after 900s",
            error_category=TIMED_OUT,
        )

    job._run_single_task = run_one
    with caplog.at_level(logging.WARNING):
        await job.run()

    assert any(
        "1 tasks (50%) had a step fail after the agent's turn ended the trial "
        "(1 exec_timeout)" in message
        for message in caplog.messages
    ), caplog.messages


# Fourth delta on PR #141, 2026-11-13 (the lead's ruling on resume): a
# kept-ending trial is final on resume too.

_KEPT_ENDING = {
    "rewards": None,
    "error": "Agent prompt timed out after 900s",
    "error_category": TIMED_OUT,
    "verifier_error": None,
    "post_turn_failure_info": {
        "step": "verify",
        "kind": "exec_timeout",
        "detail": "ExecTimeoutError: Command timed out after 600 seconds",
    },
}


def _resumed(
    tmp_path: Path,
    monkeypatch,
    task_0: dict,
    caplog,
    *,
    task_1: dict | None = None,
    retried: bool = False,
):
    """A two-task job whose task-0 trial ended as *task_0* says (task-1's as
    *task_1* says, when given), resumed through Evaluation.resume(): the
    tasks it ran again, and its result. With *retried* the run recorded that
    it retried task-0's attempt, and stopped before another attempt."""
    from tests.test_resume_settings_match import _fake_runner, _first_job

    job_dir = _first_job(tmp_path, job_name="j")
    for task, ending in (("task-0", task_0), ("task-1", task_1 or {})):
        trial = job_dir / f"{task}__abcd1234"
        saved = json.loads((trial / "result.json").read_text())
        (trial / "result.json").write_text(json.dumps({**saved, **ending}))
    if retried:
        record_decision(job_dir / "task-0__abcd1234", RETRIED, task="task-0", attempt=1)
    job = Evaluation.resume(job_dir)
    job._run_single_task = AsyncMock(side_effect=_fake_runner(job))
    with caplog.at_level(logging.INFO):
        result = asyncio.run(job.run())
    ran = [call.args[0].name for call in job._run_single_task.call_args_list]
    return job_dir, result, ran


_RESUME_SENTENCE = (
    "1 reused trial(s) whose verify step failed are not run again, because "
    "the agent's turn had already ended them; `bench regrade "
)


def test_a_resume_does_not_run_a_kept_ending_trial_again(tmp_path, monkeypatch, caplog):
    """Fourth delta on PR #141, 2026-11-13 (the lead's ruling): a trial whose
    agent's turn ended it and whose verify step then failed is unscored with
    no verifier error, so load_task_results left it out and `bench eval
    resume` ran the trial again, paying for the agent again. It is reused,
    counted once as errored with its category, still counted by the
    post-turn warning, and the resume says why and that regrade re-runs the
    verifier only."""
    job_dir, result, ran = _resumed(tmp_path, monkeypatch, _KEPT_ENDING, caplog)

    assert ran == []
    assert (result.reused, result.ran) == (2, 0)
    assert (result.passed, result.errored) == (1, 1)
    summary = json.loads((job_dir / "summary.json").read_text())
    assert summary["error_categories"] == {TIMED_OUT: 1}
    messages = " ".join(caplog.messages)
    assert _RESUME_SENTENCE in messages, messages
    assert "re-runs only the verifier on a trial whose final workspace" in messages
    assert (
        "1 tasks (50%) had a step fail after the agent's turn ended the trial "
        "(1 exec_timeout)" in messages
    )


def test_a_resume_still_runs_a_plain_errored_trial_again(tmp_path, monkeypatch, caplog):
    """Fourth delta on PR #141, 2026-11-13: the other half of the ruling. A
    trial that errored with no ending of the agent's turn before it (here a
    cleanup failure, infra_failure) is still run again by a resume."""
    _, result, ran = _resumed(
        tmp_path,
        monkeypatch,
        {
            "rewards": None,
            "error": "trial cleanup failed: ConnectionError: teardown transport lost",
            "error_category": INFRA_ERROR,
            "verifier_error": None,
            "post_turn_failure_info": None,
        },
        caplog,
    )

    assert ran == ["task-0"]
    assert (result.reused, result.ran) == (1, 1)
    assert _RESUME_SENTENCE not in " ".join(caplog.messages)


def test_a_resume_reuses_a_kept_ending_on_a_recoverable_task(
    tmp_path, monkeypatch, caplog
):
    """Fourth delta on PR #141, 2026-11-13: a kept ending on a task with
    verifier-only recovery keeps its verifier crash in the [solver-preserved]
    form the third review defined: reused (never the agent again), its
    verifier-only re-run still possible, and the resume says so as for any
    kept ending."""
    crash = (
        f"{recovery.PRESERVED_SOLVER} verifier crashed: preparation failed: "
        "ConnectionError: sandbox connection lost"
    )
    _, result, ran = _resumed(
        tmp_path,
        monkeypatch,
        {
            **_KEPT_ENDING,
            "verifier_error": crash,
            "post_turn_failure_info": {
                "step": "verify",
                "kind": "verifier_crash",
                "detail": crash.removeprefix(f"{recovery.PRESERVED_SOLVER} "),
            },
        },
        caplog,
    )

    assert ran == []
    assert result.reused == 2
    assert recovery.needs_verifier_recovery(crash)
    assert _RESUME_SENTENCE in " ".join(caplog.messages)


@pytest.mark.parametrize("include_terminal_errors", [False, True])
def test_which_unscored_results_a_resume_counts_as_finished(
    tmp_path, include_terminal_errors
):
    """Fourth delta on PR #141, 2026-11-13: load_task_results counts a kept
    ending as finished in every mode; a plain errored result stays finished
    only where terminal errors are (a sequential-shared job), as before."""
    for name, extra in (
        ("kept", _KEPT_ENDING),
        ("errored", {"rewards": None, "error": "install failed"}),
    ):
        trial = tmp_path / f"{name}__1"
        trial.mkdir()
        (trial / "result.json").write_text(
            json.dumps({"task_name": name, "rollout_name": trial.name, **extra})
        )

    loaded = load_task_results(
        tmp_path, include_terminal_errors=include_terminal_errors
    )

    assert "kept" in loaded
    assert ("errored" in loaded) == include_terminal_errors


# Follow-up to PR #141, 2026-11-13 (the lead's ruling refining resume): a kept
# ending is final on resume only when the trial is scored under it. An ending
# that leaves the trial unscored anyway is run again, as it is with nothing
# failing after it.

_STALL = (
    "provider api error [provider_stall/transient] the agent ran out of time "
    "while 1 model request(s) had no response; the oldest was pending 181.0s\n"
    "next: check the model endpoint and trajectory/provider_requests.jsonl, "
    "then resume the job"
)
_CRASH = (
    "verifier crashed: preparation failed: ConnectionError: sandbox connection lost"
)
# What the rollout writes for each ending after its verify step raised
# ExecTimeoutError (the last one: lost its connection on a task whose
# verifier can be recovered).
_UNSCORED_KEPT_ENDINGS = {
    "provider stall": {**_KEPT_ENDING, "error": _STALL, "error_category": API_ERROR},
    "no activity, transient provider errors": {
        **_KEPT_ENDING,
        "error": "provider api error [rate_limit/transient] HTTP 429 on 3/3 requests",
        "error_category": API_ERROR,
    },
    "no activity, agent integration failure": {
        **_KEPT_ENDING,
        "error": (
            "agent integration failure [empty_trajectory]: no event after the "
            "prompt (unknown time)"
        ),
        "error_category": AGENT_INTEGRATION,
        "integration_failure_info": {
            "cause": "empty_trajectory",
            "evidence": "no event after the prompt (unknown time)",
            "evidence_source": "trajectory",
            "tool_calls": 0,
            "agent_messages": 0,
            "agent_thoughts": 0,
            "output_tokens": 0,
            "agent_seconds": None,
            "reward_withheld": None,
        },
    },
    "provider stall, verifier crash on a recoverable task": {
        **_KEPT_ENDING,
        "error": _STALL,
        "error_category": API_ERROR,
        "verifier_error": f"{recovery.PRESERVED_SOLVER} {_CRASH}",
        "post_turn_failure_info": {
            "step": "verify",
            "kind": "verifier_crash",
            "detail": _CRASH,
        },
    },
}


@pytest.mark.parametrize("ending", list(_UNSCORED_KEPT_ENDINGS))
def test_a_resume_runs_an_ending_the_trial_is_unscored_under_again(
    tmp_path, monkeypatch, caplog, ending
):
    """Follow-up to PR #141, 2026-11-13 (the lead's ruling): load_task_results
    counted every result with post_turn_failure_info as finished, so a resume
    reused a trial whose ending leaves it unscored anyway once its verify step
    had failed: a provider stall or a no-activity ending with transient
    provider errors, which the run itself retries, or an agent integration
    failure, and on a task whose verifier can be recovered its
    [solver-preserved] crash. With nothing failing after the turn, the same
    trial is unscored and a resume runs it again. It now runs again."""
    _, result, ran = _resumed(
        tmp_path, monkeypatch, _UNSCORED_KEPT_ENDINGS[ending], caplog
    )

    assert ran == ["task-0"]
    assert (result.reused, result.ran) == (1, 1)
    assert "are not run again, because" not in " ".join(caplog.messages)


def test_the_resume_message_counts_only_the_kept_endings_it_reuses(
    tmp_path, monkeypatch, caplog
):
    """Follow-up to PR #141, 2026-11-13: a provider stall (run again) and a
    prompt timeout (reused), each with a failed verify step. The resume says
    it reused one such trial, the timeout."""
    _, result, ran = _resumed(
        tmp_path,
        monkeypatch,
        _UNSCORED_KEPT_ENDINGS["provider stall"],
        caplog,
        task_1=_KEPT_ENDING,
    )

    assert ran == ["task-0"]
    assert (result.reused, result.ran) == (1, 1)
    assert _RESUME_SENTENCE in " ".join(caplog.messages), caplog.messages


@pytest.mark.parametrize("ending", ["provider stall", "prompt timeout"])
def test_a_resume_runs_a_kept_ending_the_run_retried_again(
    tmp_path, monkeypatch, caplog, ending
):
    """Follow-up to PR #141, 2026-11-13, with the final-attempt rule: the run
    retried a kept-ending attempt (a provider stall, which it retries) and
    stopped before another attempt. An attempt the run retried is never the
    task's result, so the kept-ending rule cannot reuse it, even for an ending
    the rule keeps (a prompt timeout): a resume runs the task again."""
    task_0 = (
        _UNSCORED_KEPT_ENDINGS["provider stall"]
        if ending == "provider stall"
        else _KEPT_ENDING
    )
    _, result, ran = _resumed(tmp_path, monkeypatch, task_0, caplog, retried=True)

    assert ran == ["task-0"]
    assert (result.reused, result.ran) == (1, 1)


# Each ending: how the turn ends, its tool calls, and what else the trial saw
# (every model request failing, a request with no response, a message with no
# model tokens).
_RESUME_ROWS: dict[str, tuple[str, int, list[int | None] | str | None]] = {
    "prompt timeout after activity": ("prompt timeout", 3, None),
    "idle timeout": ("idle timeout", 2, None),
    "agent timeout naming no cause": ("unnamed timeout", 2, None),
    "a message, no tokens or tool calls": ("bare timeout", 0, "message"),
    "bare timeout, no activity": ("bare timeout", 0, None),
    "no activity, transient provider errors": ("bare timeout", 0, [429, 429, 503]),
    "no activity, permanent provider errors": ("bare timeout", 0, [401, 401]),
    "deadline stop": ("deadline", 2, None),
    "budget stop": ("budget", 2, None),
    "model refusal": ("refusal", 2, None),
    "output cap": ("output cap", 2, None),
    "provider stall": ("prompt timeout", 2, "stall"),
}
_SCORED_ENDINGS = (
    "prompt timeout after activity",
    "idle timeout",
    "agent timeout naming no cause",
    "deadline stop",
    "budget stop",
    "model refusal",
    "output cap",
)
# The fields of result.json that decide a resume.
_DECIDING = (
    "rewards",
    "error",
    "error_category",
    "verifier_error",
    "verifier_error_category",
    "post_turn_failure_info",
    "model_refusal",
    "integration_failure_info",
)


class _StalledProxy:
    """A model proxy with one request unanswered for 181s."""

    def stalled_model_requests(self):
        return (1, 181.0)


def _row_result(path: Path, monkeypatch, row: str, later: str | None) -> dict:
    """result.json of a trial that ended as *row*, its verify step scoring
    0.7 (*later* None) or raising ExecTimeoutError, or scoring 0.7 before
    cleanup raises."""
    from benchflow.rollout._usage import _provider_api_failure_summary_from_runtime
    from tests.test_api_error_capture import _runtime

    path.mkdir()
    ending, tool_calls, seen = _RESUME_ROWS[row]
    with monkeypatch.context() as patch:
        rollout = _lifecycle(path)
        if ending == "deadline":
            patch.setenv(HARD_DEADLINE_ENV, "0.5")
            patch.setattr(_deadline, "post_agent_grace_sec", lambda _config: 1.0)
        if ending == "refusal":
            patch.setattr(
                rollout, "_model_refused_so_far", AsyncMock(return_value=True)
            )
            # result.json's model_refusal, read from the proxy's record by
            # the rule _model_refused_so_far reads.
            patch.setattr(
                results_module,
                "refusal_ending",
                lambda *_args: {"by": "stop_reason", "stop_reason": "refusal"},
            )
        if seen == "stall":
            rollout._usage_runtime = SimpleNamespace(server=_StalledProxy())
            rollout._config.agent_idle_timeout = 30
        elif seen is not None:
            if isinstance(seen, list):
                rollout._api_failure_summary_cached = (
                    _provider_api_failure_summary_from_runtime(_runtime(seen))
                )
            rollout._agent_env = {"BENCHFLOW_PROVIDER_NAME": "litellm"}
            rollout._usage_metrics = {"total_tokens": 0}
        said = [{"type": "agent_message", "text": "Reading the task."}]

        async def run_turn(_steps):
            rollout._trajectory = [*_PROMPTED, *(said if seen == "message" else [])]
            rollout._n_tool_calls = tool_calls
            if ending == "deadline":
                await asyncio.sleep(30)
            if ending == "idle timeout":
                raise IdleTimeoutError(
                    "Agent idle for 600s with no new tool call",
                    IdleTimeoutDiagnostic(idle_timeout_sec=600),
                )
            if ending == "unnamed timeout":
                raise TimeoutError("the agent's watchdog fired")
            raise _ending(ending)

        patch.setattr(rollout, "_run_steps", run_turn)
        if later == "verify fails":
            _verify_fails(rollout, patch, _LATER["exec timeout"][0], "capture")
        else:
            _verifier_scores(patch, 0.7)
        if later == "cleanup raises":
            rollout.cleanup = AsyncMock(
                side_effect=ConnectionError("teardown transport lost")
            )
        asyncio.run(asyncio.wait_for(rollout.run(), timeout=60))
    return json.loads((path / "trial" / "result.json").read_text())


@pytest.mark.parametrize("row", list(_RESUME_ROWS))
def test_a_resume_decides_a_kept_ending_as_without_the_later_failure(
    tmp_path, monkeypatch, caplog, row
):
    """Follow-up to PR #141, 2026-11-13 (the lead's ruling): the same ending
    with nothing failing after the turn, with its verify step raising
    ExecTimeoutError, and with its cleanup raising, gets the same resume
    decision about the agent. A trial scored under its ending is reused; one
    its ending leaves unscored anyway is run again, whether or not the run
    retries that ending (a no-activity ending with permanent provider errors
    and a bare timeout with no activity are not retried)."""
    ran = {}
    scored = row in _SCORED_ENDINGS
    for later in (None, "verify fails", "cleanup raises"):
        saved = _row_result(tmp_path / f"trial-{later}", monkeypatch, row, later)
        assert bool(saved["post_turn_failure_info"]) == (later is not None)
        if later is None:
            # The rollout's own decision, and the table a resume reads for
            # the ending when a later step failed, agree.
            assert (saved["rewards"] is not None) == scored, saved["rewards"]
            assert ENDING_KEEPS_REWARD[saved["error_category"]] == scored
        resumed = tmp_path / f"resume-{later}"
        resumed.mkdir()
        _, _, ran[later] = _resumed(
            resumed, monkeypatch, {key: saved.get(key) for key in _DECIDING}, caplog
        )

    assert ran == dict.fromkeys(ran, [] if scored else ["task-0"])


def test_every_error_category_says_whether_an_ending_under_it_keeps_its_reward():
    """Follow-up to PR #141, 2026-11-13 (the lead's condition): a resume
    reads ENDING_KEEPS_REWARD for a trial whose later step failed after its
    turn's ending, so every category the code defines is classified there,
    and a new category fails here until it is. The table test above checks
    the endings a turn records against what the rollout writes for them."""
    assert set(ENDING_KEEPS_REWARD) == ERROR_CATEGORIES
