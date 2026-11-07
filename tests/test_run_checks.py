"""How each trial ended, and what its model calls carried (benchflow.run_checks)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pytest

from benchflow.acp.runtime import _agent_prompt_timeout_error
from benchflow.acp.session import ACPSession
from benchflow.acp.types import StopReason
from benchflow.diagnostics import AgentPromptTimeoutError
from benchflow.rollout._results import _build_rollout_result
from benchflow.trajectories._capture import _capture_session_trajectory
from benchflow.trajectories.model_calls import write_model_calls


def _proxy_call(
    finish: str = "tool_calls",
    *,
    max_tokens: int | None = 128000,
    model: str = "benchflow-anthropic-claude-sonnet-5",
    served: str = "claude-sonnet-5-20260301",
    route: str = "anthropic/claude-sonnet-5",
    effort: str | None = "high",
    status: int = 200,
) -> dict:
    """One line of the proxy's trajectory/llm_trajectory.jsonl, as
    providers/litellm_logging.py writes it for a chat-completions call."""
    body: dict = {"model": model, "messages": [{"role": "user", "content": "task"}]}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if effort is not None:
        body["reasoning_effort"] = effort
    return {
        "request": {"body": body, "timestamp": "2026-11-05T10:00:00+00:00"},
        "response": {
            "status_code": status,
            "body": {
                "model": served,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "x"},
                        "finish_reason": finish,
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        },
        "metadata": {
            "call_purpose": "agent",
            "request_model": model,
            "provider_model": route,
        },
        "duration_ms": 1000,
    }


def _write_proxy_log(trial: Path, calls: list[dict]) -> None:
    (trial / "trajectory").mkdir(parents=True, exist_ok=True)
    (trial / "trajectory" / "llm_trajectory.jsonl").write_text(
        "".join(json.dumps(call) + "\n" for call in calls)
    )
    write_model_calls(trial)


def _session_trajectory() -> list[dict]:
    session = ACPSession("trial")
    session.record_user_prompt("task")
    session.handle_update(
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "answer"},
        }
    )
    session.mark_prompt_end()
    return _capture_session_trajectory(session)


def _result(
    trial: Path,
    calls: list[dict] | None = None,
    *,
    stop_reasons: list[str | None] | None = None,
    error: str | None = None,
    model: str = "anthropic/claude-sonnet-5",
) -> dict:
    """result.json of a trial whose proxy recorded *calls*: teardown writes
    the model calls, then the result is built, as a rollout does."""
    trial.mkdir(parents=True, exist_ok=True)
    if calls:
        _write_proxy_log(trial, calls)
    last = (stop_reasons or [None])[-1]
    _build_rollout_result(
        trial,
        task_name="task",
        rollout_name=trial.name,
        agent="claude-agent-acp",
        agent_name="claude-agent-acp",
        model=model,
        n_tool_calls=3,
        prompts=["task"],
        error=error,
        verifier_error=None,
        trajectory=_session_trajectory(),
        partial_trajectory=False,
        stop_reason=last,
        stop_reasons=stop_reasons,
        rewards={"reward": 0.0},
        started_at=datetime.now(),
        timing={},
        total_tokens=5000,
    )
    return json.loads((trial / "result.json").read_text())


# -- how each turn ended ---------------------------------------------------


def test_the_session_keeps_every_turns_stop_reason():
    """Before, only the last prompt's stop reason was kept: a turn that hit
    the output cap before a later turn ended was lost."""
    session = ACPSession("trial")
    session.record_user_prompt("first")
    session.stop_reason = StopReason.MAX_TOKENS
    session.record_user_prompt("second")
    session.stop_reason = StopReason.END_TURN
    session.record_user_prompt("third")  # never ends: a timeout, an error
    assert session.stop_reasons == ["max_tokens", "end_turn", None]
    assert session.stop_reason is None


def test_a_stop_reason_without_a_recorded_prompt_is_its_own_turn():
    session = ACPSession("trial")
    session.stop_reason = StopReason.END_TURN
    session.stop_reason = StopReason.REFUSAL
    assert session.stop_reasons == ["end_turn", "refusal"]


def test_the_last_call_at_the_output_cap_truncates_the_trial(tmp_path):
    result = _result(
        tmp_path / "t",
        [_proxy_call("tool_calls"), _proxy_call("tool_calls"), _proxy_call("length")],
        stop_reasons=["end_turn"],
    )
    outcome = result["agent_outcome"]
    assert outcome["finish_reasons"] == {"tool_calls": 2, "length": 1}
    assert outcome["stop_reasons"] == ["end_turn"]
    assert (outcome["truncated"], outcome["truncated_by"]) == (True, "last_call")


def test_a_cut_the_agent_recovered_from_does_not_truncate_the_trial(tmp_path):
    result = _result(
        tmp_path / "t",
        [_proxy_call("length"), _proxy_call("stop")],
        stop_reasons=["end_turn"],
    )
    outcome = result["agent_outcome"]
    assert outcome["finish_reasons"] == {"length": 1, "stop": 1}
    assert (outcome["truncated"], outcome["truncated_by"]) == (False, None)


def test_a_last_turn_that_ended_max_tokens_truncates_the_trial(tmp_path):
    """No proxy record (a Claude login): the ACP stop reason says it."""
    result = _result(tmp_path / "t", stop_reasons=["end_turn", "max_tokens"])
    outcome = result["agent_outcome"]
    assert outcome["finish_reasons"] is None
    assert (outcome["truncated"], outcome["truncated_by"]) == (True, "stop_reason")


def test_claude_codes_output_cap_error_truncates_the_trial(tmp_path):
    error = (
        "ACP error -32603: Internal error: API Error: Claude's response exceeded "
        "the 32000 output token maximum. To configure this behavior, set the "
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS environment variable."
    )
    result = _result(tmp_path / "t", stop_reasons=[None], error=error)
    outcome = result["agent_outcome"]
    assert (outcome["truncated"], outcome["truncated_by"]) == (True, "error")


CAP_ERROR = (
    "ACP error -32603: Internal error: API Error: Claude's response exceeded "
    "the 64000 output token maximum. To configure this behavior, set the "
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS environment variable."
)


def test_an_answer_cut_at_the_cap_is_not_retried():
    """Guards the run checks merged at 47a52fdf7: Claude Code ends a turn cut
    at the cap with an ACP error, which the default retry policy ran again
    until an attempt stayed under the cap. A cut is the model's ending under
    the run's cap; other ACP errors are still retried."""
    from benchflow.evaluation import RetryConfig

    retry = RetryConfig()
    assert retry.should_retry(CAP_ERROR) is False
    assert retry.reruns_on_resume(CAP_ERROR) is False
    agy = (
        "ACP error -32603: agy turn ended with ERROR: Your previous response was "
        "cut off because it exceeded the output token limit. Please continue. "
        "Retries remaining: 0"
    )
    assert retry.should_retry(agy) is False
    assert retry.should_retry("ACP error -32603: Internal error") is True


async def test_a_turn_the_agent_ended_at_the_cap_is_scored(tmp_path):
    """Guards the run checks merged at 47a52fdf7: the trial whose turn
    Claude Code ended on the cap was an error with no score, as an agent
    crash is. Like a turn that stopped at max_tokens, its workspace is
    scored, and the error stays on the record."""
    from unittest.mock import AsyncMock

    from benchflow._utils.scoring import classify_score_outcome
    from benchflow.acp.client import ACPError
    from benchflow.rollout import Rollout, RolloutConfig

    task = tmp_path / "t"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n[environment]\n')
    (task / "instruction.md").write_text("Write hello.txt.\n")
    rollout = Rollout(
        RolloutConfig(
            task_path=task, agent="claude-agent-acp", model="claude-haiku-4-5"
        )
    )
    trial = tmp_path / "trial"
    trial.mkdir()
    rollout._rollout_dir = trial
    rollout._rollout_name = "t__cut"
    rollout._started_at = datetime.now()
    rollout.setup = AsyncMock()
    rollout.start = AsyncMock()
    rollout.install_agent = AsyncMock()
    rollout._env = AsyncMock()
    rollout._agent_cwd = "/app"

    async def turn_cut_at_the_cap(_steps):
        raise ACPError(-32603, CAP_ERROR.split(": ", 1)[1])

    async def verify():
        rollout._rewards = {"reward": 1.0}

    async def cleanup():
        rollout._phase = "cleaned"

    rollout._run_steps = turn_cut_at_the_cap
    rollout.verify = verify
    rollout.cleanup = AsyncMock(side_effect=cleanup)

    await rollout.run()

    saved = json.loads((trial / "result.json").read_text())
    assert saved["rewards"] == {"reward": 1.0}
    assert saved["error"] == CAP_ERROR
    assert classify_score_outcome(saved) == "passed"


def _attempts_that(job, attempts, ending):
    """A stand-in for the rollout, attempt by attempt: ``ending(task name,
    attempt)`` gives the attempt's last finish reason and its error."""

    async def run(task_path, _cfg):
        n = attempts[task_path.name] = attempts.get(task_path.name, 0) + 1
        finish, error = ending(task_path.name, n)
        trial = job._jobs_dir / job._job_name / f"{task_path.name}__{n:08d}"
        trial.mkdir(parents=True, exist_ok=True)
        _write_proxy_log(trial, [_proxy_call("tool_calls"), _proxy_call(finish)])
        return _build_rollout_result(
            trial,
            task_name=task_path.name,
            rollout_name=trial.name,
            agent="claude-agent-acp",
            agent_name="claude-agent-acp",
            model=job._config.model,
            n_tool_calls=2,
            prompts=["task"],
            error=error,
            verifier_error=None,
            trajectory=_session_trajectory(),
            partial_trajectory=False,
            stop_reason=None if error else "end_turn",
            stop_reasons=[None] if error else ["end_turn"],
            rewards=None if error else {"reward": 1.0},
            started_at=datetime.now(),
            timing={},
            total_tokens=5000,
        )

    return run


async def test_trials_cut_at_the_cap_run_once_and_count(tmp_path):
    """Guards the run checks merged at 47a52fdf7: with the default retry
    policy, every task whose first attempt Claude Code ended on the cap ran
    again, and the job's truncation rate, read from each task's last
    attempt, said 0 of 10. Now nothing is rerun, and the first five cut
    trials refuse the run."""
    from benchflow.evaluation import RetryConfig

    job = _job(tmp_path, 10, retry=RetryConfig(max_wait_sec=0, min_wait_sec=0))
    attempts: dict[str, int] = {}
    job._run_single_task = _attempts_that(
        job, attempts, lambda name, n: ("length", CAP_ERROR)
    )
    result = await job.run()
    checks = _summary(job)["run_checks"]
    assert set(attempts.values()) == {1}
    assert (checks["truncation"]["known"], checks["truncation"]["truncated"]) == (5, 5)
    assert checks["failed"] is True and result.not_run == 5


async def test_an_attempt_cut_at_the_cap_counts_though_its_task_was_retried(
    tmp_path,
):
    """Guards the run checks merged at 47a52fdf7: an attempt cut at the cap
    that the retry policy ran again for another reason (the agent then went
    idle) was not counted; only its task's last attempt was. A resume of the
    job counts it too."""
    from benchflow.evaluation import Evaluation, RetryConfig

    job = _job(tmp_path, 10, retry=RetryConfig(max_wait_sec=0, min_wait_sec=0))
    attempts: dict[str, int] = {}
    idle = "Agent idle for 600s with no new tool call"
    job._run_single_task = _attempts_that(
        job,
        attempts,
        lambda name, n: (
            ("length", idle)
            if name in {"task-002", "task-005"} and n == 1
            else ("stop", None)
        ),
    )
    result = await job.run()
    truncation = _summary(job)["run_checks"]["truncation"]
    assert result.total == 10 and attempts["task-002"] == attempts["task-005"] == 2
    assert (truncation["known"], truncation["truncated"], truncation["retried"]) == (
        12,
        2,
        2,
    )
    assert truncation["truncated_trials"] == ["task-002", "task-005"]
    assert _summary(job)["run_checks"]["warnings"] == [
        "truncation: 2 of 12 attempts with a known ending (2 of them retried) were "
        "cut at the output cap (16.7%), at or above 10%"
    ]

    again = Evaluation(
        tasks_dir=job._tasks_dir,
        jobs_dir=job._jobs_dir,
        config=job._config,
        job_name=job._job_name,
        preflight=False,
    )
    await again.run()
    resumed = _summary(again)["run_checks"]["truncation"]
    assert (resumed["known"], resumed["truncated"], resumed["retried"]) == (12, 2, 2)


def test_a_trial_with_no_known_ending_is_unknown_not_untruncated(tmp_path):
    result = _result(tmp_path / "t", stop_reasons=[None], error="Agent timed out")
    assert result["agent_outcome"]["truncated"] is None
    assert result["agent_outcome"]["stop_reasons"] == [None]


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


async def test_every_turn_of_every_execute_reaches_the_result(tmp_path, monkeypatch):
    """Two prompts in one execute(), then one that times out: result.json
    lists all three turns, the last one unended."""
    rollout = _rollout(tmp_path)
    endings = iter([StopReason.MAX_TOKENS, StopReason.END_TURN])

    async def prompts_end(_client, session, prompts, *_args, **_kwargs):
        for prompt in prompts:
            session.record_user_prompt(prompt)
            if prompt == "third":
                raise _agent_prompt_timeout_error(session, 60)
            session.stop_reason = next(endings)
            session.mark_prompt_end()
        return _capture_session_trajectory(session), 0

    monkeypatch.setattr(rollout._planes, "execute_prompts", prompts_end)
    await rollout.execute(prompts=["first", "second"])
    with pytest.raises(AgentPromptTimeoutError):
        await rollout.execute(prompts=["third"])
    rollout._error = "Agent timed out after 60s"
    rollout._build_result()
    outcome = json.loads((rollout._rollout_dir / "result.json").read_text())[
        "agent_outcome"
    ]
    assert outcome["stop_reasons"] == ["max_tokens", "end_turn", None]
    # The last turn did not end and no call was recorded: unknown.
    assert outcome["truncated"] is None


# -- the job's truncation rate ---------------------------------------------


def _tasks(tmp_path: Path, n: int) -> Path:
    tasks_dir = tmp_path / "tasks"
    for i in range(n):
        task = tasks_dir / f"task-{i:03d}"
        task.mkdir(parents=True)
        (task / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
        (task / "instruction.md").write_text("Write 42 to /app/answer.txt.\n")
    return tasks_dir


def _job(tmp_path: Path, n: int, *, concurrency: int = 1, **config):
    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig

    base = {
        "agent": "claude-agent-acp",
        "model": "anthropic/claude-sonnet-5",
        "concurrency": concurrency,
        "retry": RetryConfig(max_retries=0),
    }
    base.update(config)
    return Evaluation(
        tasks_dir=_tasks(tmp_path, n),
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(**base),
        preflight=False,
    )


def _trials_that(job, calls_for, endings=lambda name: ["end_turn"]):
    """A stand-in for the rollout: each trial folder gets the proxy's
    records ``calls_for(task name)`` gives, its model calls written as
    teardown writes them, and the result.json the rollout builds, with
    the prompt turns' stop reasons ``endings(task name)`` gives."""

    async def run(task_path, _cfg):
        trial = job._jobs_dir / job._job_name / f"{task_path.name}__abcd1234"
        calls = calls_for(task_path.name)
        turns = endings(task_path.name)
        trial.mkdir(parents=True, exist_ok=True)
        if calls:
            _write_proxy_log(trial, calls)
        return _build_rollout_result(
            trial,
            task_name=task_path.name,
            rollout_name=trial.name,
            agent="claude-agent-acp",
            agent_name="claude-agent-acp",
            model=job._config.model,
            n_tool_calls=2,
            prompts=["task"],
            error=None,
            verifier_error=None,
            trajectory=_session_trajectory(),
            partial_trajectory=False,
            stop_reason=turns[-1] if turns else None,
            stop_reasons=turns,
            rewards={"reward": 0.0},
            started_at=datetime.now(),
            timing={},
            total_tokens=5000,
        )

    return run


def _summary(job) -> dict:
    return json.loads((job._jobs_dir / job._job_name / "summary.json").read_text())


def _exit_code(result, job_dir) -> int:
    """What `bench run` exits with for *result* (cli.main._finish_run)."""
    from types import SimpleNamespace

    import typer

    from benchflow.cli.main import _finish_run

    request = SimpleNamespace(fail_under=None, fail_on=None, summary_out=None)
    try:
        _finish_run(request, result, job_dir)
    except typer.Exit as stop:
        return stop.exit_code
    return 0


async def test_half_the_trials_cut_at_the_cap_stops_the_job(tmp_path):
    job = _job(tmp_path, 10)
    job._run_single_task = _trials_that(
        job, lambda name: [_proxy_call("tool_calls"), _proxy_call("length")]
    )
    result = await job.run()
    checks = _summary(job)["run_checks"]
    # min(5, 10 planned) trials with a known ending, all cut: no sixth starts.
    assert result.total == 5
    assert checks["truncation"]["level"] == "refuse"
    assert (checks["truncation"]["known"], checks["truncation"]["truncated"]) == (5, 5)
    assert checks["stopped"] is True
    assert checks["failed"] is True
    assert (
        "5 of 5 trials with a known ending were cut at the output cap"
        in (checks["reason"])
    )
    assert len(checks["not_started"]) == 5
    assert result.not_run == 5
    assert result.run_checks == checks
    assert _exit_code(result, job._jobs_dir / job._job_name) == 1
    card = json.loads((job._jobs_dir / job._job_name / "run-card.json").read_text())
    assert card["after_run"]["truncation"]["rate"] == 1.0


async def test_a_rate_over_the_warning_warns_and_runs_every_trial(tmp_path, caplog):
    job = _job(tmp_path, 10, concurrency=2)
    cut = {"task-003", "task-007"}
    job._run_single_task = _trials_that(
        job,
        lambda name: (
            [_proxy_call("tool_calls"), _proxy_call("length")]
            if name in cut
            else [_proxy_call("tool_calls"), _proxy_call("stop")]
        ),
    )
    result = await job.run()
    checks = _summary(job)["run_checks"]
    assert result.total == 10
    assert checks["truncation"]["rate"] == 0.2
    assert checks["truncation"]["level"] == "warn"
    assert checks["truncation"]["truncated_trials"] == sorted(cut)
    assert (checks["failed"], checks["stopped"]) == (False, False)
    assert checks["warnings"] == [
        "truncation: 2 of 10 trials with a known ending were cut at the output "
        "cap (20.0%), at or above 10%"
    ]
    assert _exit_code(result, job._jobs_dir / job._job_name) == 0
    assert "Truncated: 2 of 10 trials with a known ending" in caplog.text


async def test_a_job_of_fewer_than_five_trials_is_too_few_to_refuse(tmp_path):
    """Guards the run checks merged at 47a52fdf7: a refusal waited for
    min(5, planned) trials, so a job of one or two trials was refused on
    them; `bench run` of a single task whose one trial was cut exited 1, and
    so did every repeat of `--trials N` on one task. A refusal now waits for
    five attempts with a known ending; a run of several jobs counts them
    together (test_trials_n_on_one_task_is_not_stopped_by_one_cut_trial)."""
    job = _job(tmp_path, 2)
    job._run_single_task = _trials_that(job, lambda name: [_proxy_call("length")])
    result = await job.run()
    checks = _summary(job)["run_checks"]
    assert result.total == 2
    assert checks["truncation"]["min_trials"] == 5
    assert (checks["truncation"]["level"], checks["truncation"]["rate"]) == (
        "too_few",
        1.0,
    )
    assert (checks["failed"], checks["stopped"]) == (False, False)
    # Too few to refuse the run on, not too few to say so.
    assert checks["warnings"] == [
        "truncation: 2 of 2 trials with a known ending were cut at the output cap "
        "(100.0%), at or above 10%; too few to judge (a refusal waits for 5)"
    ]
    assert _exit_code(result, job._jobs_dir / job._job_name) == 0


async def test_trials_with_no_known_ending_are_not_in_the_rate(tmp_path):
    """Two trials cut at the cap, four that recorded no call and no ending:
    the rate is over the two, which are too few to refuse the run."""
    job = _job(tmp_path, 6)
    cut = {"task-000", "task-001"}
    job._run_single_task = _trials_that(
        job,
        lambda name: [_proxy_call("length")] if name in cut else [],
        endings=lambda name: ["end_turn"] if name in cut else [None],
    )
    result = await job.run()
    checks = _summary(job)["run_checks"]
    truncation = checks["truncation"]
    assert (truncation["trials"], truncation["known"], truncation["truncated"]) == (
        6,
        2,
        2,
    )
    assert (truncation["rate"], truncation["level"]) == (1.0, "too_few")
    assert (checks["failed"], checks["stopped"], result.total) == (False, False, 6)


async def test_a_job_the_truncation_rate_stopped_stays_failed(tmp_path):
    """Guards the run checks merged at 47a52fdf7: at --concurrency 8 the
    three trials cut at the cap end first, so the first five to finish are 3
    of 5 cut (60%) and the job stops starting trials. The five still running
    end uncut, so the job's own rate is under half, and summary.json said
    the run passed its checks: `bench run` exited 0 with 9 trials never run."""
    import asyncio

    cut = {"task-000", "task-001", "task-002"}
    job = _job(tmp_path, 20, concurrency=8)
    inner = _trials_that(
        job,
        lambda name: [
            _proxy_call("tool_calls"),
            _proxy_call("length" if name in cut else "stop"),
        ],
    )
    delay = {"task-000": 0, "task-001": 0, "task-002": 0}
    delay |= {"task-003": 0.05, "task-004": 0.05}

    async def run(task_path, cfg):
        await asyncio.sleep(delay.get(task_path.name, 0.3))
        return await inner(task_path, cfg)

    job._run_single_task = run
    result = await job.run()
    checks = _summary(job)["run_checks"]
    assert checks["stopped"] is True and len(checks["not_started"]) == 9
    assert checks["truncation"]["refused_at"] == {"known": 5, "truncated": 3}
    assert (checks["truncation"]["known"], checks["truncation"]["level"]) == (
        11,
        "warn",
    )
    assert checks["failed"] is True
    assert checks["failures"] == [
        "truncation: 3 of 5 trials with a known ending were cut at the output cap "
        "(60.0%) when the job stopped, at or above 50%; 3 of 11 (27.3%) by its end"
    ]
    assert checks["warnings"] == []
    assert _exit_code(result, job._jobs_dir / job._job_name) == 1


async def test_a_resumed_job_that_was_refused_starts_nothing_new(tmp_path):
    """The trials the first run finished count first: the same setting cuts
    them, so the resume starts none of the five left."""
    from benchflow.evaluation import Evaluation

    job = _job(tmp_path, 10)
    job._run_single_task = _trials_that(job, lambda name: [_proxy_call("length")])
    await job.run()
    again = Evaluation(
        tasks_dir=job._tasks_dir,
        jobs_dir=job._jobs_dir,
        config=job._config,
        job_name=job._job_name,
        preflight=False,
    )
    started: list[str] = []

    async def run(task_path, cfg):
        started.append(task_path.name)
        raise AssertionError("no trial should start")

    again._run_single_task = run
    result = await again.run()
    checks = _summary(again)["run_checks"]
    assert started == []
    assert checks["stopped"] is True and len(checks["not_started"]) == 5
    assert (result.total, result.not_run) == (5, 5)


def test_a_failed_run_check_in_any_trial_job_fails_the_matrix(tmp_path):
    from types import SimpleNamespace

    from benchflow.cli.eval_artifacts import _matrix_result, _TrialJob

    def trial_job(trial: int, failures: list[str]) -> _TrialJob:
        result = SimpleNamespace(
            total=5,
            passed=0,
            failed=5,
            errored=0,
            verifier_errored=0,
            not_run=0,
            results={},
            run_checks={"failed": bool(failures), "failures": failures},
        )
        return _TrialJob("claude", trial, result, "claude-agent-acp", None)

    jobs = [trial_job(1, ["truncation: 5 of 5 trials ..."]), trial_job(2, [])]
    whole = _matrix_result(tmp_path, jobs, [])
    assert (whole.run_checks["failed"], whole.run_checks["failures"]) == (
        True,
        ["claude/trial-01: truncation: 5 of 5 trials ..."],
    )
    assert _exit_code(whole, tmp_path) == 1


async def _one_task_trial(tmp_path: Path, cut: bool):
    """One trial job of `bench run <task> --trials N`: a job of one task."""
    job = _job(tmp_path, 1)
    job._run_single_task = _trials_that(
        job,
        lambda name: [
            _proxy_call("tool_calls"),
            _proxy_call("length" if cut else "stop"),
        ],
    )
    return job, await job.run()


async def test_trials_n_on_one_task_is_not_stopped_by_one_cut_trial(tmp_path):
    """Guards the run checks merged at 47a52fdf7: with `--trials 10` on one
    task each trial is a job of one trial, which refused itself when that
    trial was cut, and the loop stopped at the first failed check: trials 2
    to 10 never ran."""
    from benchflow.cli.eval_artifacts import _stop_after_run_check

    job, result = await _one_task_trial(tmp_path, cut=True)
    checks = _summary(job)["run_checks"]
    assert checks["truncation"]["known"] == 1
    assert checks["failed"] is False
    assert _stop_after_run_check("claude", 1, 10, result) is False


async def test_trials_n_is_refused_on_the_repeats_counted_together(tmp_path):
    """Guards the run checks merged at 47a52fdf7: the repeats of `--trials`
    are judged together. Four cut repeats of one task are too few to judge;
    the fifth refuses the run, and the later repeats do not run. Repeats that
    are not cut never refuse it."""
    from benchflow.cli.eval_artifacts import _stop_after_run_check

    cut = [
        (await _one_task_trial(tmp_path / f"cut-{n}", cut=True))[1] for n in range(5)
    ]
    earlier = list(enumerate(cut[:3], 1))
    assert _stop_after_run_check("claude", 4, 10, cut[3], earlier=earlier) is False
    earlier = list(enumerate(cut[:4], 1))
    assert _stop_after_run_check("claude", 5, 10, cut[4], earlier=earlier) is True
    clean = [
        (await _one_task_trial(tmp_path / f"clean-{n}", cut=n == 0))[1]
        for n in range(6)
    ]
    earlier = list(enumerate(clean[:5], 1))
    assert _stop_after_run_check("claude", 6, 10, clean[5], earlier=earlier) is False


async def test_the_matrix_fails_on_the_repeats_truncation_rate(tmp_path):
    """Guards the run checks merged at 47a52fdf7: five repeats of one task,
    each a job too small to judge itself, all cut: the matrix's own check
    over them fails the run."""
    from benchflow.cli.eval_artifacts import _matrix_result, _TrialJob

    jobs = [
        _TrialJob(
            "claude",
            n,
            (await _one_task_trial(tmp_path / f"t{n}", cut=True))[1],
            "claude-agent-acp",
            None,
        )
        for n in range(1, 6)
    ]
    assert not any(job.result.run_checks["failed"] for job in jobs)
    whole = _matrix_result(tmp_path, jobs, [])
    assert whole.run_checks["failures"] == [
        "claude: truncation: 5 of 5 trials with a known ending were cut at the "
        "output cap (100.0%), at or above 50%"
    ]
    assert whole.run_checks["aliases"]["claude"]["truncation"]["known"] == 5
    assert _exit_code(whole, tmp_path) == 1


# -- what the calls carried, against what the run asked for ----------------


@pytest.mark.parametrize(
    "requested,served",
    [
        ("anthropic/claude-sonnet-5", "claude-sonnet-5-20260301"),
        ("claude-sonnet-5", "claude-sonnet-5"),
        ("openai/gpt-5.5", "gpt-5.5-2026-08-01"),
        ("aws-bedrock/us.anthropic.claude-opus-4-8", "anthropic.claude-opus-4-8-v1:0"),
        ("vertex_ai/claude-sonnet-4-5@20250929", "claude-sonnet-4-5"),
        ("mistral/mistral-large-latest", "mistral-large-2511"),
        ("moonshot/kimi-k3", "kimi-k3-0905-preview"),
        (
            "fireworks_ai/accounts/fireworks/models/kimi-k3",
            "accounts/fireworks/models/kimi-k3",
        ),
        ("claude-opus-4-6[1m]", "claude-opus-4-6"),
        ("gemini/gemini-3-pro-preview", "models/gemini-3-pro-preview"),
    ],
)
def test_a_dated_or_routed_id_of_the_requested_model_is_the_same(requested, served):
    from benchflow.run_checks import same_model

    assert same_model(requested, served)


@pytest.mark.parametrize(
    "requested,served",
    [
        ("anthropic/claude-sonnet-5", "claude-haiku-4-5-20251001"),
        ("openai/gpt-5.5", "gpt-5.5-mini"),
        ("deepseek/deepseek-v4-pro", "deepseek-v4-flash"),
        ("claude-opus-5-5", "claude-opus-5"),
    ],
)
def test_another_model_is_not_the_requested_one(requested, served):
    from benchflow.run_checks import same_model

    assert not same_model(requested, served)


def _side_call() -> dict:
    """Claude Code's background call to its small model, through the proxy."""
    return _proxy_call(
        "stop",
        max_tokens=512,
        model="claude-haiku-4-5",
        served="claude-haiku-4-5-20251001",
        route="anthropic/claude-haiku-4-5",
        effort=None,
    )


def test_settings_sent_counts_main_calls_and_keeps_side_calls_apart(tmp_path):
    reviewer = _proxy_call("stop", max_tokens=4096, effort=None)
    reviewer["metadata"]["call_purpose"] = "reviewer"
    result = _result(
        tmp_path / "t",
        [_proxy_call(), _proxy_call(), _side_call(), reviewer],
        stop_reasons=["end_turn"],
    )
    sent = result["settings_sent"]
    assert (sent["source"], sent["main_calls"], sent["side_calls"]) == ("proxy", 2, 1)
    assert sent["max_output_tokens"] == {"128000": 2}
    assert sent["effort"] == {"high": 2}
    assert sent["models_served"] == {"claude-sonnet-5-20260301": 2}
    assert sent["routes"] == {"anthropic/claude-sonnet-5": 2}
    assert sent["side_models"] == {"claude-haiku-4-5": 1}
    assert sent["without"]["temperature"] == 2


def _calls(**fields):
    return lambda name: [
        _proxy_call("tool_calls", **fields),
        _proxy_call("stop", **fields),
        _side_call(),
    ]


async def test_a_cap_the_calls_did_not_carry_fails_after_the_first_trial(tmp_path):
    job = _job(tmp_path, 5, agent_env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000"})
    job._run_single_task = _trials_that(job, _calls(max_tokens=32000))
    result = await job.run()
    checks = _summary(job)["run_checks"]
    assert result.total == 1 and result.not_run == 4
    [mismatch] = checks["settings"]["mismatches"]
    assert mismatch == {
        "setting": "max_output_tokens",
        "requested": 128000,
        "source": "agent_env",
        "kind": "mismatch",
        "fails": True,
        "sent": {"32000": 2},
        "calls": 2,
        "of": 2,
        "trials": 1,
        "first_trial": "task-000__abcd1234",
    }
    assert checks["failed"] is True and checks["stopped"] is True
    assert checks["failures"] == [
        "max_output_tokens: requested 128000 (from --agent-env); the main calls "
        "carried 32000 (2): 2 of 2 calls differ, in 1 trial(s), first "
        "task-000__abcd1234"
    ]
    assert _exit_code(result, job._jobs_dir / job._job_name) == 1


async def test_benchflow_s_default_cap_lowered_by_the_agent_only_warns(tmp_path):
    """Claude Code lowers 128000 to the model's own cap: BenchFlow asked
    for it, the run did not."""
    job = _job(tmp_path, 3)
    job._run_single_task = _trials_that(job, _calls(max_tokens=64000))
    result = await job.run()
    checks = _summary(job)["run_checks"]
    assert result.total == 3
    assert checks["failed"] is False
    assert checks["warnings"] == [
        "max_output_tokens: BenchFlow's default 128000; the main calls carried "
        "64000 (6), in 3 trial(s): the agent lowers it to the model's own cap; set "
        "the cap you mean to measure with --agent-env"
    ]


async def test_a_smaller_cap_on_one_call_is_not_a_mismatch(tmp_path):
    """The largest cap the calls carried is the cap applied: a compaction's
    summary may ask for less."""
    job = _job(tmp_path, 2, agent_env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000"})
    job._run_single_task = _trials_that(
        job,
        lambda name: [_proxy_call("tool_calls"), _proxy_call("stop", max_tokens=20000)],
    )
    await job.run()
    checks = _summary(job)["run_checks"]
    assert checks["settings"]["mismatches"] == []
    assert checks["failed"] is False


async def test_an_effort_the_calls_replaced_fails_and_one_never_sent_warns(tmp_path):
    job = _job(tmp_path, 3, reasoning_effort="max")
    job._run_single_task = _trials_that(job, _calls(effort="high"))
    result = await job.run()
    checks = _summary(job)["run_checks"]
    assert result.total == 1
    assert checks["failures"][0].startswith(
        "effort: requested max; the main calls carried high (2): 2 of 2 calls differ"
    )
    quiet = _job(tmp_path / "quiet", 3, reasoning_effort="max")
    quiet._run_single_task = _trials_that(quiet, _calls(effort=None))
    result = await quiet.run()
    checks = _summary(quiet)["run_checks"]
    assert result.total == 3 and checks["failed"] is False
    assert checks["warnings"] == [
        "effort: requested max; no main call carried it (6 calls in 3 trial(s)), "
        "so the provider's default applied"
    ]


def _efforts(odd: dict[str, int], *, calls: int = 50, other: str = "high"):
    """Each trial's main calls carry max but ``odd[task name]`` of them."""

    def rows(name):
        n = odd.get(name, 0)
        return [
            _proxy_call("tool_calls", effort=other if i < n else "max")
            for i in range(calls - 1)
        ] + [_proxy_call("stop", effort="max")]

    return rows


async def test_one_call_in_250_with_another_effort_is_within_the_tolerance(
    tmp_path,
):
    """Guards the run checks merged at 47a52fdf7: the 1% of the calls that
    may carry another value was judged per trial, so one call of 50 (2%) in
    a trial stopped the job and failed the run. Over the job's calls it is 1
    of 250 (0.4%)."""
    job = _job(tmp_path, 5, reasoning_effort="max")
    job._run_single_task = _trials_that(job, _efforts({"task-000": 1}))
    result = await job.run()
    checks = _summary(job)["run_checks"]
    assert (result.total, checks["failed"], checks["warnings"]) == (5, False, [])
    [effort] = [c for c in checks["settings"]["compared"] if c["setting"] == "effort"]
    assert (effort["differing"], effort["trials"], effort["clear_trials"]) == (1, 1, 0)


async def test_odd_calls_above_the_job_s_tolerance_fail_without_a_stop(tmp_path):
    """Guards the run checks merged at 47a52fdf7: two calls of 50 in each
    trial carry another effort, 4% of the job's calls. No trial shows a clear
    mismatch, so every trial runs, and the run fails at the end."""
    names = {f"task-{i:03d}": 2 for i in range(5)}
    job = _job(tmp_path, 5, reasoning_effort="max")
    job._run_single_task = _trials_that(job, _efforts(names))
    result = await job.run()
    checks = _summary(job)["run_checks"]
    assert (result.total, checks["stopped"], checks["failed"]) == (5, False, True)
    assert checks["failures"] == [
        "effort: requested max; the main calls carried max (240), high (10): 10 of "
        "250 calls differ, in 5 trial(s), first task-000__abcd1234"
    ]


async def test_a_trial_whose_calls_mostly_carried_another_effort_stops_the_job(
    tmp_path,
):
    """With the tolerance judged over the job's calls, a trial whose calls
    carried another effort in 30 of 50 is still evidence enough that the
    setting was not applied: the job stops after it, as when every call
    differs."""
    job = _job(tmp_path, 5, reasoning_effort="max")
    job._run_single_task = _trials_that(job, _efforts({"task-000": 30}))
    result = await job.run()
    checks = _summary(job)["run_checks"]
    assert (result.total, result.not_run, checks["failed"]) == (1, 4, True)
    assert checks["stopped_by"] == "effort"


async def test_a_served_model_other_than_the_requested_one_fails(tmp_path):
    job = _job(tmp_path, 3)
    job._run_single_task = _trials_that(job, _calls(served="claude-haiku-4-5-20251001"))
    result = await job.run()
    checks = _summary(job)["run_checks"]
    assert result.total == 1
    assert checks["failures"][0].startswith(
        "model: requested anthropic/claude-sonnet-5; the main calls carried "
        "claude-haiku-4-5-20251001 (2): 2 of 2 calls differ"
    )


async def test_an_azure_deployment_named_otherwise_than_its_model_passes(tmp_path):
    """Guards the run checks merged at 47a52fdf7: Azure routes by deployment
    name and answers with the model's own id (the request names
    azure/prod-gpt5, the response's ``model`` is gpt-5-2025-08-07), so every
    run on a deployment not named after its model failed after one trial.
    The served model is recorded, with a warning."""
    model = "azure/prod-gpt5"
    job = _job(tmp_path, 3, agent="codex-acp", model=model)
    call = {
        "model": "benchflow-azure-prod-gpt5",
        "served": "gpt-5-2025-08-07",
        "route": model,
        "effort": None,
        "max_tokens": None,
    }
    job._run_single_task = _trials_that(
        job,
        lambda name: [_proxy_call("tool_calls", **call), _proxy_call("stop", **call)],
    )
    result = await job.run()
    checks = _summary(job)["run_checks"]
    assert (result.total, checks["failed"]) == (3, False)
    assert checks["warnings"] == [
        "model: requested azure/prod-gpt5, a deployment's or an alias's name; the "
        "calls were served as gpt-5-2025-08-07 (6), recorded, not compared"
    ]


@pytest.mark.parametrize("recorded", ["high", "max"])
async def test_an_effort_the_route_sets_is_the_one_that_ran(tmp_path, recorded):
    """Guards the run checks merged at 47a52fdf7: on Bedrock, Claude 4.8's
    route sets high on each request (LiteLLM takes no more for it), while
    the run asked for max. With the calls recorded at high the run failed
    after its first trial; recorded at the agent's max, it passed as a max
    run. The route's effort is the one that ran: a warning names it."""
    job = _job(
        tmp_path,
        3,
        model="aws-bedrock/us.anthropic.claude-opus-4-8",
        reasoning_effort="max",
    )

    async def run(task_path, _cfg):
        trial = job._jobs_dir / job._job_name / f"{task_path.name}__abcd1234"
        trial.mkdir(parents=True, exist_ok=True)
        route = {
            "model": "benchflow-aws-bedrock-us.anthropic.claude-opus-4-8",
            "served": "claude-opus-4-8",
            "route": "bedrock/us.anthropic.claude-opus-4-8",
            "effort": recorded,
        }
        _write_proxy_log(
            trial, [_proxy_call("tool_calls", **route), _proxy_call("stop", **route)]
        )
        return _build_rollout_result(
            trial,
            task_name=task_path.name,
            rollout_name=trial.name,
            agent="claude-agent-acp",
            agent_name="claude-agent-acp",
            model=job._config.model,
            n_tool_calls=2,
            prompts=["task"],
            error=None,
            verifier_error=None,
            trajectory=_session_trajectory(),
            partial_trajectory=False,
            stop_reason="end_turn",
            stop_reasons=["end_turn"],
            rewards={"reward": 1.0},
            started_at=datetime.now(),
            timing={},
            total_tokens=5000,
            reasoning_effort="max",
            effort_applied="max",
            effort_route="high",
        )

    job._run_single_task = run
    result = await job.run()
    checks = _summary(job)["run_checks"]
    assert (result.total, checks["failed"]) == (3, False)
    assert checks["warnings"] == [
        "effort: requested max; the model proxy's route set it to high on each "
        "request, in 3 trial(s)"
    ]


async def test_matching_calls_pass_and_the_card_records_route_and_served_id(
    tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger="benchflow.evaluation")
    job = _job(
        tmp_path,
        3,
        reasoning_effort="high",
        agent_env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000"},
    )
    job._run_single_task = _trials_that(job, _calls())
    result = await job.run()
    checks = _summary(job)["run_checks"]
    assert (result.total, checks["failed"], checks["warnings"]) == (3, False, [])
    settings = checks["settings"]
    assert [w["setting"] for w in settings["requested"]] == [
        "model",
        "effort",
        "max_output_tokens",
    ]
    assert settings["models_served"] == {"claude-sonnet-5-20260301": 6}
    assert settings["routes"] == {"anthropic/claude-sonnet-5": 6}
    assert (settings["main_calls"], settings["side_calls"]) == (6, 3)
    assert _exit_code(result, job._jobs_dir / job._job_name) == 0
    card = json.loads((job._jobs_dir / job._job_name / "run-card.json").read_text())
    assert card["after_run"]["settings"]["routes"] == {"anthropic/claude-sonnet-5": 6}
    assert (
        "Calls:     6 to the run's model, served as claude-sonnet-5-20260301 (6), "
        "by route anthropic/claude-sonnet-5 (6); 3 side calls to "
        "claude-haiku-4-5 (3), recorded, not compared"
    ) in caplog.text
    assert (
        "Sent:      max_output_tokens: requested 128000 (from --agent-env); the "
        "main calls carried it"
    ) in caplog.text
    assert "Recorded:  temperature none sent; top_p none sent (not requested)" in (
        caplog.text
    )


FIXTURES = Path(__file__).parent / "fixtures" / "model_calls"


@pytest.mark.parametrize(
    "fixture,mismatch",
    [("hello-proxy.jsonl", None), ("trunc-proxy.jsonl", {"400": 30})],
)
def test_recorded_proxy_calls_are_compared_with_the_requested_cap(
    tmp_path, fixture, mismatch
):
    """Claude Code on MiMo through the proxy, as recorded: a run that asked
    for 128000 and one whose calls carried a 400-token cap."""
    from benchflow.run_checks import compare_sent, settings_sent

    trial = tmp_path / "trial"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "trajectory" / "llm_trajectory.jsonl").write_bytes(
        (FIXTURES / fixture).read_bytes()
    )
    write_model_calls(trial)
    from benchflow.run_checks import read_calls

    sent = settings_sent(read_calls(trial), "xiaomi/mimo-v2.6-flash")
    assert sent is not None
    assert sent["side_calls"] == 0
    assert sent["models_served"] == {"mimo-v2.6-flash": sent["main_calls"]}
    found = compare_sent(
        {
            "setting": "max_output_tokens",
            "value": 128000,
            "source": "agent_env",
            "fails": True,
        },
        sent,
    )
    if mismatch is None:
        assert found is None
    else:
        assert found == {"kind": "mismatch", "sent": mismatch, "calls": 30, "of": 30}
    assert (
        compare_sent(
            {
                "setting": "model",
                "value": "xiaomi/mimo-v2.6-flash",
                "source": "model",
                "fails": True,
            },
            sent,
        )
        is None
    )


def _session_log_trial(trial: Path) -> None:
    """A trial whose calls went around the proxy (a Claude login): Claude
    Code's own session log, as the rollout copies it, with one message
    Claude Code wrote itself (an API error it reports, ``<synthetic>``)."""
    logs = trial / "agent" / "claude-sessions" / "-app"
    logs.mkdir(parents=True)
    rows = [
        {"type": "user", "timestamp": "2026-11-05T10:00:00Z", "message": {}},
        {
            "type": "assistant",
            "timestamp": "2026-11-05T10:00:05Z",
            "message": {
                "id": "msg_1",
                "model": "claude-sonnet-5-20260301",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-11-05T10:00:06Z",
            "message": {
                "id": "msg_2",
                "model": "<synthetic>",
                "stop_reason": "stop_sequence",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    ]
    (logs / "s-1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_model_calls(trial)


async def test_a_session_log_s_own_messages_and_unrecorded_settings(tmp_path, caplog):
    """Claude Code's ``<synthetic>`` message is no call to another model, and
    an effort its session log does not record is not compared."""
    caplog.set_level(logging.INFO, logger="benchflow.evaluation")
    job = _job(tmp_path, 2, reasoning_effort="high")

    async def run(task_path, _cfg):
        trial = job._jobs_dir / job._job_name / f"{task_path.name}__abcd1234"
        _session_log_trial(trial)
        return _build_rollout_result(
            trial,
            task_name=task_path.name,
            rollout_name=trial.name,
            agent="claude-agent-acp",
            agent_name="claude-agent-acp",
            model=job._config.model,
            n_tool_calls=0,
            prompts=["task"],
            error=None,
            verifier_error=None,
            trajectory=_session_trajectory(),
            partial_trajectory=False,
            stop_reasons=["end_turn"],
            rewards={"reward": 1.0},
            started_at=datetime.now(),
            timing={},
        )

    job._run_single_task = run
    result = await job.run()
    checks = _summary(job)["run_checks"]
    assert (result.total, checks["failed"], checks["warnings"]) == (2, False, [])
    assert checks["settings"]["models_served"] == {"claude-sonnet-5-20260301": 2}
    assert checks["settings"]["sources"] == ["claude-code-session-log"]
    assert (
        "Sent:      effort: requested high; not recorded in the calls, so not compared"
    ) in caplog.text
    assert "Recorded:  temperature not recorded; top_p not recorded" in caplog.text


async def test_a_control_run_says_nothing_after_the_run(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="benchflow.evaluation")
    job = _job(tmp_path, 2, agent="oracle", model=None)
    job._run_single_task = _trials_that(job, lambda name: [], endings=lambda name: [])
    await job.run()
    assert "Run card, after the run" not in caplog.text
    card = json.loads((job._jobs_dir / job._job_name / "run-card.json").read_text())
    assert card["after_run"]["truncation"]["known"] == 0


async def test_a_refused_run_names_what_to_do_next(tmp_path, capsys):
    """Guards the run checks merged at 47a52fdf7: a refusal named no next
    step, and the stopped line said to fix the setting and resume the job,
    which never runs them: a resume with the same setting is refused again,
    and one with another is a changed run. Each check that fails the run
    names its own next step."""
    from benchflow.cli._shared import _report_eval_result
    from benchflow.run_checks import NEXT_STEPS

    job = _job(tmp_path, 10)
    job._run_single_task = _trials_that(
        job, lambda name: [_proxy_call("tool_calls"), _proxy_call("length")]
    )
    result = await job.run()
    assert result.run_checks["next_steps"] == [NEXT_STEPS["truncation"]]
    job_dir = job._jobs_dir / job._job_name
    _report_eval_result(result, job_dir)
    out = " ".join(capsys.readouterr().out.split())
    assert "5 trial(s) not started (not counted as failures)" in out
    assert "resume the job" not in out
    assert _exit_code(result, job_dir) == 1
    err = capsys.readouterr().err
    assert "Run check failed: truncation: 5 of 5 trials" in err
    assert f"next: {NEXT_STEPS['truncation']}" in err

    cap = _job(
        tmp_path / "cap", 3, agent_env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000"}
    )
    cap._run_single_task = _trials_that(cap, _calls(max_tokens=32000))
    result = await cap.run()
    assert result.run_checks["next_steps"] == [NEXT_STEPS["max_output_tokens"]]
    warned = _job(tmp_path / "warned", 10, concurrency=2)
    warned._run_single_task = _trials_that(
        warned,
        lambda name: [
            _proxy_call("tool_calls"),
            _proxy_call("length" if name == "task-003" else "stop"),
        ],
    )
    result = await warned.run()
    # A warning only: nothing to do before the run counts.
    assert result.run_checks["warnings"] and result.run_checks["next_steps"] == []


def test_bench_eval_resume_exits_1_on_a_failed_run_check(capsys):
    """`bench eval resume` exits as `bench run` does: a run check that
    failed is an exit 1, not a finished run."""
    from types import SimpleNamespace

    import typer

    from benchflow.cli._shared import _exit_if_evaluation_had_errors

    failed = SimpleNamespace(
        errored=0,
        verifier_errored=0,
        run_checks={"failed": True, "failures": ["truncation: 3 of 5 ..."]},
    )
    with pytest.raises(typer.Exit) as stop:
        _exit_if_evaluation_had_errors(failed)
    assert stop.value.exit_code == 1
    assert "Run check failed: truncation: 3 of 5 ..." in capsys.readouterr().err
    _exit_if_evaluation_had_errors(
        SimpleNamespace(errored=0, verifier_errored=0, run_checks={"failures": []})
    )


async def test_a_branch_child_s_turns_are_its_own_and_the_parent_s_return(
    tmp_path, monkeypatch
):
    """An in-place branch child drives the parent's rollout: its turns are
    its result's, and the fork's restore puts the parent's back."""
    from benchflow.branch_result import scope_child_result_state
    from benchflow.rollout_branch import _LinearState

    rollout = _rollout(tmp_path)

    async def prompts_end(_client, session, prompts, *_args, **_kwargs):
        for prompt in prompts:
            session.record_user_prompt(prompt)
            session.stop_reason = (
                StopReason.MAX_TOKENS if prompt == "child" else StopReason.END_TURN
            )
            session.mark_prompt_end()
        return _capture_session_trajectory(session), 0

    monkeypatch.setattr(rollout._planes, "execute_prompts", prompts_end)
    await rollout.execute(prompts=["parent"])
    saved = _LinearState.capture(rollout)
    scope_child_result_state(rollout)
    await rollout.execute(prompts=["child"])
    assert rollout._turn_stop_reasons == ["max_tokens"]
    saved.restore_onto(rollout)
    assert rollout._turn_stop_reasons == ["end_turn"]


def _turn(finish: str, effort: str = "max") -> dict:
    """A turn of the agent's loop: it offers the agent its tools."""
    call = _proxy_call(finish, effort=effort)
    call["request"]["body"]["tools"] = [
        {"type": "function", "function": {"name": "Bash", "parameters": {}}}
    ]
    return call


def test_a_title_call_to_the_runs_own_model_is_no_turn_of_the_agents(tmp_path):
    """Claude Code's ACP session also asks the run's own model for a session
    title: no tools and the default effort. Counted as a main call, it
    failed every run at another effort ("effort: requested max; the main
    calls carried max (2), medium (1)"), and coming last, it hid how the
    agent's own last turn ended."""
    title = _proxy_call("stop", effort="medium")
    result = _result(
        tmp_path / "t",
        [_turn("tool_calls"), _turn("length"), title],
        stop_reasons=[None],
    )
    sent = result["settings_sent"]
    assert (sent["main_calls"], sent["side_calls"]) == (2, 1)
    assert sent["effort"] == {"max": 2}
    assert sent["side_models"] == {"benchflow-anthropic-claude-sonnet-5": 1}
    totals = json.loads((tmp_path / "t" / "model-calls.json").read_text())
    assert totals["agent_effort"] == {"max": 2}
    assert totals["last_agent_call"]["finish_reason"] == "length"
    assert result["agent_outcome"]["truncated"] is True
    rows = [
        json.loads(line)
        for line in (tmp_path / "t" / "trajectory" / "model-calls.jsonl").open()
    ]
    assert [row["tools"] for row in rows] == [1, 1, 0]


def test_an_agent_that_names_its_tools_in_text_keeps_every_call(tmp_path):
    """mini-swe-agent's requests carry no tool list at all: none of its calls
    is set apart, so its effort is still compared."""
    result = _result(
        tmp_path / "t",
        [_proxy_call("stop", effort="max"), _proxy_call("stop", effort="medium")],
        stop_reasons=["end_turn"],
    )
    sent = result["settings_sent"]
    assert (sent["main_calls"], sent["side_calls"]) == (2, 0)
    assert sent["effort"] == {"max": 1, "medium": 1}


def _messages_call(*, effort, tools, title=False, stop="tool_use"):
    """One Anthropic Messages call through the proxy, shaped as the
    deterministic tier recorded Claude Code 2.1.316 at head 47a52fdf7
    (the claude-effort-acp job's trajectory/llm_trajectory.jsonl):
    the agent loop's calls offer its tools; after the turn the adapter asks
    the run's own model for the session's title, with no tools and
    output_config {effort: medium, format: json_schema(title)}."""
    config = {"effort": effort}
    if title:
        config["format"] = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
                "additionalProperties": False,
            },
        }
    content = (
        "<session>\nWrite hello.txt.\n</session>\n\nWrite the title"
        if title
        else "task"
    )
    return {
        "metadata": {
            "request_model": "claude-opus-5-5",
            "provider_model": "claude-opus-5-5",
            "call_type": "anthropic_messages",
            "call_purpose": "agent",
        },
        "request": {
            "body": {
                "model": "claude-opus-5-5",
                "max_tokens": 128000,
                "stream": True,
                "output_config": config,
                "messages": [{"role": "user", "content": content}],
                "tools": [{"name": n} for n in ("Bash", "Read", "Write")]
                if tools
                else [],
            }
        },
        "response": {
            "status_code": 200,
            "body": {
                "model": "claude-opus-5-5",
                "stop_reason": stop,
                "usage": {"input_tokens": 1000, "output_tokens": 50},
            },
        },
    }


async def test_claude_codes_session_title_call_does_not_fail_a_max_effort_run(
    tmp_path,
):
    """Guards 8fbeadf52 on the record shape Claude Code writes: every Claude
    Code (ACP) trial at --reasoning-effort max asks the run's own model for
    its session title after the turn, with no tools and effort medium. That
    background call is no turn of the agent's: the run does not fail, and
    every planned trial runs."""
    job = _job(tmp_path, 3, model="claude-opus-5-5", reasoning_effort="max")
    job._run_single_task = _trials_that(
        job,
        lambda name: [
            _messages_call(effort="max", tools=True),
            _messages_call(effort="max", tools=True, stop="end_turn"),
            _messages_call(effort="medium", tools=False, title=True, stop="end_turn"),
        ],
    )
    result = await job.run()
    checks = _summary(job)["run_checks"]
    assert (result.total, checks["failed"]) == (3, False)
    assert checks["settings"]["side_calls"] == 3


def test_a_passing_trial_is_not_withheld_for_a_failed_title_call(tmp_path):
    """Guards 8fbeadf52 on the same record shape: the agent's turn ended with
    its answer (end_turn, reward 1.0), and the session-title call the adapter
    sends after the turn got a 429. The title call is the log's last record
    but no call of the agent's, so the trial is no provider api_error: its
    reward stands and it is not retried."""
    from benchflow.diagnostics import RolloutDiagnostics
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.trajectories.model_calls import write_model_calls

    trial = tmp_path / "task__1"
    (trial / "trajectory").mkdir(parents=True)
    title = _messages_call(effort="medium", tools=False, title=True, stop="end_turn")
    title["response"] = {"status_code": 429, "body": {"error": {"type": "rate_limit"}}}
    rows = [
        _messages_call(effort="max", tools=True),
        _messages_call(effort="max", tools=True, stop="end_turn"),
        title,
    ]
    (trial / "trajectory" / "llm_trajectory.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    write_model_calls(trial)
    rollout = Rollout(
        RolloutConfig(
            task_path=Path("task"), agent="claude-agent-acp", model="claude-opus-5-5"
        )
    )
    rollout._rollout_dir = trial
    rollout._executed_prompts = ["task"]
    rollout._agent_env = {"BENCHFLOW_PROVIDER_NAME": "litellm"}
    rollout._usage_metrics = {"total_tokens": 4000, "n_output_tokens": 200}
    rollout._n_tool_calls = 4
    rollout._rewards = {"reward": 1.0}
    rollout._diagnostics = RolloutDiagnostics()

    rollout._maybe_classify_api_error()

    assert (rollout._error, rollout._rewards) == (None, {"reward": 1.0})


@pytest.mark.parametrize(
    ("served", "same"),
    [
        ("runtime-vllm-fake-policy", True),
        ("benchflow-vllm-fake-policy", True),
        ("runtime-vllm-other-policy", False),
    ],
)
def test_the_proxys_own_alias_for_the_model_is_the_model(served, same):
    """An upstream that echoes the request's model answers with BenchFlow's
    proxy alias for it (mini-swe through a vLLM route): that is the model the
    run asked for, under either the neutral name or the earlier one."""
    from benchflow.run_checks import same_model

    assert same_model("vllm/fake-policy", served) is same
