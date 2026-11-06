"""The run checks on a published truncation case.

A published run of Claude Code on Sonnet 5 scored 330 trials (66 tasks, 5
tries each) and 37 of them ended at the output cap (11.2%). Its configuration
set ``CLAUDE_CODE_MAX_OUTPUT_TOKENS`` to 128,000; the calls carried 32,000.
This builds that job from the records a trial leaves (the proxy's
``llm_trajectory.jsonl``, the per-call ``model-calls.jsonl`` teardown writes,
and the ``result.json`` the rollout builds), runs it as BenchFlow runs a job,
and reads what the card, ``summary.json`` and ``bench run``'s exit code say.

A job keeps one trial per task, so the 5 tries of each task are 5 task
folders here; ``--trials 5`` would run 5 jobs of 66, each checked alike.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from benchflow.acp.session import ACPSession
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.rollout._results import _build_rollout_result
from benchflow.run_checks import after_run_lines
from benchflow.trajectories._capture import _capture_session_trajectory
from benchflow.trajectories.model_calls import write_model_calls

TASKS, TRIES = 66, 5
MODEL = "anthropic/claude-sonnet-5"
CAP_ERROR = (
    "ACP error -32603: Internal error: API Error: Claude's response exceeded the "
    "32000 output token maximum. To configure this behavior, set the "
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS environment variable."
)


def _message(stop: str, *, max_tokens: int, n: int) -> dict:
    """One Anthropic Messages call through the proxy, as its log records it."""
    return {
        "duration_ms": 8000.0,
        "metadata": {
            "request_model": "claude-sonnet-5",
            "provider_model": MODEL,
            "call_type": "anthropic_messages",
            "call_purpose": "agent",
            "response_cost": 0.02,
        },
        "request": {
            "timestamp": f"2026-08-28T10:{n:02d}:00",
            "body": {
                "model": "claude-sonnet-5",
                "stream": True,
                "max_tokens": max_tokens,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            },
        },
        "response": {
            "status_code": 200,
            "timestamp": f"2026-08-28T10:{n:02d}:08",
            "body": {
                "id": f"msg_{n}",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-20260301",
                "stop_reason": stop,
                "usage": {
                    "input_tokens": 1200,
                    "output_tokens": max_tokens if stop == "max_tokens" else 900,
                    "cache_read_input_tokens": 30000,
                },
            },
        },
    }


# Claude Code's background call to its small model: the run's proxy serves
# only the run's model, so it is refused and recorded as such.
_REFUSED = {
    "event": "refused",
    "rule": "model-not-served",
    "request_model": "claude-haiku-4-5",
    "call_type": "anthropic_messages",
}


def _trial(job_dir: Path, task: str, *, cut: bool, max_tokens: int):
    """One finished trial's folder: 6 tool calls, then the answer, or a
    response that ran into the cap and ended the agent's turn."""
    trial = job_dir / f"{task}__0a1b2c3d"
    (trial / "trajectory").mkdir(parents=True)
    calls = [_message("tool_use", max_tokens=max_tokens, n=i) for i in range(6)]
    calls.append(
        _message("max_tokens" if cut else "end_turn", max_tokens=max_tokens, n=6)
    )
    log = [json.dumps(_REFUSED)] + [json.dumps(call) for call in calls]
    (trial / "trajectory" / "llm_trajectory.jsonl").write_text("\n".join(log) + "\n")
    write_model_calls(trial)
    session = ACPSession(task)
    session.record_user_prompt("task")
    session.mark_prompt_end()
    return _build_rollout_result(
        trial,
        task_name=task,
        rollout_name=trial.name,
        agent="claude-agent-acp",
        agent_name="claude-agent-acp",
        model=MODEL,
        n_tool_calls=6,
        prompts=["task"],
        error=CAP_ERROR if cut else None,
        verifier_error=None,
        trajectory=_capture_session_trajectory(session),
        partial_trajectory=False,
        stop_reason=None if cut else "end_turn",
        stop_reasons=[None] if cut else ["end_turn"],
        rewards={"reward": 0.0 if cut else 1.0},
        started_at=datetime.now(),
        timing={},
        total_tokens=200_000,
    )


def _names() -> list[str]:
    return [f"task-{t:02d}-try-{k}" for t in range(TASKS) for k in range(1, TRIES + 1)]


def _job(tmp_path: Path, agent_env: dict[str, str], *, name: str) -> Evaluation:
    tasks = tmp_path / "tasks"
    for task in _names():
        folder = tasks / task
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "task.toml").write_text(
            'version = "1.0"\n[agent]\ntimeout_sec = 3600\n[verifier]\ntimeout_sec = 600\n'
        )
        (folder / "instruction.md").write_text("Solve the task.\n")
    return Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent="claude-agent-acp",
            model=MODEL,
            agent_env=agent_env,
            concurrency=1,
            retry=RetryConfig(max_retries=0),
        ),
        job_name=name,
        preflight=False,
    )


def _exit_code(result, job_dir) -> int:
    """What `bench run` exits with for *result* (cli.main._finish_run)."""
    from types import SimpleNamespace

    import typer

    from benchflow.cli.main import _finish_run

    try:
        _finish_run(
            SimpleNamespace(fail_under=None, fail_on=None, summary_out=None),
            result,
            job_dir,
        )
    except typer.Exit as stop:
        return stop.exit_code
    return 0


def _read(job: Evaluation, name: str) -> dict:
    return json.loads((job._jobs_dir / job._job_name / name).read_text())


def test_the_published_case_fails_on_the_cap_and_warns_on_its_truncation(tmp_path):
    """128,000 asked for, 32,000 sent, 37 of 330 trials cut: the cap's
    source is on the card, the mismatch fails the run, and the 11.2%
    truncation rate warns without refusing it."""
    job = _job(tmp_path, {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000"}, name="published")
    job_dir = job._jobs_dir / job._job_name
    cut = set(_names()[::9][:37])
    for task in _names():
        _trial(job_dir, task, cut=task in cut, max_tokens=32000)

    async def no_new_trial(task_path, _cfg):
        raise AssertionError(f"{task_path.name} had already run")

    job._run_single_task = no_new_trial
    result = asyncio.run(job.run())
    assert result.total == 330

    card = _read(job, "run-card.json")
    assert (card["agent"]["output_cap"], card["agent"]["output_cap_source"]) == (
        "128000",
        "agent_env",
    )
    checks = _read(job, "summary.json")["run_checks"]
    [mismatch] = checks["settings"]["mismatches"]
    assert {
        key: mismatch[key] for key in ("setting", "requested", "source", "fails")
    } == {
        "setting": "max_output_tokens",
        "requested": 128000,
        "source": "agent_env",
        "fails": True,
    }
    assert mismatch["sent"] == {"32000": 330 * 7}
    assert (mismatch["calls"], mismatch["of"], mismatch["trials"]) == (2310, 2310, 330)
    assert checks["failed"] is True
    assert checks["failures"] == [
        "max_output_tokens: requested 128000 (from --agent-env); the main calls "
        "carried 32000 (2310): 2310 of 2310 calls differ, in 330 trial(s), first "
        f"{sorted(_names())[0]}__0a1b2c3d"
    ]
    assert _exit_code(result, job_dir) == 1

    truncation = checks["truncation"]
    assert (truncation["known"], truncation["truncated"]) == (330, 37)
    assert truncation["rate"] == 0.1121
    assert truncation["level"] == "warn"
    assert not any(line.startswith("truncation") for line in checks["failures"])
    assert checks["warnings"] == [
        "truncation: 37 of 330 trials with a known ending were cut at the output "
        "cap (11.2%), at or above 10%"
    ]
    assert after_run_lines(checks)[1] == (
        "  Truncated: 37 of 330 trials with a known ending were cut at the output cap "
        "(11.2%): at or above 10%, a warning; 50% refuses the run"
    )
    # The background calls the proxy refused are not the agent's main calls.
    assert checks["settings"]["main_calls"] == 2310


def _live(tmp_path: Path, *, cut: set[str], name: str):
    """The same run, live: each trial written as it finishes, with the cap
    and the calls agreeing at 128,000."""
    job = _job(tmp_path, {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000"}, name=name)
    job_dir = job._jobs_dir / job._job_name

    async def run(task_path, _cfg):
        return _trial(
            job_dir, task_path.name, cut=task_path.name in cut, max_tokens=128000
        )

    job._run_single_task = run
    return job, asyncio.run(job.run())


def test_half_the_trials_cut_refuses_the_run(tmp_path):
    names = sorted(_names())
    job, result = _live(tmp_path, cut=set(names[::2]), name="half")
    checks = _read(job, "summary.json")["run_checks"]
    # The first 5 trials finished, 3 of them cut (60%): no sixth started.
    assert result.total == 5 and result.not_run == 325
    assert checks["truncation"]["level"] == "refuse"
    assert checks["settings"]["mismatches"] == []
    assert checks["stopped"] is True and checks["failed"] is True
    assert checks["failures"] == [
        "truncation: 3 of 5 trials with a known ending were cut at the output cap "
        "(60.0%), at or above 50%"
    ]
    assert _exit_code(result, job._jobs_dir / job._job_name) == 1


def test_a_cap_the_calls_carried_and_few_cuts_pass(tmp_path):
    names = sorted(_names())
    job, result = _live(tmp_path, cut=set(names[::11]), name="clean")
    checks = _read(job, "summary.json")["run_checks"]
    assert result.total == 330
    assert checks["truncation"]["truncated"] == 30
    assert checks["truncation"]["level"] == "ok"
    assert (checks["failed"], checks["failures"], checks["warnings"]) == (False, [], [])
    assert _exit_code(result, job._jobs_dir / job._job_name) == 0
