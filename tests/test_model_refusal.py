"""A model refusal or a content-filter stop is the trial's own ending.

Before 0.10.0rc2 a trial whose model refused (Anthropic's ``stop_reason:
refusal``, an OpenAI-compatible ``finish_reason: content_filter``, or ACP's
``refusal`` stop reason) ended like any other turn: ``agent_outcome.status``
said ``refusal`` beside an ordinary failure, and no summary counted it.
result.json now records the ending (``model_refusal``: which call, which
reason), the scoring rule counts such trials apart with their rate beside the
score while each still counts by its grade (0, as Terminal-Bench counts it),
and a refused attempt is final: never retried, never run again on resume.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

from benchflow._utils.result_paths import load_task_results
from benchflow._utils.scoring import score_summary_fields
from benchflow.evaluation import Evaluation, EvaluationConfig
from benchflow.models import RolloutResult
from benchflow.rollout._results import _build_rollout_result
from benchflow.run_checks import after_run_lines
from benchflow.scoring import Score, card_lines, score, trial_record
from benchflow.trajectories.model_calls import (
    agent_outcome,
    refusal_ending,
    write_model_calls,
)


def _chat_call(finish: str, *, refusal: str | None = None, tools: bool = True) -> dict:
    """One line of trajectory/llm_trajectory.jsonl for a chat-completions call."""
    body: dict = {"model": "m", "messages": [{"role": "user", "content": "task"}]}
    if tools:
        body["tools"] = [{"type": "function", "function": {"name": "bash"}}]
    message: dict = {"role": "assistant", "content": "x"}
    if refusal is not None:
        message["refusal"] = refusal
    return {
        "request": {"body": body, "timestamp": "2026-11-05T10:00:00+00:00"},
        "response": {
            "status_code": 200,
            "body": {
                "model": "m",
                "choices": [{"message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        },
        "metadata": {"call_purpose": "agent", "request_model": "m"},
        "duration_ms": 100,
    }


def _anthropic_call(stop: str) -> dict:
    """The same for an Anthropic Messages call (the proxy's pass-through)."""
    return {
        "request": {
            "body": {
                "model": "m",
                "messages": [{"role": "user", "content": "task"}],
                "tools": [{"name": "Bash"}],
            },
            "timestamp": "2026-11-05T10:00:00+00:00",
        },
        "response": {
            "status_code": 200,
            "body": {
                "model": "claude-x",
                "stop_reason": stop,
                "content": [],
                "usage": {"input_tokens": 100, "output_tokens": 0},
            },
        },
        "metadata": {"call_purpose": "agent", "request_model": "m"},
        "duration_ms": 100,
    }


def _result(
    trial: Path,
    calls: list[dict],
    *,
    stop_reasons: list[str | None],
    rewards: dict | None = None,
    error: str | None = None,
) -> dict:
    """result.json of a trial whose proxy recorded *calls*, built as a rollout
    builds it after teardown wrote the model calls."""
    (trial / "trajectory").mkdir(parents=True, exist_ok=True)
    (trial / "trajectory" / "llm_trajectory.jsonl").write_text(
        "".join(json.dumps(call) + "\n" for call in calls)
    )
    write_model_calls(trial)
    _build_rollout_result(
        trial,
        task_name="task",
        rollout_name=trial.name,
        agent="claude-agent-acp",
        agent_name="claude-agent-acp",
        model="anthropic/m",
        n_tool_calls=1,
        prompts=["task"],
        error=error,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        stop_reason=stop_reasons[-1],
        stop_reasons=stop_reasons,
        rewards=rewards if rewards is not None else {"reward": 0.0},
        started_at=datetime.now(),
        timing={},
        total_tokens=220,
    )
    return json.loads((trial / "result.json").read_text())


def test_a_content_filter_stop_is_recorded_with_its_call_and_reason(tmp_path):
    """The OpenAI-compatible shape: the agent's last call ended
    ``content_filter``; the record names that call and its reason."""
    result = _result(
        tmp_path / "t__1",
        [_chat_call("tool_calls"), _chat_call("content_filter")],
        stop_reasons=["end_turn"],
    )
    assert result["agent_outcome"]["status"] == "refusal"
    ending = result["model_refusal"]
    assert ending["by"] == "last_call"
    assert ending["stop_reason"] == "end_turn"
    assert ending["refused_calls"] == 1
    assert ending["call"]["index"] == 1
    assert ending["call"]["finish_reason"] == "content_filter"
    assert ending["call"]["source"] == "proxy"
    # It stays a scored failure in the headline: the rule is unchanged.
    assert result["rewards"] == {"reward": 0.0}
    assert result["error"] is None


def test_an_anthropic_refusal_and_an_acp_refusal_stop_are_recorded(tmp_path):
    """Anthropic's ``stop_reason: refusal`` on the agent's last call, and the
    agent's own ACP ``refusal`` stop reason, say the same: the stop reason
    decides ``by``."""
    result = _result(
        tmp_path / "t__2",
        [_anthropic_call("tool_use"), _anthropic_call("refusal")],
        stop_reasons=["refusal"],
    )
    ending = result["model_refusal"]
    assert ending["by"] == "stop_reason"
    assert ending["stop_reason"] == "refusal"
    assert ending["call"]["finish_reason"] == "refusal"
    assert ending["call"]["index"] == 1


def test_a_refusal_the_agent_went_on_from_is_no_ending(tmp_path):
    """A refused call followed by an answer the agent ended on is counted in
    ``refused_calls`` only: the trial did not end on it."""
    result = _result(
        tmp_path / "t__3",
        [_chat_call("content_filter"), _chat_call("stop")],
        stop_reasons=["end_turn"],
    )
    assert result["agent_outcome"]["status"] == "completed"
    assert result["agent_outcome"]["refused_calls"] == 1
    assert result["model_refusal"] is None


def test_a_refusal_answer_that_ends_stop_is_one():
    """A Chat Completions message's ``refusal`` ends ``stop``; with no record
    of the call (only the agent's stop reason) the call is unknown."""
    summary = {
        "last_agent_call": {"index": 0, "refused": True, "finish_reason": "stop"}
    }
    calls = [{"index": 0, "refused": True, "finish_reason": "stop", "source": "proxy"}]
    outcome = agent_outcome(summary)
    assert outcome["status"] == "refusal"
    ending = refusal_ending(outcome, summary, calls)
    assert ending is not None
    assert ending["call"]["finish_reason"] == "stop"
    bare = refusal_ending({"status": "refusal", "stop_reason": "refusal"}, None, [])
    assert bare == {
        "by": "stop_reason",
        "stop_reason": "refusal",
        "call": None,
        "refused_calls": None,
    }


def _rows(refused: int, failed: int, passed: int = 0) -> list[dict]:
    rows: list[dict] = []
    for i in range(refused):
        rows.append(
            {
                "task_name": f"r{i}",
                "rewards": {"reward": 0.0},
                "model_refusal": {"by": "last_call"},
                "agent_outcome": {"status": "refusal"},
            }
        )
    for i in range(failed):
        rows.append(
            {"task_name": f"f{i}", "rewards": {"reward": 0.0}, "model_refusal": None}
        )
    for i in range(passed):
        rows.append(
            {"task_name": f"p{i}", "rewards": {"reward": 1.0}, "model_refusal": None}
        )
    return rows


def test_the_rule_counts_model_refusals_apart_and_each_counts_by_its_grade():
    """3 refused of 20: the score is unchanged (each refused trial counts 0,
    being a failure), the refusals are counted apart with their rate, and the
    rate above 5% warns."""
    rows = _rows(refused=3, failed=7, passed=10)
    records = [trial_record(row) for row in rows]
    assert sum(r["model_refusal"] for r in records) == 3
    figures = score(records)
    assert (figures.n, figures.passed, figures.score) == (20, 10, 0.5)
    assert figures.outcomes["failed"] == 10
    assert figures.model_refusals == 3
    assert figures.rates()["model_refusal"] == 0.15
    assert any(w.startswith("model-refusal rate 15.0%") for w in figures.warnings())
    # summary.json beside the score, and the scoring block round trip.
    summary = score_summary_fields(rows)
    assert summary["model_refusals"] == 3
    assert summary["model_refusal_rate"] == 0.15
    assert summary["score_ratio"] == 0.5
    assert Score.from_dict(summary["scoring"]).model_refusals == 3
    # The plugin guard's refusals are another count, untouched.
    assert summary["guard_refusals"] == 0


def test_a_result_from_before_the_field_counts_by_its_agent_outcome():
    """A result.json written before ``model_refusal`` existed counts by its
    ``agent_outcome`` status, the same rule."""
    old = {
        "task_name": "t",
        "rewards": {"reward": 0.0},
        "agent_outcome": {"status": "refusal"},
    }
    assert trial_record(old)["model_refusal"] is True
    new = {**old, "model_refusal": None}
    assert trial_record(new)["model_refusal"] is False


def test_the_run_card_and_the_reports_show_the_refusal_rate():
    """The run card's after-run lines put the rate beside the Score line."""
    block = score_summary_fields(_rows(refused=1, failed=1, passed=2))["scoring"]
    lines = card_lines(block)
    assert lines[0].startswith("  Score:     2 of 4 trials passed")
    assert lines[1] == (
        "  Refusals:  1 of 4 trials (25.0%) ended on a model refusal or a content "
        "filter (each counts 0 unless its tests passed)"
    )
    after = after_run_lines({"scoring": block})
    assert lines[1] in after
    # None refused: the card gives the rate as 0, as it gives truncation's.
    block = score_summary_fields(_rows(0, 1, 1))["scoring"]
    assert card_lines(block)[1] == (
        "  Refusals:  0 of 2 trials (0.0%) ended on a model refusal or a content filter"
    )
    # A scoring block from before the count was kept says nothing of it.
    old = {key: value for key, value in block.items() if key != "model_refusals"}
    assert not any(line.startswith("  Refusals:") for line in card_lines(old))


def _job(tmp_path: Path, results: list[RolloutResult]) -> tuple[Evaluation, AsyncMock]:
    tasks = tmp_path / "tasks"
    task = tasks / "t"
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "task.toml").write_text('version = "1.0"\n[agent]\ntimeout_sec = 60\n')
    (task / "instruction.md").write_text("Do it.\n")
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    job = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(agent="claude-agent-acp", model="anthropic/m"),
    )
    runner = AsyncMock(side_effect=results)
    job._run_single_task = runner
    return job, runner


def test_a_refused_attempt_is_not_retried_whatever_its_error(tmp_path):
    """An agent that errored after its model refused (an ACP error, which the
    retry policy would otherwise run again) keeps the refusal as its result:
    another attempt would re-roll the model's answer."""
    refused = RolloutResult(
        "t",
        rollout_name="t__a",
        error="ACP error -32603: Internal error",
        error_category="acp_error",
        model_refusal={"by": "last_call"},
    )
    job, runner = _job(tmp_path, [refused, RolloutResult("t", rewards={"reward": 1.0})])
    result = asyncio.run(job._run_task(tmp_path / "tasks" / "t"))
    assert runner.await_count == 1
    assert result is refused
    # The same error without a refusal is retried, as before.
    plain = RolloutResult(
        "t", rollout_name="t__b", error="ACP error -32603", error_category="acp_error"
    )
    job, runner = _job(
        tmp_path / "b", [plain, RolloutResult("t", rewards={"reward": 1.0})]
    )
    asyncio.run(job._run_task(tmp_path / "b" / "tasks" / "t"))
    assert runner.await_count == 2


def test_a_resume_never_runs_a_refused_trial_again(tmp_path):
    """A refused trial that ended with no score is a finished result for
    resume: running it again would hide the refusal behind a new answer."""
    trial = tmp_path / "job" / "t__a"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "t",
                "rewards": None,
                "error": "ACP error -32603: Internal error",
                "model_refusal": {"by": "last_call"},
            }
        )
    )
    other = tmp_path / "job" / "u__a"
    other.mkdir()
    (other / "result.json").write_text(
        json.dumps({"task_name": "u", "rewards": None, "error": "ACP error -32603"})
    )
    assert set(load_task_results(tmp_path / "job")) == {"t"}


class _LiveProxy:
    """A proxy whose live capture has already read every call."""

    async def settle_live_capture(self, timeout: float = 5.0) -> None:
        return None

    def trial_budget_refusal(self):
        return None


async def test_an_agent_error_after_its_models_refusal_is_graded_and_final(tmp_path):
    """An agent that reports an error once its model refused (Codex ends a
    turn whose answer a content filter stopped with an error) is graded on
    what it left, as a refused trial that ended cleanly is: its refusal is
    the model's ending, so it is never retried away."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from benchflow.agents.errors import AgentProtocolError
    from benchflow.evaluation import RetryConfig
    from benchflow.rollout import Rollout, RolloutConfig

    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n[agent]\ntimeout_sec = 600\n'
    )
    (task / "instruction.md").write_text("Do it.\n")
    rollout = Rollout(RolloutConfig(task_path=task))
    trial = tmp_path / "trial"
    (trial / "trajectory").mkdir(parents=True)
    rollout._rollout_dir = trial
    rollout._rollout_name = "task__refused"
    rollout._started_at = datetime.now()
    rollout.setup = AsyncMock()
    rollout.start = AsyncMock()
    rollout.install_agent = AsyncMock()

    async def agent(_steps):
        (trial / "trajectory" / "llm_trajectory.jsonl").write_text(
            "".join(
                json.dumps(call) + "\n"
                for call in (_chat_call("tool_calls"), _chat_call("content_filter"))
            )
        )
        rollout._usage_runtime = SimpleNamespace(kind="litellm", server=_LiveProxy())
        raise AgentProtocolError("ACP error -32603: Internal error")

    async def verify():
        rollout._rewards = {"reward": 0.0}
        rollout._phase = "verified"
        return rollout._rewards

    async def cleanup():
        write_model_calls(trial)
        rollout._phase = "cleaned"

    rollout._run_steps = agent
    rollout.verify = AsyncMock(side_effect=verify)
    rollout.cleanup = AsyncMock(side_effect=cleanup)
    result = await asyncio.wait_for(rollout.run(), timeout=30)
    rollout.verify.assert_awaited_once()
    assert result.rewards == {"reward": 0.0}
    assert result.model_refusal is not None
    assert result.model_refusal["call"]["finish_reason"] == "content_filter"
    saved = json.loads((trial / "result.json").read_text())
    assert trial_record(saved)["outcome"] == "failed"
    assert trial_record(saved)["model_refusal"] is True
    # The kept error alone would be retried; the refusal makes it final.
    assert RetryConfig().should_retry(result.error, category=result.error_category)


def test_the_end_of_run_lines_name_refusals_and_budget_stops(tmp_path, monkeypatch):
    """The Outcomes lines name the trials that ended on a model refusal and
    those a trial budget stopped, from their results alone (a row from before
    ``model_refusal`` by its agent_outcome): a passed trial's file is never
    read for it."""
    import io

    from rich.console import Console

    import benchflow.cli._shared as shared
    from benchflow.evaluation import EvaluationResult

    job = tmp_path / "jobs" / "run"
    job.mkdir(parents=True)
    (job / "evaluation.json").write_text(json.dumps({"tasks_dir": str(tmp_path)}))

    def payload(name: str, **fields) -> dict:
        folder = job / f"{name}__abcd1234"
        folder.mkdir()
        data = {"task_name": name, "rollout_name": folder.name, **fields}
        (folder / "result.json").write_text(json.dumps(data))
        return data

    ending = {"by": "stop_reason", "stop_reason": "refusal", "call": None}
    spent = {"budget": {"tokens": None, "turns": 4}, "stopped_by": "turns"}
    results = {
        "fine": RolloutResult.from_dict(payload("fine", rewards={"reward": 1.0})),
        "old": payload(
            "old", rewards={"reward": 0.0}, agent_outcome={"status": "refusal"}
        ),
        "refused": RolloutResult.from_dict(
            payload("refused", rewards={"reward": 0.0}, model_refusal=ending)
        ),
        "spent": RolloutResult.from_dict(
            payload(
                "spent",
                rewards={"reward": 0.0},
                error="trial budget reached: the turn budget of 4 is spent",
                error_category="trial_budget",
                trial_budget=spent,
            )
        ),
    }
    result = EvaluationResult(
        job_name="run",
        config=EvaluationConfig(),
        total=4,
        passed=1,
        job_dir=job,
        results=results,
    )
    reads: list[str] = []
    read_text = Path.read_text

    def counting(self, *args, **kwargs):
        reads.append(self.parent.name if self.name == "result.json" else self.name)
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting)
    console = Console(file=io.StringIO(), width=240)
    monkeypatch.setattr(shared, "console", console)
    shared._report_outcomes(result, job)
    out = " ".join(console.file.getvalue().split())
    assert (
        "2 ended on a model refusal or a content filter (each counts 0 unless its "
        "tests passed; never retried): old, refused"
    ) in out
    assert (
        "1 stopped at their trial budget (1 by turns; each counts 0 unless its "
        "tests passed): spent"
    ) in out
    assert "fine__abcd1234" not in reads
