"""An idle timeout is judged on the attempt the run keeps, and every reader
reads a recorded ending as written.

Guards the fix on branch ``idle-timeout-one-picker`` (the
lead's rulings of 2026-11-13: the writer decides a trial's ending, and an agent
that did nothing is withheld on the attempt the run keeps). The rollout leaves
an idle timeout unjudged so the run can retry it, and readers judged it again
on read: an agent that did nothing until the idle watchdog stopped it was
``idle_timeout``, reward 0.25, scored in its result.json, summary.json and run
card, and an integration failure, unscored, in ``bf.load_job``, ``bench
results`` and the job export.

A rollout no retry loop owns (``bf.run``, ``bf.run_batch``, a continued run)
is final as it ends, so it judges its zero-activity idle timeout when it writes
it; only the Evaluation's retry loop owns an attempt (the lead's ruling of
2026-11-14: the default is "not owned"). The loop owns an attempt only while
it may still retry it: its last try, and every try of a policy that never
retries an idle timeout, is judged when written too, so only an attempt the
loop keeps early (its budget, its run checks) is rewritten (F2b, after review
R187: one write and one span send where the loop knows the try is final).

Each attempt below is written by the rollout's own result writer
(``Rollout._build_result``) after the idle watchdog's timeout, inside a real
``Evaluation.run()`` (from the config the Evaluation builds for it), or from
the entry point's own rollout.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import benchflow as bf
from benchflow._utils.attempt_decision import FINAL, RETRIED, decision_of
from benchflow._utils.result_paths import load_task_results
from benchflow.cli.main import app
from benchflow.diagnostics import IdleTimeoutDiagnostic, IdleTimeoutError
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.job_export import job_export
from benchflow.metrics import collect_metrics
from benchflow.models import RolloutResult
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.rollout._results import judge_kept_idle_timeout

AGENT = "claude-agent-acp"
MODEL = "anthropic/claude-sonnet-5"
INSTRUCTION = "Write 42 to /app/answer.txt."
PROMPT = {"type": "user_message", "text": INSTRUCTION}
IDLE = {
    "type": "agent_timeout",
    "reason": "idle_timeout",
    "timeout_sec": 600,
    "pending_tool_call_ids": [],
    "terminal_trajectory_complete": True,
}
TOOL_CALL = {
    "type": "tool_call",
    "tool_call_id": "call-1",
    "kind": "edit",
    "title": "Write /app/answer.txt",
    "status": "completed",
    "content": [],
}
MESSAGE = {"type": "agent_message", "text": "I will write the answer first."}


@dataclass(frozen=True)
class Attempt:
    """What the agent did after the prompt: *events*, its tool calls and
    output tokens; then the idle watchdog stopped it (*idle*), and the
    verifier gave the workspace *reward*."""

    events: tuple[dict[str, Any], ...]
    tool_calls: int = 0
    output_tokens: int = 0
    idle: bool = True
    reward: float = 0.25


NOTHING = Attempt((PROMPT, IDLE))
A_TOOL_CALL = Attempt((PROMPT, TOOL_CALL, IDLE), tool_calls=1)
AN_ANSWER = Attempt((PROMPT, MESSAGE, IDLE), output_tokens=40)
ONLY_TOKENS = Attempt((PROMPT, IDLE), output_tokens=50)
SOLVED = Attempt(
    (PROMPT, TOOL_CALL, {"type": "agent_message", "text": "Done."}),
    tool_calls=1,
    output_tokens=80,
    idle=False,
    reward=1.0,
)


def _write_attempt(
    job: Evaluation, task: Path, attempt: Attempt, flags: list[bool] | None = None
) -> RolloutResult:
    """One of the retry loop's attempts, from the config the Evaluation builds
    for it (*flags* collects each attempt's ``owned_by_retry_loop``)."""
    folder = job._jobs_dir / job._job_name
    n = len(list(folder.glob(f"{task.name}__*"))) if folder.is_dir() else 0
    config = job._rollout_config(task, job._config)
    if flags is not None:
        flags.append(config.owned_by_retry_loop)
    return _finish(Rollout(config), folder / f"{task.name}__{n:08x}", attempt)


def _finish(
    rollout: Rollout,
    folder: Path,
    attempt: Attempt,
    *,
    usage: dict[str, Any] | None = None,
    partial: bool = False,
    setup: Any = None,
) -> RolloutResult:
    """End *rollout*'s turn as *attempt* says and write its records (*usage*
    adds to the usage the turn reported; *partial*: the trajectory was
    salvaged from the timed-out session)."""
    _rollout(folder, rollout._config.task_path, attempt, rollout=rollout)
    rollout._usage_metrics.update(usage or {})
    if partial:
        rollout._partial_trajectory = True
        rollout._trajectory_source = "partial_acp"
    if setup is not None:
        setup(rollout)
    if attempt.idle:
        detail = (
            "Agent idle for 600s with no new tool call "
            f"({attempt.tool_calls} tool calls so far)"
        )
        diagnostic = IdleTimeoutDiagnostic(
            idle_timeout_sec=600, n_tool_calls=attempt.tool_calls
        )
        rollout._record_agent_timeout(
            IdleTimeoutError(detail, diagnostic), agent_phase=True
        )
    return rollout._build_result()


def _rollout(
    folder: Path, task: Path, attempt: Attempt, *, rollout: Rollout | None = None
) -> Rollout:
    """A rollout whose agent's turn left *attempt*, writing into *folder*."""
    if rollout is None:
        rollout = Rollout(RolloutConfig(task_path=task, agent=AGENT, model=MODEL))
    rollout._rollout_dir = folder
    (rollout._rollout_dir / "agent").mkdir(parents=True)
    rollout._rollout_name = rollout._rollout_dir.name
    rollout._started_at = datetime.now()
    rollout._timeout = 3600
    rollout._resolved_prompts = [INSTRUCTION]
    rollout._executed_prompts = [INSTRUCTION]
    rollout._agent_env = {"BENCHFLOW_PROVIDER_NAME": "litellm"}
    rollout._usage_metrics = {
        "n_input_tokens": 1200,
        "n_output_tokens": attempt.output_tokens,
        "total_tokens": 1200 + attempt.output_tokens,
        "usage_source": "provider_response",
    }
    rollout._rewards = {"reward": attempt.reward}
    rollout._trajectory = [dict(event) for event in attempt.events]
    rollout._trajectory_source = "acp"
    rollout._n_tool_calls = attempt.tool_calls
    rollout._timing = {"agent_execution": 600.0}
    return rollout


def _task(tmp_path: Path) -> Path:
    task = tmp_path / "tasks" / "t"
    task.mkdir(parents=True)
    (task / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 3600\n[environment]\n"
    )
    (task / "instruction.md").write_text(INSTRUCTION + "\n")
    return task


def _job(
    tmp_path: Path,
    attempts: list[Attempt],
    retry: RetryConfig,
    flags: list[bool] | None = None,
) -> Evaluation:
    """A one-task Evaluation whose attempts are *attempts*, in order."""
    _task(tmp_path)
    job = Evaluation(
        tasks_dir=tmp_path / "tasks",
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(agent=AGENT, model=MODEL, retry=retry),
        preflight=False,
    )

    async def run_single(task_path, _cfg):
        return _write_attempt(job, task_path, attempts.pop(0), flags)

    async def no_sweep():
        return None

    job._run_single_task = run_single
    job._sweep_docker = no_sweep
    return job


def _run(
    tmp_path: Path,
    *attempts: Attempt,
    flags: list[bool] | None = None,
    retry: RetryConfig | None = None,
) -> Path:
    """Run a one-task job whose attempts are *attempts* (retries allowed up to
    one fewer than there are), and check that each of them ran."""
    queue = list(attempts)
    if retry is None:
        retry = RetryConfig(max_retries=len(attempts) - 1, min_wait_sec=0)
    job = _job(tmp_path, queue, retry, flags)
    asyncio.run(job.run())
    assert queue == [], "the run did not retry every attempt"
    return job._jobs_dir / job._job_name


class _NoRoomForARetry:
    """A job budget with no room for another rollout: the retry loop keeps
    the attempt it would have retried (an early final exit)."""

    reason = "the budget has no room for another rollout"

    def attempt_done(self, name: str, result: Any) -> None:
        return None

    def retry(self, name: str) -> bool:
        return False


def _kept_early(tmp_path: Path, attempt: Attempt) -> tuple[Path, RolloutResult]:
    """The retry loop's first try, which it may retry, kept by the budget."""
    queue = [attempt]
    job = _job(tmp_path, queue, RetryConfig(max_retries=1, min_wait_sec=0))
    job._budget_guard = _NoRoomForARetry()
    result = asyncio.run(job._run_task(tmp_path / "tasks" / "t"))
    return job._jobs_dir / job._job_name, result


def _attempts(job_dir: Path) -> list[Path]:
    return sorted(p for p in job_dir.iterdir() if (p / "result.json").is_file())


def _result(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "result.json").read_text())


def _reward(rewards: Any) -> Any:
    return rewards.get("reward") if isinstance(rewards, dict) else rewards


def _row(job_dir: Path, folder: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (job_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    [row] = [r for r in rows if r["info"]["rollout_dir"] == str(folder)]
    return row


def _trial_span(folder: Path) -> dict[str, Any]:
    spans = json.loads((folder / "trajectory" / "otel-spans.json").read_text())
    span = spans["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    return {a["key"]: next(iter(a["value"].values()), None) for a in span["attributes"]}


def _outcome(counts: dict[str, Any]) -> str:
    return next(k for k in ("passed", "failed", "errored") if counts.get(k))


def _views(job_dir: Path) -> dict[str, tuple[Any, ...]]:
    """The trial the job kept, as each reader reads it: its category and its
    reward (None when unscored), and what it says of the trial's ending."""
    loaded = bf.load_job(job_dir)
    [trial] = loaded.trials
    folder = Path(trial.path)
    alone = bf.load_trial(folder)
    [exported] = job_export(loaded).trials
    metrics = collect_metrics(job_dir)
    [task] = metrics.tasks
    listed = CliRunner().invoke(app, ["results", str(job_dir), "--json"])
    assert listed.exit_code == 0, listed.output
    [shown] = json.loads(listed.output)["trials"]
    resumed = load_task_results(job_dir, include_terminal_errors=True)["t"]
    recorded = _result(folder)
    row = _row(job_dir, folder)
    span = _trial_span(folder)
    summary = json.loads((job_dir / "summary.json").read_text())
    card = json.loads((job_dir / "run-card.json").read_text())["after_run"]["scoring"]
    return {
        "result.json": (recorded["error_category"], _reward(recorded["rewards"])),
        "results.jsonl": (
            row["info"]["diagnostics"]["error_category"],
            row["reward"],
            row["attempt_decision"],
        ),
        "trial spans": (span["benchflow.error.category"], span.get("benchflow.reward")),
        "resume": (resumed["error_category"], _reward(resumed["rewards"])),
        "load_job": (trial.result.error_category, trial.reward, trial.execution),
        "load_trial": (alone.result.error_category, alone.reward, alone.execution),
        "export": (exported.error_category, exported.reward, exported.execution),
        "bench results": (shown["error_category"], shown["reward"], shown["execution"]),
        "metrics": (metrics.error_breakdown, task.reward),
        "summary.json": (summary["error_categories"], _outcome(summary)),
        "run card": (_outcome(card["outcomes"]),),
    }


def _everywhere(category: str, reward: float | None, execution: str):
    outcome = "errored" if reward is None else "failed"
    return {
        "result.json": (category, reward),
        "results.jsonl": (category, reward, FINAL),
        "trial spans": (category, reward),
        "resume": (category, reward),
        "load_job": (category, reward, execution),
        "load_trial": (category, reward, execution),
        "export": (category, reward, execution),
        "bench results": (category, reward, execution),
        # The breakdown counts the errored (unscored) trials.
        "metrics": ({category: 1} if reward is None else {}, reward),
        "summary.json": ({category: 1}, outcome),
        "run card": (outcome,),
    }


WITHHELD = _everywhere("agent_integration", None, "integration_failed")
SCORED_IDLE = _everywhere("idle_timeout", 0.25, "errored")


def test_a_zero_activity_idle_timeout_that_will_be_retried_is_retried(tmp_path):
    """1. The attempt the run retries stays an idle timeout: it is retried,
    and its own records and every reader of it still say idle_timeout with
    the verifier's reward (a judged attempt would not be retried)."""
    job_dir = _run(tmp_path, NOTHING, SOLVED)

    retried, kept = _attempts(job_dir)
    assert (decision_of(retried), decision_of(kept)) == (RETRIED, FINAL)
    recorded = _result(retried)
    assert (recorded["error_category"], recorded["rewards"]) == (
        "idle_timeout",
        {"reward": 0.25},
    )
    assert recorded["integration_failure_info"] is None
    assert (retried / "rewards.jsonl").is_file()
    row = _row(job_dir, retried)
    assert (row["attempt_decision"], row["reward"]) == (RETRIED, 0.25)
    assert row["info"]["diagnostics"]["error_category"] == "idle_timeout"
    attempt = bf.load_trial(retried)
    assert (attempt.result.error_category, attempt.execution) == (
        "idle_timeout",
        "errored",
    )
    assert attempt.integration_failure is None
    assert attempt.reward == 0.25
    # The run kept the attempt that solved the task.
    [trial] = bf.load_job(job_dir).trials
    assert (Path(trial.path), trial.reward) == (kept, 1.0)
    assert [Path(a.path) for a in trial.attempts] == [retried, kept]
    assert trial.attempts[0].execution == "errored"


def test_the_final_zero_activity_idle_timeout_is_withheld_everywhere(tmp_path):
    """2. The attempt the run keeps is judged when the run keeps it: an
    agent_integration failure (no_activity), the verifier's reward withheld,
    in result.json, its row, trainer files and spans, and in every reader."""
    job_dir = _run(tmp_path, NOTHING, NOTHING)

    first, kept = _attempts(job_dir)
    assert (decision_of(first), decision_of(kept)) == (RETRIED, FINAL)
    assert _result(first)["error_category"] == "idle_timeout"
    assert _views(job_dir) == WITHHELD
    recorded = _result(kept)
    info = recorded["integration_failure_info"]
    assert (info["cause"], info["reward_withheld"]) == ("no_activity", {"reward": 0.25})
    assert (info["tool_calls"], info["agent_messages"], info["output_tokens"]) == (
        0,
        0,
        0,
    )
    assert recorded["error"].startswith("agent integration failure [no_activity]")
    # The watchdog's own record stays; the integration failure owns the category.
    assert recorded["idle_timeout_info"]["idle_timeout_sec"] == 600
    assert not (kept / "rewards.jsonl").exists()
    trainer = json.loads((kept / "trainer" / "verifiers.jsonl").read_text())
    assert trainer["info"]["reward_valid"] is False
    summary = json.loads((job_dir / "summary.json").read_text())
    assert summary["integration_failures"] == {
        "total": 1,
        "by_cause": {"no_activity": 1},
    }
    assert (
        collect_metrics(job_dir).integration_failures == summary["integration_failures"]
    )
    # A resume runs it again, as it runs any trial whose agent did nothing.
    assert load_task_results(job_dir) == {}


@pytest.mark.parametrize(
    "active",
    [A_TOOL_CALL, AN_ANSWER, ONLY_TOKENS],
    ids=["a-tool-call", "the-model-answered", "output-tokens"],
)
def test_an_idle_timeout_with_activity_is_scored_everywhere(tmp_path, active):
    """3. An agent that acted before it went idle keeps its timeout and its
    reward, in every reader."""
    job_dir = _run(tmp_path, active)

    [kept] = _attempts(job_dir)
    assert decision_of(kept) == FINAL
    assert _views(job_dir) == SCORED_IDLE
    assert _result(kept)["integration_failure_info"] is None
    assert (kept / "rewards.jsonl").is_file()


@pytest.mark.parametrize(
    ("attempts", "categories", "views"),
    [
        (
            (NOTHING, NOTHING, NOTHING),
            ["idle_timeout", "idle_timeout", "agent_integration"],
            WITHHELD,
        ),
        (
            (NOTHING, A_TOOL_CALL),
            ["idle_timeout", "idle_timeout"],
            SCORED_IDLE,
        ),
    ],
    ids=["did-nothing-every-time", "acted-on-the-last-try"],
)
def test_the_attempt_record_and_every_reader_agree(
    tmp_path, attempts, categories, views
):
    """4. attempt-decision.json marks one attempt final; every reader takes
    that attempt, with the category its result.json records, and reads each
    attempt the run retried as it was written."""
    job_dir = _run(tmp_path, *attempts)

    folders = _attempts(job_dir)
    *retried, final = folders
    assert [decision_of(f) for f in folders] == [RETRIED] * len(retried) + [FINAL]
    assert [_result(f)["error_category"] for f in folders] == categories
    assert _views(job_dir) == views
    loaded = bf.load_job(job_dir)
    listed = json.loads(
        CliRunner().invoke(app, ["results", str(job_dir), "--json"]).output
    )
    picks = {
        "load_job": Path(loaded.trials[0].path),
        "export": Path(job_export(loaded).trials[0].path),
        "bench results": Path(listed["trials"][0]["path"]),
        "resume": job_dir
        / load_task_results(job_dir, include_terminal_errors=True)["t"]["rollout_name"],
        "results.jsonl": Path(
            next(
                row["info"]["rollout_dir"]
                for row in map(
                    json.loads, (job_dir / "results.jsonl").read_text().splitlines()
                )
                if row["attempt_decision"] == FINAL
            )
        ),
    }
    assert picks == dict.fromkeys(picks, final)
    for folder in retried:
        attempt = bf.load_trial(folder)
        assert (attempt.result.error_category, attempt.execution, attempt.reward) == (
            "idle_timeout",
            "errored",
            0.25,
        )
        assert _row(job_dir, folder)["attempt_decision"] == RETRIED


# Stored results: what an earlier run wrote is read as written.


def _stored_trial(root: Path, recorded: dict[str, Any]) -> Path:
    """A finished trial folder whose agent did nothing after the prompt."""
    folder = root / "t__00000001"
    (folder / "trajectory").mkdir(parents=True)
    (folder / "agent").mkdir()
    (folder / "result.json").write_text(
        json.dumps(
            {
                "task_name": "t",
                "rollout_name": folder.name,
                "agent": AGENT,
                "model": MODEL,
                "rewards": {"reward": 0.25},
                "n_tool_calls": 0,
                "n_prompts": 1,
                "trajectory_source": "acp",
                "agent_result": {"n_output_tokens": 0},
                "timing": {"agent_execution": 600.0},
                **recorded,
            }
        )
    )
    events = [PROMPT, IDLE] if recorded.get("error") else [PROMPT]
    (folder / "trajectory" / "acp_trajectory.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    return folder


@pytest.mark.parametrize(
    ("recorded", "read_as"),
    [
        # No category: a result from before the writer judged endings.
        ({"error": None}, (None, None, "integration_failed")),
        (
            {
                "error": "Agent idle for 600s with no new tool call (0 tool calls so far)",
                "error_category": "idle_timeout",
            },
            ("idle_timeout", 0.25, "errored"),
        ),
        (
            {
                "error": "Agent prompt exceeded wall-clock budget 3600s",
                "error_category": "timeout",
            },
            ("timeout", 0.25, "timed_out"),
        ),
    ],
    ids=["no-category-is-judged", "idle_timeout-as-written", "timeout-as-written"],
)
def test_only_a_result_with_no_category_is_judged_on_read(tmp_path, recorded, read_as):
    """5. A trial whose result.json records no category still gets the
    read-time diagnosis; one with a category is read as written."""
    job = tmp_path / "job"
    folder = _stored_trial(job, recorded)
    category, reward, execution = read_as

    alone = bf.load_trial(folder)
    [trial] = bf.load_job(job).trials
    [exported] = job_export(bf.load_job(job)).trials
    listed = CliRunner().invoke(app, ["results", str(job), "--json"])
    assert listed.exit_code == 0, listed.output
    [shown] = json.loads(listed.output)["trials"]
    [task] = collect_metrics(job).tasks
    seen = {
        "load_trial": (alone.reward, alone.execution),
        "load_job": (trial.reward, trial.execution),
        "export": (exported.reward, exported.execution),
        "bench results": (shown["reward"], shown["execution"]),
        "metrics": (task.reward,),
    }
    assert seen == {
        "load_trial": (reward, execution),
        "load_job": (reward, execution),
        "export": (reward, execution),
        "bench results": (reward, execution),
        "metrics": (reward,),
    }
    judged = trial.integration_failure
    if category is None:
        assert judged is not None
        assert (judged["cause"], judged["detected"]) == ("empty_trajectory", "on read")
        assert collect_metrics(job).integration_failures["total"] == 1
    else:
        assert judged is None
        assert trial.result.error_category == category
        assert collect_metrics(job).integration_failures["total"] == 0


def test_the_metrics_error_breakdown_counts_the_recorded_category(tmp_path):
    """``bench eval metrics``' error breakdown classified each error's text
    again. An agent connection lost with a message that names no transport
    marker is ``pipe_closed`` in result.json (its transport record decides),
    and the breakdown counted it as ``other``."""
    task = tmp_path / "tasks" / "t"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text(INSTRUCTION + "\n")
    folder = tmp_path / "job" / "t__00000001"
    rollout = _rollout(folder, task, Attempt((PROMPT, TOOL_CALL), tool_calls=1))
    # What the lifecycle's handler records for a lost agent connection.
    lost = ConnectionError("Server disconnected without sending a response.")
    rollout._error = str(lost)
    rollout._diagnostics.capture_transport(lost)
    rollout._rewards = None
    rollout._build_result()
    assert _result(folder)["error_category"] == "pipe_closed"

    metrics = collect_metrics(tmp_path / "job")
    assert metrics.error_breakdown == {"pipe_closed": 1}
    [trial] = bf.load_job(tmp_path / "job").trials
    assert (trial.result.error_category, trial.execution) == ("pipe_closed", "errored")


# Who owns an attempt. The Evaluation's retry loop owns each attempt it runs
# and judges the idle timeout it keeps; every other rollout is final as it
# ends, so it judges a zero-activity idle timeout when it writes it (the
# lead's ruling, 2026-11-14: the default is "not owned").


def _stub_lifecycle(monkeypatch, attempt: Attempt, jobs: Path) -> list[RolloutConfig]:
    """``Rollout.run`` as it ends when the idle watchdog stops the agent: the
    entry point's own rollout writes its records with its own writer."""
    seen: list[RolloutConfig] = []

    async def run(self):
        seen.append(self._config)
        folder = jobs / "job" / f"{self._config.task_path.name}__{len(seen):08x}"
        return _finish(self, folder, attempt)

    monkeypatch.setattr(Rollout, "run", run)
    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    return seen


def _by_hand(tmp_path: Path, attempt: Attempt, monkeypatch) -> RolloutResult:
    """A rollout built with no retry loop, as a caller builds one."""
    rollout = Rollout(
        RolloutConfig(task_path=_task(tmp_path), agent=AGENT, model=MODEL)
    )
    assert rollout._config.owned_by_retry_loop is False
    return _finish(rollout, tmp_path / "jobs" / "job" / "t__00000001", attempt)


def _bf_run(tmp_path: Path, attempt: Attempt, monkeypatch) -> RolloutResult:
    task = _task(tmp_path)
    seen = _stub_lifecycle(monkeypatch, attempt, tmp_path / "jobs")
    result = asyncio.run(
        bf.run(RolloutConfig(task_path=task, agent=AGENT, model=MODEL))
    )
    assert [c.owned_by_retry_loop for c in seen] == [False]
    return result


def _run_batch(tmp_path: Path, attempt: Attempt, monkeypatch) -> RolloutResult:
    task = _task(tmp_path)
    seen = _stub_lifecycle(monkeypatch, attempt, tmp_path / "jobs")
    [result] = bf.run_batch([RolloutConfig(task_path=task, agent=AGENT, model=MODEL)])
    assert [c.owned_by_retry_loop for c in seen] == [False]
    return result


def _read_trial(folder: Path) -> dict[str, tuple[Any, ...]]:
    """The trial in *folder* as each reader reads it."""
    recorded = _result(folder)
    alone = bf.load_trial(folder)
    loaded = bf.load_job(folder.parent)
    [trial] = loaded.trials
    [exported] = job_export(loaded).trials
    listed = CliRunner().invoke(app, ["results", str(folder.parent), "--json"])
    assert listed.exit_code == 0, listed.output
    [shown] = json.loads(listed.output)["trials"]
    [task] = collect_metrics(folder.parent).tasks
    return {
        "result.json": (recorded["error_category"], _reward(recorded["rewards"])),
        "load_trial": (alone.result.error_category, alone.reward, alone.execution),
        "load_job": (trial.result.error_category, trial.reward, trial.execution),
        "export": (exported.error_category, exported.reward, exported.execution),
        "bench results": (shown["error_category"], shown["reward"], shown["execution"]),
        "metrics": (task.reward,),
    }


def _read_as(category: str, reward: float | None, execution: str):
    return {
        "result.json": (category, reward),
        "load_trial": (category, reward, execution),
        "load_job": (category, reward, execution),
        "export": (category, reward, execution),
        "bench results": (category, reward, execution),
        "metrics": (reward,),
    }


@pytest.mark.parametrize(
    "entry", [_by_hand, _bf_run, _run_batch], ids=["Rollout", "bf.run", "bf.run_batch"]
)
@pytest.mark.parametrize(
    ("attempt", "read_as"),
    [
        (NOTHING, _read_as("agent_integration", None, "integration_failed")),
        (A_TOOL_CALL, _read_as("idle_timeout", 0.25, "errored")),
        (AN_ANSWER, _read_as("idle_timeout", 0.25, "errored")),
    ],
    ids=["did-nothing", "a-tool-call", "the-model-answered"],
)
def test_a_rollout_no_retry_loop_owns_judges_its_idle_timeout_when_written(
    tmp_path, monkeypatch, entry, attempt, read_as
):
    """Nothing retries a rollout outside an Evaluation: a zero-activity idle
    timeout is withheld in its own records, and every reader agrees; one with
    activity stays idle_timeout, scored."""
    result = entry(tmp_path, attempt, monkeypatch)

    folder = Path(result.rollout_dir)
    assert (result.error_category, _reward(result.rewards)) == read_as["result.json"]
    assert _read_trial(folder) == read_as
    assert not (folder / "attempt-decision.json").exists()
    info = _result(folder)["integration_failure_info"]
    if attempt is NOTHING:
        assert (info["cause"], info["reward_withheld"]) == (
            "no_activity",
            {"reward": 0.25},
        )
        assert not (folder / "rewards.jsonl").exists()
    else:
        assert info is None


def test_only_the_evaluations_retry_loop_owns_an_attempt(tmp_path):
    """The flag is opt-in: a new entry point is "not owned" unless it sets it,
    and only the Evaluation's attempt config sets it to anything but False (a
    branch or checkpoint-retry child is set False)."""
    import re

    import benchflow

    assert RolloutConfig(task_path=tmp_path).owned_by_retry_loop is False
    task = _task(tmp_path)
    job = Evaluation(
        tasks_dir=tmp_path / "tasks",
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(agent=AGENT, model=MODEL),
        preflight=False,
    )
    assert job._rollout_config(task, job._config).owned_by_retry_loop is True
    src = Path(benchflow.__file__).parent
    setters = sorted(
        str(path.relative_to(src))
        for path in src.rglob("*.py")
        if re.search(r"\bowned_by_retry_loop\s*=(?!\s*False\b)", path.read_text())
    )
    assert setters == ["evaluation.py"]


def _rewrites(monkeypatch) -> list[Path]:
    """The attempt folders the run rewrites after the rollout wrote them."""
    from benchflow.rollout import _results

    rewritten: list[Path] = []
    real = _results._withhold_kept_idle_timeout

    def rewrite(folder, recorded, failure):
        rewritten.append(Path(folder))
        return real(folder, recorded, failure)

    monkeypatch.setattr(_results, "_withhold_kept_idle_timeout", rewrite)
    return rewritten


def test_an_evaluation_with_no_retries_judges_its_attempt_when_written(
    tmp_path, monkeypatch
):
    """Retries set to 0: the loop's one try is its last, so the rollout judges
    the zero-activity idle timeout when it writes it (nothing is rewritten),
    the record marks it final, and every reader reads it withheld."""
    rewritten = _rewrites(monkeypatch)
    flags: list[bool] = []
    job_dir = _run(tmp_path, NOTHING, flags=flags)

    [kept] = _attempts(job_dir)
    assert flags == [False]
    assert decision_of(kept) == FINAL
    assert (
        json.loads((job_dir / "evaluation.json").read_text())["config"]["retry"][
            "max_retries"
        ]
        == 0
    )
    assert _views(job_dir) == WITHHELD
    assert _result(kept)["integration_failure_info"]["cause"] == "no_activity"
    assert rewritten == []


@pytest.mark.parametrize(
    ("retry", "flags"),
    [
        (RetryConfig(max_retries=1, min_wait_sec=0), [True, False]),
        (RetryConfig(max_retries=2, retry_on_idle_timeout=False), [False]),
    ],
    ids=["its-last-try", "a-policy-that-never-retries-it"],
)
def test_the_loop_judges_on_write_every_attempt_it_will_not_retry(
    tmp_path, monkeypatch, retry, flags
):
    """The loop knows before an attempt whether it can retry an idle timeout:
    when it cannot, the attempt is final as it ends, judged when the rollout
    writes it, once."""
    rewritten = _rewrites(monkeypatch)
    seen: list[bool] = []
    job_dir = _run(tmp_path, *([NOTHING] * len(flags)), flags=seen, retry=retry)

    assert seen == flags
    assert _views(job_dir) == WITHHELD
    assert rewritten == []


def test_an_attempt_the_loop_keeps_early_is_judged_when_kept(tmp_path, monkeypatch):
    """The budget keeps an attempt the loop could have retried: it is judged
    when the loop keeps it (the one rewrite left), and every reader agrees."""
    rewritten = _rewrites(monkeypatch)
    job_dir, result = _kept_early(tmp_path, NOTHING)

    [kept] = _attempts(job_dir)
    assert decision_of(kept) == FINAL
    assert rewritten == [kept]
    assert (result.error_category, result.rewards) == ("agent_integration", None)
    assert _read_trial(kept) == _read_as(
        "agent_integration", None, "integration_failed"
    )


# The rewrite itself (an attempt the loop kept early): what it changes and
# what it must not.


def _owned_attempt(
    tmp_path: Path,
    attempt: Attempt,
    *,
    log: str | None = None,
    owned: bool = True,
    partial: bool = False,
    scored: bool = False,
) -> tuple[Path, RolloutResult]:
    """An attempt the retry loop owns (with *owned*), written by the rollout,
    with the cache and the cost counted and an effort asked for, (*log*) an
    agent log, and (*scored*) a scoring block and a checkpoint-retry fork."""
    task = _task(tmp_path)
    config = RolloutConfig(
        task_path=task,
        agent=AGENT,
        model=MODEL,
        reasoning_effort="high",
        owned_by_retry_loop=owned,
    )
    folder = tmp_path / "jobs" / "job" / "t__00000001"
    usage = {"n_cache_read_tokens": 300, "total_tokens": 1500, "cost_usd": 0.0123}
    result = _finish(
        Rollout(config),
        folder,
        attempt,
        usage=usage,
        partial=partial,
        setup=_scored_and_branched if scored else None,
    )
    assert result.error_category == ("idle_timeout" if owned else "agent_integration")
    if log is not None:
        (folder / "agent" / "claude-code.txt").write_text(log)
    return folder, result


def _scored_and_branched(rollout: Rollout) -> None:
    """A trial with a review's scoring block and a checkpoint retry's fork,
    as result.json's ``scoring`` and ``branches`` record them."""
    from benchflow.review.outcome import ScoringResult

    rollout._scoring = ScoringResult(
        status="complete",
        passed=False,
        tests_pass=False,
        all_blockers_pass=True,
        verifier_reward=0.25,
        rubric_reward=0.5,
        reviewer_run="t__00000001__review",
    )
    rollout._branch_forks = [
        {
            "id": "f1",
            "status": "done",
            "value": "on-failure",
            "parent_node": "n1",
            "parent_restore": "restored",
            "rollout": "n5",
            "cost": None,
            "children": [
                {
                    "index": 0,
                    "node_id": "n5",
                    "intervention": {"label": "retry from checkpoint"},
                    "status": "scored",
                    "reward": 1.0,
                    "reward_source": "verifier",
                    "artifacts": {"path": "branches/f1/children/n5"},
                    "usage": {"n_output_tokens": 80},
                    "timing_sec": {"agent_execution": 3.0},
                }
            ],
        }
    ]


def _snapshot(folder: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(folder)): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def test_the_judgement_leaves_atif_json_as_the_rollout_wrote_it(tmp_path):
    """atif.json carries no reward and no ending: the rewrite leaves it byte
    for byte with every metric the writer gave it (the prompt counting the
    cache, the cost, the effort), and rewrites only the reward-bearing
    trainer files."""
    folder, result = _owned_attempt(tmp_path, NOTHING)
    before = _snapshot(folder)
    atif = json.loads(before["trainer/atif.json"])
    assert atif["final_metrics"]["total_prompt_tokens"] == 1500

    assert judge_kept_idle_timeout(result) is True

    after = _snapshot(folder)
    assert after["trainer/atif.json"] == before["trainer/atif.json"]
    assert after["trainer/verifiers.jsonl"] != before["trainer/verifiers.jsonl"]
    assert after["trainer/adp.jsonl"] != before["trainer/adp.jsonl"]
    assert "rewards.jsonl" in before and "rewards.jsonl" not in after


def test_a_failed_result_write_leaves_the_attempt_as_written(tmp_path, monkeypatch):
    """result.json is rewritten first: when it cannot be (a full disk), no
    other file has changed and the attempt reads as it was written."""
    from benchflow.rollout import _results

    folder, result = _owned_attempt(tmp_path, NOTHING)
    before = _snapshot(folder)
    real = _results.write_json_atomic

    def full_disk(path, data):
        if Path(path).name == "result.json":
            raise OSError(28, "No space left on device")
        return real(path, data)

    monkeypatch.setattr(_results, "write_json_atomic", full_disk)

    assert judge_kept_idle_timeout(result) is False
    assert _snapshot(folder) == before
    assert (result.error_category, result.rewards) == ("idle_timeout", {"reward": 0.25})


@pytest.mark.parametrize(
    ("partial", "scored"),
    [(False, False), (True, False), (False, True)],
    ids=["whole", "salvaged", "scored-and-branched"],
)
def test_the_rewrite_ends_with_the_records_a_judged_write_has(
    tmp_path, partial, scored
):
    """An attempt kept early is rewritten to the records the rollout writes
    when it judges the same attempt as it ends: result.json, its row and
    every trainer file, timestamps aside. Rebuilding them from result.json
    drifts each time the writer passes something new (review R187 M1, the
    rc's verifiers.jsonl completion and truncation)."""
    kept, result = _owned_attempt(
        tmp_path / "kept", NOTHING, partial=partial, scored=scored
    )
    assert judge_kept_idle_timeout(result) is True
    written, _ = _owned_attempt(
        tmp_path / "written", NOTHING, owned=False, partial=partial, scored=scored
    )
    if scored:
        recorded = _result(kept)
        assert recorded["scoring"]["status"] == "complete"
        assert recorded["branches"]["children"] == 1

    def records(folder: Path) -> dict[str, Any]:
        root = str(folder.parents[2])
        out: dict[str, Any] = {}
        for name in (
            "result.json",
            "results.jsonl",
            "trainer/verifiers.jsonl",
            "trainer/adp.jsonl",
            "trainer/atif.json",
        ):
            text = (folder / name).read_text().replace(root, "<root>")
            out[name] = (
                [json.loads(line) for line in text.splitlines() if line]
                if name.endswith(".jsonl")
                else json.loads(text)
            )
        for key in ("started_at", "finished_at"):
            out["result.json"].pop(key)
        return out

    assert records(kept) == records(written)
    assert not (kept / "rewards.jsonl").exists()
    assert not (written / "rewards.jsonl").exists()


# Made up, shaped like an Anthropic key.
_FAKE_KEY = "sk-ant-api03-" + "Zq9FAKEvalue" * 8 + "AA"


def test_the_rewrite_masks_what_it_records(tmp_path):
    """The finding quotes the agent's log: the error and the record it writes
    are masked as the rollout masks them."""
    folder, result = _owned_attempt(
        tmp_path, NOTHING, log=f"API Error: 401 invalid x-api-key {_FAKE_KEY}\n"
    )

    assert judge_kept_idle_timeout(result) is True

    recorded = _result(folder)
    assert recorded["integration_failure_info"]["cause"] == "agent_auth"
    assert "***REDACTED***" in recorded["error"]
    assert "***REDACTED***" in recorded["integration_failure_info"]["evidence"]
    leaked = [
        name
        for name, data in _snapshot(folder).items()
        if not name.startswith("agent/") and _FAKE_KEY.encode() in data
    ]
    assert leaked == []


# Each attempt's trial spans reach the collector once.


def _sends(monkeypatch) -> list[tuple[Any, ...]]:
    from benchflow.trajectories import otel_spans

    sent: list[tuple[Any, ...]] = []

    def send(payload, url, headers):
        span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        attributes = {
            a["key"]: next(iter(a["value"].values()), None) for a in span["attributes"]
        }
        sent.append(
            (
                attributes["benchflow.trial.name"],
                attributes.get("benchflow.error.category"),
                attributes.get("benchflow.reward"),
            )
        )
        return 200

    monkeypatch.setattr(
        otel_spans, "_endpoint", lambda: ("http://collector.invalid/v1/traces", {})
    )
    monkeypatch.setattr(otel_spans, "send_spans", send)
    return sent


def test_each_attempt_sends_its_spans_once_with_its_ending(tmp_path, monkeypatch):
    sent = _sends(monkeypatch)
    _run(tmp_path / "a", NOTHING, NOTHING)
    # The retried attempt when the loop retries it; the last try when written.
    assert sent == [
        ("t__00000000", "idle_timeout", 0.25),
        ("t__00000001", "agent_integration", None),
    ]
    sent.clear()
    _kept_early(tmp_path / "b", NOTHING)
    # Kept early: once, after the judgement.
    assert sent == [("t__00000000", "agent_integration", None)]


# bench train stream: an attempt of a running job waits for the run's decision.


def _running(job_dir: Path, max_retries: int) -> Path:
    """Mark *job_dir* as a running Evaluation job (its lock held by this
    process) with that retry policy; the lock file."""
    import os
    import socket

    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "evaluation.json").write_text(
        json.dumps(
            {"schema_version": 1, "config": {"retry": {"max_retries": max_retries}}}
        )
    )
    lock = job_dir / ".evaluation.lock"
    lock.write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname()}))
    return lock


def test_the_stream_reads_a_kept_attempt_once_the_run_has_decided_it(
    tmp_path, monkeypatch
):
    """Between the rollout's write and the loop's decision the attempt still
    says idle_timeout 0.25; a trainer streaming the job then must not get it,
    since the stream never reads a rollout twice."""
    from benchflow.rollout import _results
    from benchflow.trajectories.rollout_stream import stream_rollouts

    inside: list[tuple[str, Any]] = []
    real = _results.judge_kept_idle_timeout

    def judge(result):
        job_dir = Path(result.rollout_dir).parent
        inside.extend(
            (r.rollout, r.reward) for r in stream_rollouts(job_dir, follow=False)
        )
        return real(result)

    monkeypatch.setattr(_results, "judge_kept_idle_timeout", judge)
    queue = [NOTHING]
    job = _job(tmp_path, queue, RetryConfig(max_retries=1, min_wait_sec=0))
    job._budget_guard = _NoRoomForARetry()
    job_dir = job._jobs_dir / job._job_name
    lock = _running(job_dir, max_retries=1)
    asyncio.run(job._run_task(tmp_path / "tasks" / "t"))

    assert inside == []
    lock.unlink()
    assert [(r.rollout, r.reward) for r in stream_rollouts(job_dir, follow=False)] == [
        ("t__00000000", None)
    ]


def test_an_undecided_attempt_is_read_once_the_job_has_ended(tmp_path):
    from benchflow.trajectories.rollout_stream import stream_rollouts

    job_dir = tmp_path / "job"
    lock = _running(job_dir, max_retries=1)
    _stored_trial(
        job_dir,
        {
            "error": "Agent idle for 600s with no new tool call (2 tool calls so far)",
            "error_category": "idle_timeout",
            "n_tool_calls": 2,
        },
    )
    assert list(stream_rollouts(job_dir, follow=False)) == []
    lock.unlink()
    assert [r.reward for r in stream_rollouts(job_dir, follow=False)] == [0.25]


# Review R187B (round 3).


def test_a_checkpoint_retry_child_is_never_one_of_the_loops_tries(
    tmp_path, monkeypatch
):
    """M1. A child (``--retry-from-checkpoint``, a branch) copies its parent's
    config, but the loop neither retries nor decides it: it sends its own
    spans when it writes them, while its parent's wait for the decision."""
    from benchflow.rollout_branch import _make_sub_rollout

    sent = _sends(monkeypatch)
    parent = Rollout(
        RolloutConfig(
            task_path=_task(tmp_path),
            agent=AGENT,
            model=MODEL,
            owned_by_retry_loop=True,
        )
    )
    node = parent._tree.attach(parent._tree.root)
    child = _make_sub_rollout(parent, tmp_path / "fork", node)
    assert child._config.owned_by_retry_loop is False

    from benchflow.trajectories.otel_spans import (
        defer_trial_spans,
        stop_deferring_trial_spans,
    )

    pending: list[Path] = []
    token = defer_trial_spans(pending)
    try:
        _finish(child, tmp_path / "fork" / "children" / node.id, SOLVED)
    finally:
        stop_deferring_trial_spans(token)
    assert pending == []
    assert sent == [(node.id, None, 1.0)]


def test_an_owned_try_and_its_checkpoint_retry_child_each_send_once(
    tmp_path, monkeypatch
):
    """M1 through the retry loop: a retried try, the kept one and the
    checkpoint-retry child each try ran reach the collector exactly once."""
    from benchflow.rollout_branch import _make_sub_rollout

    sent = _sends(monkeypatch)
    queue = [NOTHING, NOTHING]
    job = _job(tmp_path, [], RetryConfig(max_retries=1, min_wait_sec=0))
    children: list[str] = []

    async def run_single(task_path, _cfg):
        folder = job._jobs_dir / job._job_name
        trial = folder / f"t__{len(children):08x}"
        parent = Rollout(job._rollout_config(task_path, job._config))
        result = _finish(parent, trial, queue.pop(0))
        node = parent._tree.attach(parent._tree.root)
        child = _make_sub_rollout(parent, trial / "branches" / "f1", node)
        _finish(child, trial / "branches" / "f1" / "children" / node.id, SOLVED)
        children.append(node.id)
        return result

    job._run_single_task = run_single
    asyncio.run(job.run())

    assert queue == []
    assert sorted(sent, key=repr) == sorted(
        [
            ("t__00000000", "idle_timeout", 0.25),
            ("t__00000001", "agent_integration", None),
            *[(name, None, 1.0) for name in children],
        ],
        key=repr,
    )


def test_a_try_stopped_before_its_decision_sends_what_it_wrote(tmp_path, monkeypatch):
    """L1. A try the loop could retry leaves its spans for the decision; one
    stopped before it has one (Ctrl-C, a kill, an error after its result was
    written) sends them as it stops."""
    sent = _sends(monkeypatch)
    job = _job(tmp_path, [NOTHING], RetryConfig(max_retries=1, min_wait_sec=0))
    write = job._run_single_task

    async def write_then_stop(task_path, cfg):
        await write(task_path, cfg)
        raise asyncio.CancelledError

    job._run_single_task = write_then_stop
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(job._run_task(tmp_path / "tasks" / "t"))
    assert sent == [("t__00000000", "idle_timeout", 0.25)]
    assert not (job._jobs_dir / job._job_name / "t__00000000" / FINAL_RECORD).exists()


FINAL_RECORD = "attempt-decision.json"


def _break(step: str, monkeypatch) -> None:
    """Make one step of the rewrite after the result.json commit fail."""
    from benchflow.trajectories import export, otel_spans, results

    def fail(*args, **kwargs):
        raise OSError(28, "No space left on device")

    if step == "row":
        monkeypatch.setattr(results, "write_rollout_results_jsonl", fail)
    elif step == "trainer":
        monkeypatch.setattr(export, "write_rollout_verifiers_jsonl", fail)
    elif step == "spans":
        monkeypatch.setattr(otel_spans, "trial_spans", fail)
    elif step == "rewards":
        real = Path.unlink

        def unlink(self, missing_ok=False):
            if self.name == "rewards.jsonl":
                raise PermissionError(13, "Permission denied")
            return real(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", unlink)


def _nothing_says_otherwise(folder: Path) -> None:
    """Every record of *folder* derived from result.json is either gone or
    says what result.json says: no reward."""
    row = folder / "results.jsonl"
    if row.exists():
        assert json.loads(row.read_text())["reward"] is None
    verifiers = folder / "trainer" / "verifiers.jsonl"
    if verifiers.exists():
        assert json.loads(verifiers.read_text().splitlines()[0])["reward"] is None
    adp = folder / "trainer" / "adp.jsonl"
    if adp.exists():
        details = json.loads(adp.read_text().splitlines()[0])["details"]
        assert details.get("reward") is None
    rewards = folder / "rewards.jsonl"
    assert not rewards.exists() or rewards.read_text() == ""
    if (folder / "trajectory" / "otel-spans.json").exists():
        span = _trial_span(folder)
        assert span["benchflow.error.category"] == "agent_integration"
        assert "benchflow.reward" not in span


@pytest.mark.parametrize("step", ["row", "trainer", "rewards", "spans"])
def test_a_failure_after_the_commit_leaves_no_record_saying_otherwise(
    tmp_path, monkeypatch, step
):
    """L2. result.json is the commit point: once it is written the trial is
    judged, the result handed back says so, and a derived file that cannot be
    rewritten is removed (or emptied) rather than left with the reward."""
    folder, result = _owned_attempt(tmp_path, NOTHING)
    _break(step, monkeypatch)

    assert judge_kept_idle_timeout(result) is True

    assert (result.error_category, result.rewards) == ("agent_integration", None)
    assert result.integration_failure_info["cause"] == "no_activity"
    assert _read_trial(folder) == _read_as(
        "agent_integration", None, "integration_failed"
    )
    _nothing_says_otherwise(folder)


def test_a_failed_row_after_the_commit_still_agrees_with_the_summary(
    tmp_path, monkeypatch
):
    """L2 through a real budget keep: summary.json follows result.json."""
    _task(tmp_path)
    queue = [NOTHING]
    job = Evaluation(
        tasks_dir=tmp_path / "tasks",
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent=AGENT,
            model=MODEL,
            retry=RetryConfig(max_retries=1, min_wait_sec=0),
            budget={"max_rollouts": 1},
        ),
        preflight=False,
    )

    async def run_single(task_path, _cfg):
        written = _write_attempt(job, task_path, queue.pop(0))
        # The rollout wrote its row (0.25); the rewrite of it is what fails.
        assert (
            json.loads((written.rollout_dir / "results.jsonl").read_text())["reward"]
            == 0.25
        )
        _break("row", monkeypatch)
        return written

    async def no_sweep():
        return None

    job._run_single_task = run_single
    job._sweep_docker = no_sweep
    asyncio.run(job.run())
    job_dir = job._jobs_dir / job._job_name

    [kept] = _attempts(job_dir)
    assert decision_of(kept) == FINAL
    summary = json.loads((job_dir / "summary.json").read_text())
    assert (summary["errored"], summary["error_categories"]) == (
        1,
        {"agent_integration": 1},
    )
    assert _read_trial(kept) == _read_as(
        "agent_integration", None, "integration_failed"
    )
    _nothing_says_otherwise(kept)


def test_the_stream_reads_undecided_attempts_at_its_timeout_under_a_foreign_lock(
    tmp_path,
):
    """L3. A lock the stream cannot check (another host's) may outlive the
    job; at its timeout the stream reads the attempts the run never decided
    before it ends."""
    from benchflow.trajectories.rollout_stream import StreamTimeout, stream_rollouts

    job_dir = tmp_path / "job"
    lock = _running(job_dir, max_retries=1)
    lock.write_text(json.dumps({"pid": 4242, "host": "another-host.invalid"}))
    _stored_trial(
        job_dir,
        {
            "error": "Agent idle for 600s with no new tool call (2 tool calls so far)",
            "error_category": "idle_timeout",
            "n_tool_calls": 2,
        },
    )
    got: list[Any] = []
    with pytest.raises(StreamTimeout):
        for record in stream_rollouts(job_dir, poll_interval=0.05, timeout=0.3):
            got.append(record.reward)
    assert got == [0.25]


def test_the_sdk_hands_back_the_finding_of_a_withheld_trial(tmp_path, monkeypatch):
    """L4. bf.run, run_batch (its records, JSONL and CSV), Results'
    denominators and RolloutResult.load all carry the finding of a trial the
    writer withheld; a trial the loop kept early does too."""
    import csv

    task = _task(tmp_path)
    _stub_lifecycle(monkeypatch, NOTHING, tmp_path / "jobs")
    config = RolloutConfig(task_path=task, agent=AGENT, model=MODEL)

    result = asyncio.run(bf.run(config))
    info = result.integration_failure_info
    assert (info["cause"], info["reward_withheld"]) == ("no_activity", {"reward": 0.25})
    assert RolloutResult.load(result.rollout_dir).integration_failure_info == info

    results = bf.run_batch([config])
    [record] = results.to_records()
    assert (
        record["reward"],
        record["error_category"],
        record["integration_failure_info"]["cause"],
        record["execution"],
    ) == (None, "agent_integration", "no_activity", "integration_failed")
    [row] = results.to_jsonl(tmp_path / "rows.jsonl").read_text().splitlines()
    assert json.loads(row)["integration_failure_info"]["cause"] == "no_activity"
    with results.to_csv(tmp_path / "rows.csv").open() as handle:
        [cell] = [r["integration_failure_info"] for r in csv.DictReader(handle)]
    assert json.loads(cell)["cause"] == "no_activity"
    assert (results.denominators().integration_failures, results.n_scored) == (1, 0)

    _, kept = _kept_early(tmp_path / "early", NOTHING)
    assert kept.integration_failure_info["cause"] == "no_activity"


def test_a_warm_pool_rollout_takes_the_loops_last_try(tmp_path, monkeypatch):
    """The warm pool prepares a rollout before its attempt, with the config
    of a try the loop may retry; run as the loop's last try it is not owned,
    so it judges its idle timeout when it writes it."""
    from benchflow import evaluation

    seen = _stub_lifecycle(monkeypatch, NOTHING, tmp_path / "jobs")
    task = _task(tmp_path)
    job = Evaluation(
        tasks_dir=tmp_path / "tasks",
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent=AGENT, model=MODEL, retry=RetryConfig(max_retries=0)
        ),
        preflight=False,
    )
    prepared = Rollout(job._rollout_config(task, job._config))
    assert prepared._config.owned_by_retry_loop is True
    token = evaluation._WARM_ROLLOUT.set(prepared)
    try:
        result = asyncio.run(job._run_task(task))
    finally:
        evaluation._WARM_ROLLOUT.reset(token)

    assert [c.owned_by_retry_loop for c in seen] == [False]
    assert (result.error_category, result.rewards) == ("agent_integration", None)


def test_evaluation_stream_and_on_result_hand_back_the_finding(tmp_path):
    """Evaluation.stream() (built on on_result) yields a withheld trial with
    its finding: one judged when written (the loop's last try) and one the
    loop kept early (a real budget with no room for the retry)."""

    async def streamed(job: Evaluation) -> list[RolloutResult]:
        return [result async for _, result in job.stream()]

    last = _job(tmp_path / "last", [NOTHING], RetryConfig(max_retries=0))
    [written] = asyncio.run(streamed(last))
    _task(tmp_path / "early")
    queue = [NOTHING]
    early = Evaluation(
        tasks_dir=tmp_path / "early" / "tasks",
        jobs_dir=tmp_path / "early" / "jobs",
        config=EvaluationConfig(
            agent=AGENT,
            model=MODEL,
            retry=RetryConfig(max_retries=1, min_wait_sec=0),
            budget={"max_rollouts": 1},
        ),
        preflight=False,
    )

    async def run_single(task_path, _cfg):
        return _write_attempt(early, task_path, queue.pop(0))

    async def no_sweep():
        return None

    early._run_single_task = run_single
    early._sweep_docker = no_sweep
    [kept] = asyncio.run(streamed(early))

    for result in (written, kept):
        assert (
            result.reward,
            result.error_category,
            result.integration_failure_info["cause"],
        ) == (None, "agent_integration", "no_activity")
