"""summary.json says what a job spent, retries included.

Before: summary.json's ``total_cost_usd`` and token totals counted each
task's final attempt only, while ``budget.spent`` in the same file counted
every attempt, so a job whose task failed twice on infrastructure and then
passed reported two different costs and no figure named which was which.
The ``total_*`` keys still count the final attempts (the trials the score
reads); ``spend_all_attempts`` counts every attempt by the budget's own
per-attempt rule, so it agrees with ``budget.spent`` and with
``bf.load_job``'s cost line.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow import Budget
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.models import RolloutResult

FLAKY = "ACP error -32603: Internal error"
# task-0 errors twice on infrastructure, then passes; task-1 passes at once.
PLAN = {
    "task-0": [
        {"error": FLAKY, "tokens": 40_000, "cost": 0.5},
        {"error": FLAKY, "tokens": 20_000, "cost": 0.25},
        {"tokens": 10_000, "cost": 0.125},
    ],
    "task-1": [{"tokens": 8_000, "cost": 0.0625}],
}


def _tasks(tmp_path: Path) -> Path:
    tasks_dir = tmp_path / "tasks"
    for name in PLAN:
        (tasks_dir / name).mkdir(parents=True)
        (tasks_dir / name / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    return tasks_dir


def _agent_result(tokens: int, cost: float) -> dict:
    """agent_result as rollout/_results.py writes it for a proxied trial."""
    return {
        "n_tool_calls": 4,
        "n_skill_invocations": 0,
        "n_prompts": 1,
        "n_input_tokens": tokens // 2,
        "n_output_tokens": tokens // 4,
        "n_cache_read_tokens": tokens // 4,
        "n_cache_write_tokens": 0,
        "n_cache_write_5m_tokens": 0,
        "n_cache_write_1h_tokens": 0,
        "n_cache_creation_tokens": 0,
        "total_tokens": tokens,
        "cost_usd": cost,
        "usage_source": "proxy",
        "cost_source": "list_price_estimate",
        "price_table_date": None,
        "price_source": "litellm",
        "usage_source_legacy": "provider_response",
    }


def _runner(job: Evaluation):
    seen: dict[str, int] = {}

    async def fake_run(task_path, _cfg):
        name = task_path.name
        n = seen[name] = seen.get(name, 0) + 1
        outcome = PLAN[name][min(n, len(PLAN[name])) - 1]
        await asyncio.sleep(0)
        rollout_dir = job._jobs_dir / job._job_name / f"{name}__attempt{n}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        error = outcome.get("error")
        rewards = None if error else {"reward": 1.0}
        usage = _agent_result(outcome["tokens"], outcome["cost"])
        (rollout_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": name,
                    "rollout_name": rollout_dir.name,
                    "rewards": rewards,
                    "agent": "claude-agent-acp",
                    "model": "claude-haiku-5-5",
                    "n_tool_calls": 4,
                    "agent_result": usage,
                    "judge_usage": None,
                    "error": error,
                    "error_category": "acp_error" if error else None,
                    "verifier_error": None,
                }
            )
        )
        return RolloutResult(
            task_name=name,
            rollout_name=rollout_dir.name,
            rewards=rewards,
            agent="claude-agent-acp",
            model="claude-haiku-5-5",
            n_tool_calls=4,
            n_input_tokens=usage["n_input_tokens"],
            n_output_tokens=usage["n_output_tokens"],
            n_cache_read_tokens=usage["n_cache_read_tokens"],
            n_cache_creation_tokens=0,
            total_tokens=usage["total_tokens"],
            cost_usd=usage["cost_usd"],
            usage_source="proxy",
            cost_source="list_price_estimate",
            price_source="litellm",
            error=error,
            error_category="acp_error" if error else None,
            rollout_dir=rollout_dir,
        )

    return fake_run


def _job(tmp_path: Path, budget: Budget | None) -> Evaluation:
    return Evaluation(
        tasks_dir=_tasks(tmp_path),
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent="claude-agent-acp",
            model="claude-haiku-5-5",
            concurrency=1,
            retry=RetryConfig(max_retries=2, min_wait_sec=0.0),
        ),
        budget=budget,
        preflight=False,
    )


def _summary(job: Evaluation) -> dict:
    return json.loads((job._jobs_dir / job._job_name / "summary.json").read_text())


@pytest.mark.asyncio
async def test_summary_has_every_attempts_spend_and_it_agrees_with_the_budget(
    tmp_path,
):
    job = _job(tmp_path, Budget(max_cost_usd=100.0))
    job._run_single_task = AsyncMock(side_effect=_runner(job))
    await job.run()
    assert job._run_single_task.await_count == 4
    summary = _summary(job)

    # The final attempts: the trials the score reads.
    assert summary["total_cost_usd"] == pytest.approx(0.125 + 0.0625)
    assert summary["total_tokens"] == 18_000

    # Every attempt, retries included: what the job paid.
    spend = summary["spend_all_attempts"]
    assert spend["attempts"] == 4
    assert spend["cost_usd"] == pytest.approx(0.5 + 0.25 + 0.125 + 0.0625)
    assert spend["tokens"] == 78_000
    assert spend["judge_cost_usd"] is None and spend["judge_tokens"] is None
    assert spend["attempts_without_cost"] == 0

    spent = summary["budget"]["spent"]
    assert spent["cost_usd"] == pytest.approx(spend["cost_usd"])
    assert spent["tokens"] == spend["tokens"]


@pytest.mark.asyncio
async def test_spend_all_attempts_without_a_budget_and_after_a_resume(tmp_path):
    from benchflow.jobs import load_job

    job = _job(tmp_path, None)
    job._run_single_task = AsyncMock(side_effect=_runner(job))
    await job.run()
    first = _summary(job)["spend_all_attempts"]
    assert first["cost_usd"] == pytest.approx(0.9375)
    assert "budget" not in _summary(job)

    # print(bf.load_job(...)) sums every attempt too.
    loaded = load_job(job._jobs_dir / job._job_name)
    costs = [
        attempt.cost_usd
        for trial in loaded.trials
        for attempt in trial.attempts
        if attempt.cost_usd is not None
    ]
    assert sum(costs) == pytest.approx(first["cost_usd"])

    resumed = Evaluation.resume(job._jobs_dir / job._job_name)
    resumed._preflight = False
    resumed._run_single_task = AsyncMock(side_effect=_runner(resumed))
    await resumed.run()
    assert resumed._run_single_task.await_count == 0
    assert _summary(resumed)["spend_all_attempts"] == first


def test_an_attempt_with_no_cost_is_unknown_not_zero():
    from benchflow._utils.evaluation_results import attempt_spend_summary

    rows = [
        {"task_name": "a", "agent_result": _agent_result(1000, 0.01)},
        {
            "task_name": "b",
            "agent_result": {
                **_agent_result(500, 0.0),
                "cost_usd": None,
                "cost_source": None,
            },
        },
        # A stopped attempt's receipt (budget-cancelled.json): one that
        # recorded what it spent, and an older one that did not.
        {
            "_budget_cancelled": True,
            "timing": {"total": 3.0},
            "agent_result": {"total_tokens": 200, "cost_usd": 0.002},
        },
        {"_budget_cancelled": True, "timing": {"total": 3.0}},
    ]
    spend = attempt_spend_summary(rows)["spend_all_attempts"]
    assert spend["attempts"] == 2
    assert spend["stopped_attempts"] == 2
    assert spend["stopped_attempts_without_usage"] == 1
    assert spend["cost_usd"] == pytest.approx(0.012)
    assert spend["tokens"] == 1700
    assert spend["attempts_without_cost"] == 1
    nothing = attempt_spend_summary([])["spend_all_attempts"]
    assert nothing["cost_usd"] is None and nothing["tokens"] is None


@pytest.mark.asyncio
async def test_every_attempt_counts_whatever_the_kept_attempt_picker_drops(tmp_path):
    """The attempt-decision records (the R2 stack's attempt-decision.json)
    tell the kept-attempt picker to skip an attempt the run retried; what
    that attempt spent is still the job's. Written in the stack's record
    format, so this holds once the picker reads them."""
    from benchflow.evaluation import _spend_all_attempts

    job = _job(tmp_path, None)
    job._run_single_task = AsyncMock(side_effect=_runner(job))
    await job.run()
    job_dir = job._jobs_dir / job._job_name
    for n, decision in ((1, "retried"), (2, "retried"), (3, "final")):
        (job_dir / f"task-0__attempt{n}" / "attempt-decision.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "decision": decision,
                    "task": "task-0",
                    "attempt": n,
                    "sequence": n,
                    "decided_at": "2026-11-13T09:00:00+00:00",
                }
            )
        )
    spend = _spend_all_attempts(job_dir)["spend_all_attempts"]
    assert spend["attempts"] == 4
    assert spend["cost_usd"] == pytest.approx(0.9375)


def _metrics_text(job_dir: Path, *args: str) -> str:
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    result = CliRunner().invoke(app, ["eval", "metrics", str(job_dir), *args])
    assert result.exit_code == 0, result.output
    return result.output


def _rows(text: str) -> str:
    return " ".join(text.replace("│", " ").split())


@pytest.mark.asyncio
async def test_bench_eval_metrics_shows_the_kept_trials_cost_and_every_attempts_spend(
    tmp_path,
):
    """The lead's ruling (2026-11-13): two labelled Cost rows, never one
    ambiguous number, with summary.json's figures and names. task-0 ran
    three attempts (two retried), so the two rows differ."""
    job = _job(tmp_path, None)
    job._run_single_task = AsyncMock(side_effect=_runner(job))
    await job.run()
    job_dir = job._jobs_dir / job._job_name
    summary = _summary(job)

    text = _rows(_metrics_text(job_dir))
    assert (
        "Cost (total_cost_usd) kept trials: $0.1875 over 2 trials "
        "(list_price_estimate 2)"
    ) in text
    assert (
        "Spend (spend_all_attempts) every attempt: $0.9375 over 4 attempts, "
        "retries and judges included"
    ) in text
    assert summary["total_cost_usd"] == pytest.approx(0.1875)
    assert summary["spend_all_attempts"]["cost_usd"] == pytest.approx(0.9375)

    shown = json.loads(_metrics_text(job_dir, "--json"))
    assert shown["total_cost_usd"] == pytest.approx(summary["total_cost_usd"])
    assert shown["spend_all_attempts"] == summary["spend_all_attempts"]

    # A retried attempt that recorded no cost is unknown, with its count,
    # never $0; the kept trials' row is unchanged.
    first = job_dir / "task-0__attempt1" / "result.json"
    data = json.loads(first.read_text())
    data["agent_result"] = {
        **data["agent_result"],
        "cost_usd": None,
        "cost_source": None,
    }
    first.write_text(json.dumps(data))
    text = _rows(_metrics_text(job_dir))
    assert "kept trials: $0.1875 over 2 trials" in text
    assert (
        "every attempt: $0.4375 over 4 attempts, retries and judges included; "
        "1 unknown (no cost reported)"
    ) in text


@pytest.mark.asyncio
async def test_metrics_spend_counts_attempts_no_trial_kept(tmp_path):
    """The R2 re-review's case: a run stopped during a retry. Its attempts
    were retried, none kept (the R2 stack's attempt-decision.json says so, in
    its record format; its picker then lists them as Job.retried, not as
    trials). The Spend row reads them as summary.json does, equal to the
    stopped run's spend_all_attempts, and a job of only retried attempts
    still shows its $0.75 over 2 attempts (not None, not a kept trial's
    $0.25)."""
    job = _job(tmp_path, None)
    real = _runner(job)

    async def stopped_at_the_third_try(task_path, cfg):
        if task_path.name == "task-0" and len(list(job_dir.glob("task-0__*"))) == 2:
            raise asyncio.CancelledError  # Ctrl-C during the second retry
        return await real(task_path, cfg)

    job_dir = job._jobs_dir / job._job_name
    job._run_single_task = AsyncMock(side_effect=stopped_at_the_third_try)
    with pytest.raises(asyncio.CancelledError):
        await job.run()
    for n in (1, 2):
        (job_dir / f"task-0__attempt{n}" / "attempt-decision.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "decision": "retried",
                    "task": "task-0",
                    "attempt": n,
                    "sequence": n,
                    "decided_at": "2026-11-13T09:00:00+00:00",
                }
            )
        )
    from benchflow.evaluation import _spend_all_attempts

    # The stopped run's own summary.json: the Spend row and --json say the
    # same, the retried attempts included.
    spend = _summary(job)["spend_all_attempts"]
    retried = 0.5 + 0.25
    assert spend["cost_usd"] >= retried
    text = _rows(_metrics_text(job_dir))
    assert (
        f"Spend (spend_all_attempts) every attempt: ${spend['cost_usd']:.4f} over "
        f"{spend['attempts']} attempts, retries and judges included"
    ) in text
    assert json.loads(_metrics_text(job_dir, "--json"))["spend_all_attempts"] == spend

    # Only the retried attempts left: no trial was kept, the spend shows.
    for other in job_dir.glob("task-1__*"):
        shutil.rmtree(other)
    spend = _spend_all_attempts(job_dir)["spend_all_attempts"]
    assert spend["attempts"] == 2 and spend["cost_usd"] == pytest.approx(retried)
    text = _rows(_metrics_text(job_dir))
    assert "every attempt: $0.7500 over 2 attempts" in text
    assert json.loads(_metrics_text(job_dir, "--json"))["spend_all_attempts"] == spend


def _stopped_attempt(
    job_dir: Path,
    name: str,
    task: str,
    cost: float | None,
    *,
    recorded: bool = True,
) -> None:
    """A trial stopped before it finished, as Rollout.run leaves it:
    result.json marked interrupted, with what it spent, and the receipt."""
    folder = job_dir / name
    folder.mkdir(parents=True)
    usage = {**_agent_result(1000, cost), "usage_source": "proxy"}
    (folder / "result.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "rollout_name": name,
                "rewards": None,
                "agent": "claude-agent-acp",
                "agent_result": usage,
                "error": "Interrupted: the run was stopped",
                "verifier_error": None,
                "partial_trajectory": True,
                "finished_at": "2026-11-13T09:00:00+00:00",
                "interrupted": True,
            }
        )
    )
    (folder / "budget-cancelled.json").write_text(
        json.dumps(
            {
                "version": 1,
                "reason": "interrupted",
                "timing": {"total": 1.0, "verifier_sandbox_total": 0.0},
                **(
                    {
                        "agent_result": {
                            "total_tokens": 1000,
                            "cost_usd": cost,
                            "usage_source": "proxy",
                            "cost_source": "list_price_estimate",
                        }
                    }
                    if recorded
                    else {}
                ),
            }
        )
    )


_REVIEW_PLAN = {
    "task-a": [
        {"error": FLAKY, "tokens": 40_000, "cost": 0.5},
        {"tokens": 10_000, "cost": 0.25},
    ],
    "task-b": [{"tokens": 8_000, "cost": 0.2}],
}


async def _reviewers_job_stopped(tmp_path: Path) -> tuple[Evaluation, Path]:
    """The review's job, to its stop: task-a errored on infrastructure
    ($0.50) and passed on its retry ($0.25); task-b was stopped ($0.10)."""
    tasks_dir = tmp_path / "tasks"
    for name in _REVIEW_PLAN:
        (tasks_dir / name).mkdir(parents=True)
        (tasks_dir / name / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    job = Evaluation(
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent="claude-agent-acp",
            model="claude-haiku-5-5",
            concurrency=2,
            retry=RetryConfig(max_retries=2, min_wait_sec=0.0),
        ),
        preflight=False,
    )
    job_dir = job._jobs_dir / job._job_name
    a_done = asyncio.Event()
    seen: dict[str, int] = {}

    async def first_run(task_path, _cfg):
        name = task_path.name
        if name == "task-b":
            await a_done.wait()
            _stopped_attempt(job_dir, "task-b__stop1", "task-b", 0.10)
            raise asyncio.CancelledError  # Ctrl-C while task-b ran
        n = seen[name] = seen.get(name, 0) + 1
        result = await _attempt(job, name, n, _REVIEW_PLAN[name][n - 1])
        if n == 2:
            a_done.set()
        return result

    job._run_single_task = AsyncMock(side_effect=first_run)
    with pytest.raises(asyncio.CancelledError):
        await job.run()
    return job, job_dir


@pytest.mark.asyncio
async def test_the_kept_cost_counts_a_stopped_attempt_only_while_its_task_has_none(
    tmp_path,
):
    """The review of 0a3a5fab8 (F1): the stopped run's kept total is 0.35
    (task-b has no kept attempt yet), the resumed run's 0.45 (task-b then
    passed for $0.20), never 0.55; the metrics Cost row and summary.json
    agree each time, and every attempt's spend is 1.05."""
    job, job_dir = await _reviewers_job_stopped(tmp_path)
    stopped = _summary(job)
    assert stopped["total_cost_usd"] == pytest.approx(0.35)
    text = _rows(_metrics_text(job_dir))
    assert "kept trials: $0.3500 over 2 trials" in text
    assert "1 trial stopped before an attempt was kept" in text
    shown = json.loads(_metrics_text(job_dir, "--json"))
    assert shown["total_cost_usd"] == pytest.approx(stopped["total_cost_usd"])

    resumed = Evaluation.resume(job_dir)
    resumed._preflight = False

    async def second_run(task_path, _cfg):
        name = task_path.name
        return await _attempt(resumed, name, 1, _REVIEW_PLAN[name][-1])

    resumed._run_single_task = AsyncMock(side_effect=second_run)
    await resumed.run()
    summary = _summary(resumed)
    assert summary["total_cost_usd"] == pytest.approx(0.45)
    assert summary["spend_all_attempts"]["cost_usd"] == pytest.approx(1.05)
    text = _rows(_metrics_text(job_dir))
    assert "kept trials: $0.4500 over 2 trials" in text
    assert "every attempt: $1.0500 over" in text
    shown = json.loads(_metrics_text(job_dir, "--json"))
    assert shown["total_cost_usd"] == pytest.approx(summary["total_cost_usd"])
    assert shown["spend_all_attempts"] == summary["spend_all_attempts"]


@pytest.mark.asyncio
async def test_metrics_json_takes_every_kept_total_by_one_rule(tmp_path):
    """Review 2, #3: after the stop, --json had the kept rule's
    total_cost_usd (0.35) beside total_tokens and avg_cost_per_trial_usd by
    another (10000 and 0.25; summary.json says 11000 and 0.175)."""
    job, job_dir = await _reviewers_job_stopped(tmp_path)
    stopped = _summary(job)
    shown = json.loads(_metrics_text(job_dir, "--json"))
    for key in ("total_cost_usd", "total_tokens", "avg_cost_per_trial_usd"):
        assert shown[key] == pytest.approx(stopped[key]), key


@pytest.mark.asyncio
async def test_a_trial_folder_read_alone_counts_no_other_tasks_stop(tmp_path):
    """Review 2, #4: `bench eval metrics <job>/<trial folder>` added task-b's
    stopped attempt to task-a's trial: $0.35 over 2 trials, not $0.25 over 1."""
    _, job_dir = await _reviewers_job_stopped(tmp_path)
    trial = _rows(_metrics_text(job_dir / "task-a__run2"))
    assert "kept trials: $0.2500 over 1 trial" in trial
    assert "stopped before" not in trial


async def _attempt(job: Evaluation, name: str, n: int, outcome: dict) -> RolloutResult:
    rollout_dir = job._jobs_dir / job._job_name / f"{name}__run{n}"
    if rollout_dir.exists():
        rollout_dir = rollout_dir.with_name(f"{name}__rerun{n}")
    rollout_dir.mkdir(parents=True)
    error = outcome.get("error")
    rewards = None if error else {"reward": 1.0}
    usage = _agent_result(outcome["tokens"], outcome["cost"])
    (rollout_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": name,
                "rollout_name": rollout_dir.name,
                "rewards": rewards,
                "agent": "claude-agent-acp",
                "agent_result": usage,
                "error": error,
                "error_category": "acp_error" if error else None,
                "verifier_error": None,
            }
        )
    )
    return RolloutResult(
        task_name=name,
        rollout_name=rollout_dir.name,
        rewards=rewards,
        agent="claude-agent-acp",
        n_input_tokens=usage["n_input_tokens"],
        n_output_tokens=usage["n_output_tokens"],
        n_cache_read_tokens=usage["n_cache_read_tokens"],
        total_tokens=usage["total_tokens"],
        cost_usd=usage["cost_usd"],
        usage_source="proxy",
        cost_source="list_price_estimate",
        error=error,
        error_category="acp_error" if error else None,
        rollout_dir=rollout_dir,
    )


@pytest.mark.asyncio
async def test_an_unreadable_record_makes_the_spend_unknown_not_none(tmp_path):
    """The review's F2: one unreadable budget-cancelled.json makes
    summary.json's spend_all_attempts null; metrics said "none: no attempt
    called a model" and --json gave {}. It is unknown, and says why."""
    job = _job(tmp_path, None)
    job._run_single_task = AsyncMock(side_effect=_runner(job))
    await job.run()
    job_dir = job._jobs_dir / job._job_name
    (job_dir / "task-1__attempt1" / "budget-cancelled.json").write_text("{")
    from benchflow.evaluation import _spend_all_attempts

    assert _spend_all_attempts(job_dir)["spend_all_attempts"] is None
    text = _rows(_metrics_text(job_dir))
    assert (
        "every attempt: unknown: 1 job folder whose attempt records could not be read"
        in text
    )
    assert "none: no attempt called a model" not in text
    assert json.loads(_metrics_text(job_dir, "--json"))["spend_all_attempts"] is None

    # Beside a readable job, the sum says it has an unknown part.
    other = Evaluation(
        tasks_dir=_tasks(tmp_path / "second"),
        jobs_dir=job._jobs_dir,
        job_name="readable",
        config=EvaluationConfig(
            agent="claude-agent-acp",
            model="claude-haiku-5-5",
            concurrency=1,
            retry=RetryConfig(max_retries=2, min_wait_sec=0.0),
        ),
        preflight=False,
    )
    other._run_single_task = AsyncMock(side_effect=_runner(other))
    await other.run()
    text = _rows(_metrics_text(job._jobs_dir))
    assert "$0.9375 over 4 attempts" in text
    assert (
        "plus unknown spend in 1 job folder whose attempt records could not be read"
        in text
    )


@pytest.mark.asyncio
async def test_a_spend_nothing_recorded_is_unknown_not_none(tmp_path):
    """Review 2, #2: every trial stopped with nothing recorded. summary.json's
    spend_all_attempts.cost_usd is null; the Spend row said "none: no
    attempt called a model". It is unknown."""
    job = _job(tmp_path, None)
    job_dir = job._jobs_dir / job._job_name

    async def stopped(task_path, _cfg):
        _stopped_attempt(
            job_dir, f"{task_path.name}__stop1", task_path.name, 0.0, recorded=False
        )
        raise asyncio.CancelledError

    job._run_single_task = AsyncMock(side_effect=stopped)
    with pytest.raises(asyncio.CancelledError):
        await job.run()
    spend = _summary(job)["spend_all_attempts"]
    assert spend["cost_usd"] is None and spend["stopped_attempts_without_usage"] >= 1
    text = _rows(_metrics_text(job_dir))
    assert "every attempt: unknown: none of" in text
    assert "no attempt called a model" not in text


def test_the_latest_stopped_attempt_is_picked_by_its_instant_not_its_text():
    """Review 2, #6: result.json writes local time with its offset; text
    order put 10:00+09:00 (01:00 UTC) after 05:00+00:00."""
    from benchflow._utils.evaluation_results import kept_usage_rows

    stopped = {
        "task-b__early": {
            "task_name": "task-b",
            "finished_at": "2026-11-13T10:00:00+09:00",
        },
        "task-b__late": {
            "task_name": "task-b",
            "finished_at": "2026-11-13T05:00:00+00:00",
        },
    }
    assert list(kept_usage_rows({}, stopped)) == ["task-b__late"]


@pytest.mark.asyncio
async def test_a_stopped_attempt_with_tokens_and_no_price_is_unknown_spend(tmp_path):
    """The delta review of f9e30b5f2: a stopped attempt's receipt recorded
    1000 tokens and no cost (a model no price table has). summary.json says
    cost_usd null, tokens 1000; the Spend row said "none: no attempt called
    a model". It is unknown, and the attempt is counted
    (stopped_attempts_without_cost)."""
    job = _job(tmp_path, None)
    job_dir = job._jobs_dir / job._job_name

    async def stopped(task_path, _cfg):
        _stopped_attempt(job_dir, f"{task_path.name}__stop1", task_path.name, None)
        raise asyncio.CancelledError

    job._run_single_task = AsyncMock(side_effect=stopped)
    with pytest.raises(asyncio.CancelledError):
        await job.run()
    spend = _summary(job)["spend_all_attempts"]
    n = spend["stopped_attempts"]
    assert n >= 1 and spend["cost_usd"] is None and spend["tokens"] == 1000 * n
    assert spend["stopped_attempts_without_cost"] == n
    text = _rows(_metrics_text(job_dir))
    attempts = f"{n} attempt{'' if n == 1 else 's'}"
    assert f"every attempt: unknown: none of {attempts} reported a cost" in text
    assert "no attempt called a model" not in text


@pytest.mark.asyncio
async def test_a_stopped_trial_folder_read_alone_is_a_stopped_attempt(tmp_path):
    """The delta review's nit: `bench eval metrics <job>/<stopped trial>`
    counted it as a finished attempt ("$0.1000 over 1 attempt")."""
    _, job_dir = await _reviewers_job_stopped(tmp_path)
    text = _rows(_metrics_text(job_dir / "task-b__stop1"))
    assert "every attempt: $0.1000 over 1 stopped attempt, retries and judges" in text
    assert "1 trial stopped before an attempt was kept" in text
