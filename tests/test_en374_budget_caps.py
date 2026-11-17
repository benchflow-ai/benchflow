"""A job's USD and token caps count what the trials really spent.

Before: ``--max-cost-usd`` summed only the rollouts that reported a USD
figure and warned once, so on a route with no USD (a Codex login, a model no
price table has) it never tripped, and summary.json said ``stopped: false``,
``spent.cost_usd: 0``. ``--max-tokens`` read a timed-out trial's 0 tokens
as a real count. Now a rollout whose money can't be counted stops the job
(no rollout starts after it; running ones finish) and says why, and a
timed-out trial counts the tokens teardown salvaged from the agent's logs.

The result records here have result.json's own shapes: ``agent_result`` as
``rollout._results._build_rollout_result`` writes it, after that change's fields.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow import Budget
from benchflow.budget import AttemptSpend, BudgetGuard
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.models import RolloutResult

UNPRICED = "mimo-v9-unlisted"


def _tasks(tmp_path: Path, n: int) -> Path:
    tasks_dir = tmp_path / "tasks"
    for i in range(n):
        (tasks_dir / f"task-{i}").mkdir(parents=True)
        (tasks_dir / f"task-{i}" / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    return tasks_dir


def _agent_result(*, tokens: int | None, cost: float | None, source: str | None):
    """agent_result as result.json has it (rollout/_results.py)."""
    split = None if tokens is None else tokens // 4
    write = None if tokens is None else tokens - 3 * split
    return {
        "n_tool_calls": 3,
        "n_skill_invocations": 0,
        "n_prompts": 1,
        "n_input_tokens": split,
        "n_output_tokens": split,
        "n_cache_read_tokens": split,
        "n_cache_write_tokens": write,
        "n_cache_write_5m_tokens": write,
        "n_cache_write_1h_tokens": None if tokens is None else 0,
        "n_cache_creation_tokens": write,
        "total_tokens": tokens,
        "cost_usd": cost,
        "usage_source": source,
        "cost_source": None if cost is None else "list_price_estimate",
        "price_table_date": None if cost is None else "2026-11-04",
        "cost_usd_provider": None,
        "n_calls_cost_estimated": None,
        "price_source": None if cost is None else "list_price",
        "usage_source_legacy": "agent_native_acp" if source else "unavailable",
        "no_usage_reason": None,
    }


TRAJECTORY = [
    {"type": "user_message", "text": "Build the site."},
    {"type": "tool_call", "status": "completed", "title": "bash"},
    {"type": "agent_message", "text": "Done."},
]


def _runner(job: Evaluation, outcomes: dict[str, dict], *, delays=None):
    """One fake rollout per task, its result.json in its own folder."""

    async def fake_run(task_path, _cfg):
        name = task_path.name
        outcome = outcomes[name]
        await asyncio.sleep((delays or {}).get(name, 0.0))
        rollout_dir = job._jobs_dir / job._job_name / f"{name}__a1b2c3d4"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        usage = _agent_result(
            tokens=outcome.get("tokens"),
            cost=outcome.get("cost"),
            source=outcome.get("source"),
        )
        trajectory = outcome.get("trajectory", TRAJECTORY)
        payload = {
            "task_name": name,
            "rollout_name": rollout_dir.name,
            "rewards": {"reward": 1.0},
            "agent": outcome.get("agent", "opencode"),
            "model": outcome.get("model", UNPRICED),
            "n_tool_calls": 3 if trajectory else 0,
            "agent_result": usage,
            "judge_usage": outcome.get("judge_usage"),
            "trajectory_summary": {"steps": len(trajectory)},
            "error": outcome.get("error"),
            "verifier_error": None,
        }
        (rollout_dir / "result.json").write_text(json.dumps(payload))
        return RolloutResult(
            task_name=name,
            rollout_name=rollout_dir.name,
            rewards={"reward": 1.0},
            trajectory=list(trajectory),
            agent=payload["agent"],
            model=payload["model"],
            n_tool_calls=payload["n_tool_calls"],
            n_input_tokens=usage["n_input_tokens"],
            n_output_tokens=usage["n_output_tokens"],
            n_cache_read_tokens=usage["n_cache_read_tokens"],
            n_cache_write_tokens=usage["n_cache_write_tokens"],
            n_cache_creation_tokens=usage["n_cache_creation_tokens"],
            total_tokens=usage["total_tokens"],
            cost_usd=usage["cost_usd"],
            usage_source=usage["usage_source"] or "unavailable",
            cost_source=usage["cost_source"],
            price_table_date=usage["price_table_date"],
            price_source=usage["price_source"],
            judge_usage=payload["judge_usage"],
            error=outcome.get("error"),
            rollout_dir=rollout_dir,
        )

    return fake_run


def _job(tmp_path: Path, n: int, budget: Budget, *, agent: str = "opencode", **kw):
    config = EvaluationConfig(
        agent=agent,
        model=UNPRICED,
        concurrency=kw.get("concurrency", 1),
        retry=RetryConfig(max_retries=0),
    )
    return Evaluation(
        tasks_dir=_tasks(tmp_path, n),
        jobs_dir=tmp_path / "jobs",
        config=config,
        budget=budget,
        # No provider key here: the route checks would refuse the agent.
        preflight=False,
    )


def _budget(job: Evaluation) -> dict:
    return json.loads((job._jobs_dir / job._job_name / "summary.json").read_text())[
        "budget"
    ]


@pytest.mark.asyncio
async def test_a_usd_cap_stops_at_a_trial_whose_tokens_have_no_price(tmp_path):
    """A model no price table has: 4,000 tokens counted, no USD. The cap
    can't be held, so the job starts nothing more and says why."""
    job = _job(tmp_path, 3, Budget(max_cost_usd=5.0))
    outcomes = {
        f"task-{i}": {"tokens": 4000, "cost": None, "source": "end_of_turn"}
        for i in range(3)
    }
    job._run_single_task = AsyncMock(side_effect=_runner(job, outcomes))
    await job.run()
    ran = [c.args[0].name for c in job._run_single_task.await_args_list]
    assert ran == ["task-0"]
    budget = _budget(job)
    assert budget["stopped"] is True
    assert budget["reason"] == (
        f"cost unknown for {UNPRICED}/opencode (its tokens have no list price): "
        "--max-cost-usd can't be enforced"
    )
    assert budget["not_started"] == ["task-1", "task-2"]
    assert budget["cancelled"] == []
    assert budget["spent"]["usd_unknown_trials"] == 1


@pytest.mark.asyncio
async def test_a_usd_cap_stops_at_a_trial_nothing_counted(tmp_path):
    """A Codex login whose turn never reported: the agent ran (a
    trajectory), no tokens, no USD."""
    job = _job(tmp_path, 2, Budget(max_cost_usd=5.0), agent="codex-acp")
    outcomes = {
        f"task-{i}": {
            "tokens": None,
            "cost": None,
            "source": None,
            "agent": "codex-acp",
            "model": "gpt-6-astra",
        }
        for i in range(2)
    }
    job._run_single_task = AsyncMock(side_effect=_runner(job, outcomes))
    await job.run()
    budget = _budget(job)
    assert budget["reason"] == (
        "cost unknown for gpt-6-astra/codex-acp (nothing counted its tokens): "
        "--max-cost-usd can't be enforced"
    )
    assert budget["not_started"] == ["task-1"]


@pytest.mark.asyncio
async def test_running_trials_finish_when_the_cost_is_unknown(tmp_path):
    """The stop refuses starts; it cancels nothing (the trials running were
    started under the cap, and their spend is counted when they finish)."""
    job = _job(tmp_path, 3, Budget(max_cost_usd=5.0), concurrency=2)
    outcomes = {
        "task-0": {"tokens": 4000, "cost": None, "source": "end_of_turn"},
        "task-1": {"tokens": 10, "cost": 0.01, "source": "proxy"},
        "task-2": {"tokens": 10, "cost": 0.01, "source": "proxy"},
    }
    job._run_single_task = AsyncMock(
        side_effect=_runner(job, outcomes, delays={"task-1": 0.3})
    )
    await job.run()
    budget = _budget(job)
    assert budget["stopped"] is True
    assert budget["cancelled"] == []
    assert budget["not_started"] == ["task-2"]
    assert budget["spent"]["cost_usd"] == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_a_trial_that_never_reached_its_agent_does_not_stop_the_job(tmp_path):
    """A sandbox that failed to start: no trajectory, no tokens, no model
    call. It spent no model money, so the USD cap still holds."""
    job = _job(tmp_path, 2, Budget(max_cost_usd=5.0))
    outcomes = {
        "task-0": {
            "tokens": None,
            "cost": None,
            "source": None,
            "trajectory": [],
            "error": "sandbox setup failed",
        },
        "task-1": {"tokens": 10, "cost": 0.01, "source": "proxy"},
    }
    job._run_single_task = AsyncMock(side_effect=_runner(job, outcomes))
    await job.run()
    budget = _budget(job)
    assert budget["stopped"] is False
    assert budget["spent"]["usd_unknown_trials"] == 1


@pytest.mark.asyncio
async def test_a_control_run_does_not_stop_the_job(tmp_path):
    job = _job(tmp_path, 2, Budget(max_cost_usd=5.0), agent="oracle")
    outcomes = {
        f"task-{i}": {
            "tokens": None,
            "cost": None,
            "source": None,
            "agent": "oracle",
            "model": None,
        }
        for i in range(2)
    }
    job._run_single_task = AsyncMock(side_effect=_runner(job, outcomes))
    await job.run()
    assert _budget(job)["stopped"] is False


@pytest.mark.asyncio
async def test_an_unpriced_judge_stops_the_job_too(tmp_path):
    job = _job(tmp_path, 2, Budget(max_cost_usd=5.0))
    judge = {
        "calls": 1,
        "calls_without_usage": 0,
        "models": {"judge-x": 1},
        "n_input_tokens": 900,
        "n_output_tokens": 100,
        "n_cache_read_tokens": 0,
        "n_cache_creation_tokens": 0,
        "total_tokens": 1000,
        "cost_usd": None,
        "usage_source": "provider",
        "cost_source": None,
        "price_source": None,
        "price_table_date": None,
    }
    outcomes = {
        f"task-{i}": {
            "tokens": 10,
            "cost": 0.01,
            "source": "proxy",
            "judge_usage": judge,
        }
        for i in range(2)
    }
    job._run_single_task = AsyncMock(side_effect=_runner(job, outcomes))
    await job.run()
    budget = _budget(job)
    assert budget["reason"] == (
        "cost unknown for judge-x/judge (its tokens have no list price): "
        "--max-cost-usd can't be enforced"
    )
    assert budget["spent"]["tokens"] == 1010
    assert budget["spent"]["judge_tokens"] == 1000


@pytest.mark.asyncio
async def test_a_resume_stops_at_once_on_a_recorded_unknown_cost(tmp_path):
    job = _job(tmp_path, 2, Budget(max_cost_usd=5.0))
    outcomes = {
        f"task-{i}": {"tokens": 4000, "cost": None, "source": "end_of_turn"}
        for i in range(2)
    }
    job._run_single_task = AsyncMock(side_effect=_runner(job, outcomes))
    await job.run()
    resumed = Evaluation.resume(job._jobs_dir / job._job_name)
    resumed._preflight = False
    resumed._run_single_task = AsyncMock(side_effect=_runner(resumed, outcomes))
    await resumed.run()
    assert resumed._run_single_task.await_count == 0
    assert _budget(resumed)["reason"].startswith(f"cost unknown for {UNPRICED}/")


def test_without_a_usd_cap_an_unknown_cost_is_only_counted():
    guard = BudgetGuard(Budget(max_tokens=10_000))
    guard.attempt_done(
        "task-0",
        RolloutResult(
            "task-0", trajectory=TRAJECTORY, agent="opencode", total_tokens=4000
        ),
    )
    assert guard.stopped is False
    assert guard.spent()["usd_unknown_trials"] == 1
    assert guard.spent()["tokens"] == 4000


def test_a_pre_en374_zero_is_an_unknown_token_count():
    """result.json from before that change wrote 0 tokens for a trial nothing
    counted (usage_source "unavailable"): that is no count, not 0."""
    legacy = {
        "task_name": "t",
        "agent_result": {
            "total_tokens": 0,
            "cost_usd": None,
            "usage_source": "unavailable",
        },
    }
    assert AttemptSpend.of(legacy).tokens is None
    guard = BudgetGuard(Budget(max_tokens=10_000))
    guard.seed([legacy])
    assert guard.spent()["tokens_unknown_trials"] == 1
    metered = {
        "task_name": "t",
        "agent_result": {"total_tokens": 0, "usage_source": "proxy"},
    }
    assert AttemptSpend.of(metered).tokens == 0


@pytest.mark.asyncio
async def test_a_timed_out_trial_counts_its_salvaged_tokens(tmp_path, monkeypatch):
    """The forced-timeout shape of tests/test_en374_salvage.py: Claude Code's
    turn was killed after one model call and wrote no result event. Teardown
    counts the stream BenchFlow keeps on the host; the result the rollout
    builds from it is what --max-tokens counts (it counted 0 at 1633fea3)."""
    from tests.test_en374_salvage import (
        KILLED_MID_TURN,
        KILLED_TOKENS,
        KILLED_USD,
        _rollout,
        _stream_log,
    )

    home = tmp_path / "sandbox-home"
    home.mkdir()
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="native", home=home
    )
    _stream_log(trial, [KILLED_MID_TURN])
    rollout._usage_report_missing = True
    rollout._error = "Agent timed out after 14400s"
    await rollout._cleanup_before_teardown()
    result = rollout._build_result()

    guard = BudgetGuard(Budget(max_tokens=1_000_000, max_cost_usd=100.0))
    guard.attempt_done("task", result)
    spent = guard.spent()
    assert spent["tokens"] == KILLED_TOKENS["total_tokens"]
    assert spent["tokens_unknown_trials"] == 0
    assert spent["cost_usd"] == pytest.approx(KILLED_USD)
    assert guard.stopped is False

    # A resume reads the same trial from its result.json the same way.
    seeded = BudgetGuard(Budget(max_tokens=1_000_000))
    seeded.seed([json.loads((trial / "result.json").read_text())])
    assert seeded.spent()["tokens"] == KILLED_TOKENS["total_tokens"]


def test_the_cli_help_says_an_unknown_cost_stops_the_job():
    from typer.main import get_command

    from benchflow.cli.main import app

    run = get_command(app).commands["run"]
    option = next(p for p in run.params if "--max-cost-usd" in p.opts)
    assert "An unknown cost stops new starts" in option.help
    assert "judges' included" in option.help


def test_a_cost_from_a_count_that_missed_calls_is_a_floor_the_budget_names():
    """Review of PR #147 (F2b-R147, L1): the cap's ``spent`` summed a cost
    priced from some of an attempt's calls as if it were the whole. The
    budget block now counts such attempts and, under a USD cap, says the cap
    compared the counted part only. What the cap does is unchanged."""
    reason = "a turn ended without its usage report"
    guard = BudgetGuard(Budget(max_cost_usd=5.0))
    guard.attempt_done(
        "task-0",
        RolloutResult(
            "task-0",
            trajectory=TRAJECTORY,
            agent="opencode",
            total_tokens=4000,
            cost_usd=0.02,
            usage_incomplete=True,
            usage_incomplete_reason=reason,
        ),
    )
    guard.attempt_done(
        "task-1",
        RolloutResult(
            "task-1",
            trajectory=TRAJECTORY,
            agent="opencode",
            total_tokens=4000,
            cost_usd=0.01,
        ),
    )
    spent = guard.spent()
    assert spent["cost_usd"] == pytest.approx(0.03)
    assert spent["cost_incomplete_trials"] == 1
    assert guard.stopped is False
    note = guard.summary()["cost_cap_note"]
    assert note == (
        "--max-cost-usd compared the counted part only: 1 attempt counted only "
        "some of their calls, so spent.cost_usd is a floor"
    )

    # Without a USD cap: counted, no note about a cap.
    uncapped = BudgetGuard(Budget(max_tokens=100_000))
    uncapped.attempt_done(
        "task-0",
        RolloutResult(
            "task-0",
            trajectory=TRAJECTORY,
            agent="opencode",
            total_tokens=4000,
            cost_usd=0.02,
            usage_incomplete=True,
        ),
    )
    assert uncapped.spent()["cost_incomplete_trials"] == 1
    assert "cost_cap_note" not in uncapped.summary()

    # A resume reads the recorded attempt and a stopped one's receipt alike.
    resumed = BudgetGuard(Budget(max_cost_usd=5.0))
    resumed.seed(
        [
            {
                "task_name": "task-0",
                "agent_result": {
                    "cost_usd": 0.02,
                    "total_tokens": 4000,
                    "usage_incomplete": True,
                },
            },
            {
                "task_name": "task-1",
                "_budget_cancelled": True,
                "agent_result": {
                    "cost_usd": 0.01,
                    "total_tokens": 2000,
                    "usage_incomplete": True,
                },
            },
        ]
    )
    assert resumed.spent()["cost_incomplete_trials"] == 2
    assert "2 attempts counted only some" in resumed.summary()["cost_cap_note"]
