"""A usage limit fails fast: unscored, never retried, and it stops the job.

Guards the dx/errors fix for the hill-climb smoke of 2026-09-30, where every
trial on a spent Claude subscription surfaced as ``acp_error`` and was retried
twice (three sandboxes per task), and a caller could only find out by
matching ``You've hit your ... limit`` in each trial's error string.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.scoring import USAGE_LIMIT, classify_error
from benchflow.agents.env import login_label
from benchflow.agents.errors import UsageLimitError
from benchflow.diagnostics import UsageLimitDiagnostic
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.models import RolloutResult
from benchflow.rollout.session_factory_runtime import execute_prompts_session_factory

TOKEN = "sk-ant-oat01-not-a-real-token-but-long-enough-to-scan-for"
WEEKLY = "You've hit your weekly limit · resets Oct 2, 4pm (UTC)"


@pytest.mark.parametrize(
    ("agent", "env", "explicit", "expected"),
    [
        (
            "claude-agent-acp",
            {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN},
            {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN},
            "CLAUDE_CODE_OAUTH_TOKEN (agent_env)",
        ),
        (
            "claude-agent-acp",
            {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN},
            {},
            "CLAUDE_CODE_OAUTH_TOKEN (environment)",
        ),
        (
            "claude-agent-acp",
            {"ANTHROPIC_API_KEY": TOKEN, "CLAUDE_CODE_OAUTH_TOKEN": TOKEN},
            {},
            "ANTHROPIC_API_KEY (environment)",
        ),
        (
            "claude-agent-acp",
            {"_BENCHFLOW_SUBSCRIPTION_AUTH": "1"},
            {},
            "host login (~/.claude/.credentials.json)",
        ),
        (
            "claude-agent-acp",
            {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN, "BENCHFLOW_LOGIN_LABEL": "  WORK "},
            {},
            "WORK",
        ),
        (
            "codex-acp",
            {"CODEX_ACCESS_TOKEN": TOKEN},
            {},
            "CODEX_ACCESS_TOKEN (environment)",
        ),
        ("claude-agent-acp", {}, {}, None),
    ],
)
def test_the_login_is_named_by_a_label_never_its_value(agent, env, explicit, expected):
    label = login_label(agent, None, env, explicit)
    assert label == expected
    assert TOKEN not in (label or "")


def _claude_rollout(tmp_path: Path):
    from benchflow.rollout import Rollout, RolloutConfig

    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task",
            agent="claude-agent-acp",
            model="claude-haiku-4-5-20251001",
            agent_env={"CLAUDE_CODE_OAUTH_TOKEN": TOKEN},
        )
    )
    rollout._agent_env = {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN}
    return rollout


def test_the_rollout_names_the_login_and_records_the_category(tmp_path):
    rollout = _claude_rollout(tmp_path)
    error = UsageLimitError.from_text(
        WEEKLY, now=datetime(2026, 9, 30, 5, 0, tzinfo=UTC)
    )
    assert error is not None
    text = rollout._classify_acp_error(error)
    assert text == (
        "usage limit reached on login CLAUDE_CODE_OAUTH_TOKEN (agent_env): "
        "7-day window, resets 2026-10-02 16:00 UTC (" + WEEKLY + ")"
    )
    assert TOKEN not in text
    assert rollout._diagnostics.category_for_channel("error") == USAGE_LIMIT
    info = rollout._diagnostics.to_result_fields()["usage_limit_info"]
    assert info == {
        "login": "CLAUDE_CODE_OAUTH_TOKEN (agent_env)",
        "window": "7-day",
        "resets_at": "2026-10-02T16:00:00+00:00",
        "detail": WEEKLY,
    }
    assert UsageLimitDiagnostic.format_issue_from_dict("t", info).startswith(
        "t: usage limit reached on login CLAUDE_CODE_OAUTH_TOKEN (agent_env)"
    )


@pytest.mark.parametrize(
    "error",
    [
        "usage limit reached on login X: 7-day window, resets 2026-10-02 16:00 UTC (…)",
        # A result written before this fix (the smoke's own error string).
        "ACP error -32603: Internal error: You've hit your weekly limit · resets Oct 2, 4pm (UTC)",
    ],
)
def test_a_usage_limit_is_never_retried(error):
    assert classify_error(error) == USAGE_LIMIT
    assert RetryConfig().should_retry(error) is False
    # Not even when a caller's own exclude list leaves it out.
    assert RetryConfig(exclude_categories={"timeout"}).should_retry(error) is False
    # Other ACP errors still are.
    assert RetryConfig().should_retry("ACP error -32603: Internal error") is True


async def test_a_session_factory_usage_limit_is_not_turned_into_a_timeout():
    class Spent:
        def __init__(self) -> None:
            self.steps: list = []

        async def prompt(self, _text):
            raise UsageLimitError(WEEKLY)

    with pytest.raises(UsageLimitError):
        await execute_prompts_session_factory(Spent(), ["hi"], timeout=30)


def _tasks(tmp_path: Path, n: int) -> Path:
    tasks_dir = tmp_path / "tasks"
    for i in range(n):
        (tasks_dir / f"task-{i}").mkdir(parents=True, exist_ok=True)
        (tasks_dir / f"task-{i}" / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    return tasks_dir


def _job(tmp_path: Path, n: int, *, concurrency: int = 1) -> Evaluation:
    return Evaluation(
        tasks_dir=_tasks(tmp_path, n),
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent="claude-agent-acp",
            model="claude-haiku-4-5-20251001",
            concurrency=concurrency,
            retry=RetryConfig(max_retries=2),
        ),
        job_name="job",
        preflight=False,
    )


def _runner(job: Evaluation, *, spent: set[str], overlap: bool = False):
    """Trials of ``spent`` end on the usage limit, the rest pass. With
    ``overlap`` a passing trial is already running when a spent one ends."""
    running, spent_done = asyncio.Event(), asyncio.Event()
    info = {
        "login": "CLAUDE_CODE_OAUTH_TOKEN (environment)",
        "window": "7-day",
        "resets_at": "2026-10-02T16:00:00+00:00",
        "detail": WEEKLY,
    }
    text = (
        "usage limit reached on login CLAUDE_CODE_OAUTH_TOKEN (environment): "
        f"7-day window, resets 2026-10-02 16:00 UTC ({WEEKLY})"
    )

    async def fake_run(task_path, _cfg):
        limited = task_path.name in spent
        if overlap and limited:
            await running.wait()
        elif overlap:
            running.set()
            await spent_done.wait()
        rollout_dir = job._jobs_dir / job._job_name / f"{task_path.name}__abcd1234"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_name": task_path.name,
            "rollout_name": rollout_dir.name,
            "rewards": None if limited else {"reward": 1.0},
            "error": text if limited else None,
            "error_category": USAGE_LIMIT if limited else None,
            "verifier_error": None,
            "usage_limit_info": info if limited else None,
        }
        (rollout_dir / "result.json").write_text(json.dumps(payload))
        if limited:
            spent_done.set()
        return RolloutResult(
            task_name=task_path.name,
            rollout_name=rollout_dir.name,
            rewards=payload["rewards"],
            error=payload["error"],
            error_category=payload["error_category"],
            rollout_dir=rollout_dir,
        )

    return fake_run


async def test_the_first_usage_limit_stops_the_job_and_raises(tmp_path):
    job = _job(tmp_path, 3)
    job._run_single_task = AsyncMock(side_effect=_runner(job, spent={"task-0"}))
    with pytest.raises(UsageLimitError) as caught:
        await job.run()
    err = caught.value
    # One attempt, no retry, and the other trials never started.
    assert [c.args[0].name for c in job._run_single_task.await_args_list] == ["task-0"]
    assert err.login == "CLAUDE_CODE_OAUTH_TOKEN (environment)"
    assert err.window == "7-day"
    assert err.resets_at == datetime(2026, 10, 2, 16, 0, tzinfo=UTC)
    # The job finished writing: the caller gets its result on the error.
    result = err.result
    assert result is not None
    assert (result.total, result.errored, result.passed) == (1, 1, 0)
    assert result.job_dir is not None
    summary = json.loads((result.job_dir / "summary.json").read_text())
    assert summary["error_categories"] == {USAGE_LIMIT: 1}
    assert summary["usage_limit"]["not_started"] == ["task-1", "task-2"]
    assert summary["usage_limit"]["login"] == "CLAUDE_CODE_OAUTH_TOKEN (environment)"


async def test_running_trials_finish_and_a_resume_runs_the_rest(tmp_path):
    job = _job(tmp_path, 3, concurrency=2)
    job._run_single_task = AsyncMock(
        side_effect=_runner(job, spent={"task-0"}, overlap=True)
    )
    with pytest.raises(UsageLimitError) as caught:
        await job.run()
    result = caught.value.result
    # task-1 was running when task-0 hit the limit: it finished and scored.
    assert sorted(result.results) == ["task-0", "task-1"]
    assert result.passed == 1

    # On another login the same job resumes: the usage-limit trial and the
    # trial that never started run again; the passed one is kept.
    again = _job(tmp_path, 3)
    again._run_single_task = AsyncMock(side_effect=_runner(again, spent=set()))
    rerun = await again.run()
    assert sorted(c.args[0].name for c in again._run_single_task.await_args_list) == [
        "task-0",
        "task-2",
    ]
    assert (rerun.total, rerun.passed, rerun.errored) == (3, 3, 0)


def test_from_result_reads_a_saved_trial(tmp_path):
    payload = {
        "task_name": "t",
        "error": "usage limit reached on login L: 7-day window (…)",
        "error_category": USAGE_LIMIT,
        "usage_limit_info": {
            "login": "L",
            "window": "7-day",
            "resets_at": "2026-10-02T16:00:00+00:00",
            "detail": WEEKLY,
        },
    }
    err = UsageLimitError.from_result(payload)
    assert err is not None
    assert (err.login, err.window) == ("L", "7-day")
    assert err.resets_at == datetime(2026, 10, 2, 16, 0, tzinfo=UTC)
    assert (
        UsageLimitError.from_result({"error": "boom", "error_category": "acp_error"})
        is None
    )


def test_a_branch_child_on_a_spent_login_is_not_retried():
    """A branch child that fails before its agent does anything is retried
    once in a new sandbox; a usage limit would only repeat."""
    from benchflow.rollout_branch import _retry_before_work
    from benchflow.trajectories.tree import RolloutNode

    node = RolloutNode(id="child")
    assert _retry_before_work(ConnectionError("connect timeout"), node, False)
    assert not _retry_before_work(UsageLimitError(WEEKLY), node, False)
