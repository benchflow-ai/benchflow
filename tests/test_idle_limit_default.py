"""BenchFlow's idle limit follows the run's effort, and a run's own limit wins.

A model at max effort reasoned 12 to 15 minutes without output in a published
run, and the 600 s default stopped it as idle. The default is 1800 s at xhigh
and max effort; ``--agent-idle-timeout`` (or a config's
``agent_idle_timeout_sec``) always wins; the card says which applied; and a
trial the watchdog stopped is recorded as ended by the harness.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from benchflow._utils.config import default_agent_idle_timeout
from benchflow.cli.main import eval_run
from benchflow.evaluation import (
    Evaluation,
    EvaluationConfig,
    EvaluationResult,
    _config_from_record,
    _config_to_record,
)
from benchflow.run_card import build_run_card, card_lines


def _tasks(tmp_path: Path) -> Path:
    tasks = tmp_path / "tasks"
    for name in ("alpha", "beta"):
        task = tasks / name
        task.mkdir(parents=True)
        (task / "task.toml").write_text(
            'version = "1.0"\n[agent]\ntimeout_sec = 3600\n[verifier]\ntimeout_sec = 60\n'
        )
        (task / "instruction.md").write_text("solve\n")
    return tasks


def _config_of(tmp_path: Path, **flags) -> EvaluationConfig:
    """The EvaluationConfig `bench run` builds from *flags*."""
    captured: dict = {}

    async def fake_run(self):
        captured["config"] = self._config
        return EvaluationResult(job_name="test", config=self._config)

    with patch("benchflow.evaluation.Evaluation.run", new=fake_run):
        eval_run(jobs_dir=str(tmp_path / "jobs"), **flags)
    return captured["config"]


@pytest.mark.parametrize(
    "effort,limit,source",
    [
        (None, 600, "default"),
        ("high", 600, "default"),
        ("xhigh", 1800, "effort_xhigh"),
        ("max", 1800, "effort_max"),
    ],
)
def test_the_default_idle_limit_follows_the_effort(effort, limit, source):
    assert default_agent_idle_timeout(effort) == (limit, source)
    config = EvaluationConfig(reasoning_effort=effort)
    assert config.agent_idle_timeout == limit
    assert config.idle_timeout_source() == source


def test_bench_run_at_max_effort_gets_the_longer_default(tmp_path):
    config = _config_of(tmp_path, tasks_dir=_tasks(tmp_path), reasoning_effort="max")
    assert config.agent_idle_timeout == 1800
    assert config.idle_timeout_source() == "effort_max"


def test_an_explicit_idle_limit_wins_over_the_effort_s_default(tmp_path):
    config = _config_of(
        tmp_path,
        tasks_dir=_tasks(tmp_path),
        reasoning_effort="max",
        agent_idle_timeout="10m",
    )
    assert config.agent_idle_timeout == 600
    assert config.idle_timeout_source() == "given"
    python = EvaluationConfig(
        reasoning_effort="max", agent_idle_timeout=600, agent_idle_timeout_given=True
    )
    assert python.agent_idle_timeout == 600


def test_a_config_file_s_effort_from_the_command_line_moves_the_default(tmp_path):
    tasks = _tasks(tmp_path)
    plain = tmp_path / "plain.yaml"
    plain.write_text(
        f"tasks_dir: {tasks}\nagent: claude-agent-acp\nreasoning_effort: low\n"
    )
    config = _config_of(tmp_path, config_file=plain, reasoning_effort="max")
    assert (config.reasoning_effort, config.agent_idle_timeout) == ("max", 1800)
    own = tmp_path / "own.yaml"
    own.write_text(
        f"tasks_dir: {tasks}\nagent: claude-agent-acp\nreasoning_effort: max\n"
        "agent_idle_timeout_sec: 900\n"
    )
    config = _config_of(tmp_path / "b", config_file=own)
    assert config.agent_idle_timeout == 900
    assert config.idle_timeout_source() == "given"


def test_a_job_s_recorded_limit_is_kept_on_resume():
    """A job an earlier BenchFlow ran at max effort with 600 s resumes at
    600 s: its record holds the limit it ran with."""
    record = _config_to_record(EvaluationConfig(reasoning_effort="max"))
    record["agent_idle_timeout"] = 600
    record.pop("agent_idle_timeout_given")
    config = _config_from_record(record, None)
    assert config.agent_idle_timeout == 600
    assert config.idle_timeout_source() == "given"


def test_the_card_says_where_the_idle_limit_came_from(tmp_path):
    def limits(**config) -> str:
        job = Evaluation(
            tasks_dir=str(_tasks(tmp_path / str(len(config)))),
            jobs_dir=str(tmp_path / "jobs"),
            config=EvaluationConfig(agent="claude-agent-acp", **config),
            preflight=False,
        )
        card = build_run_card(job, job._get_task_dirs())
        return next(
            line
            for line in card_lines(card, tmp_path / "j")
            if line.startswith("  Limits:")
        )

    assert "; idle 30 min (BenchFlow's default at effort max);" in limits(
        reasoning_effort="max"
    )
    assert "; idle 10 min (BenchFlow's default);" in limits()
    assert "; idle 20 min (the run's);" in limits(
        reasoning_effort="max", agent_idle_timeout=1200
    )


def _result(trial: Path, *, error: str | None, stop_reasons) -> dict:
    from benchflow.acp.session import ACPSession
    from benchflow.rollout._results import _build_rollout_result
    from benchflow.trajectories._capture import _capture_session_trajectory

    trial.mkdir(parents=True)
    session = ACPSession("trial")
    session.record_user_prompt("task")
    session.mark_prompt_end()
    _build_rollout_result(
        trial,
        task_name="task",
        rollout_name=trial.name,
        agent="codex-acp",
        agent_name="codex-acp",
        model="openai/gpt-6.1-sol",
        n_tool_calls=0,
        prompts=["task"],
        error=error,
        verifier_error=None,
        trajectory=_capture_session_trajectory(session),
        partial_trajectory=False,
        stop_reasons=stop_reasons,
        rewards=None,
        started_at=datetime.now(),
        timing={},
    )
    return json.loads((trial / "result.json").read_text())


def test_a_watchdog_kill_is_a_harness_ending(tmp_path):
    result = _result(
        tmp_path / "idle",
        error="Agent idle for 600s (no tool call, message or thought); 0 tool calls",
        stop_reasons=[None],
    )
    assert result["error_category"] == "idle_timeout"
    outcome = result["agent_outcome"]
    assert (outcome["ended_by"], outcome["harness_cause"]) == (
        "harness",
        "idle_timeout",
    )
    ended = _result(tmp_path / "done", error=None, stop_reasons=["end_turn"])
    outcome = ended["agent_outcome"]
    assert (outcome["ended_by"], outcome["harness_cause"]) == ("agent", None)


def test_a_prompt_benchflow_cancelled_is_not_the_agent_s_ending():
    """The wall-clock limit cancels the prompt (its stop reason is
    ``cancelled``): a harness ending; a cancel with no harness cause is
    no agent ending either."""
    from benchflow.run_checks import harness_ending

    assert harness_ending("timeout", ["cancelled"]) == {
        "ended_by": "harness",
        "harness_cause": "timeout",
    }
    assert harness_ending(None, ["end_turn", "cancelled"]) == {
        "ended_by": None,
        "harness_cause": None,
    }
    assert harness_ending(None, ["max_tokens"])["ended_by"] == "agent"


@pytest.mark.parametrize(
    ("given", "limit", "source"),
    [(600, 600, "given"), (None, 1800, "effort_max")],
)
def test_a_sharded_worker_keeps_the_runs_idle_limit(given, limit, source):
    """A worker of a sharded run at max effort runs at the limit the run set,
    600 s included, and says where it came from as the run does."""
    from types import SimpleNamespace

    from benchflow.eval_sharding import _config_payload
    from benchflow.eval_worker import _evaluation_config

    config = EvaluationConfig(
        reasoning_effort="max",
        agent_idle_timeout=given,
        agent_idle_timeout_given=given is not None,
    )
    shard = SimpleNamespace(concurrency=1, task_names=["t"])
    worker = _evaluation_config(_config_payload(config, shard=shard))
    assert (worker.agent_idle_timeout, worker.idle_timeout_source()) == (
        limit,
        source,
    )


def _older_max_job(tmp_path: Path, *, given: bool | None) -> Path:
    """A job run with `--effort max`: its record holds 600 s, set by the run
    (``given`` True), as that build's default (False), or by a build from
    before the record said which (None)."""
    from benchflow.evaluation import EVALUATION_RECORD

    job = tmp_path / "jobs" / "fp-max"
    job.mkdir(parents=True)
    record = _config_to_record(
        EvaluationConfig(
            agent="claude-agent-acp",
            model="anthropic/claude-sonnet-5",
            reasoning_effort="max",
            agent_idle_timeout=600,
            agent_idle_timeout_given=True,
        )
    )
    if given is None:
        record.pop("agent_idle_timeout_given")
    else:
        record["agent_idle_timeout_given"] = given
    (job / EVALUATION_RECORD).write_text(
        json.dumps({"schema_version": 1, "job_name": job.name, "config": record})
    )
    trial = job / "task-a__abcd1234"
    trial.mkdir()
    (trial / "config.json").write_text(
        json.dumps({"task_name": "task-a", "agent": "claude-agent-acp"})
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "rollout_name": trial.name,
                "rewards": {"reward": 1.0},
            }
        )
    )
    return job


def _same_command() -> EvaluationConfig:
    # `bench run --job fp-max --effort max`, no --agent-idle-timeout.
    return EvaluationConfig(
        agent="claude-agent-acp",
        model="anthropic/claude-sonnet-5",
        reasoning_effort="max",
    )


@pytest.mark.parametrize("given", [None, False])
def test_an_older_max_effort_job_resumes_at_the_limit_it_ran_with(tmp_path, given):
    """The same `bench run --job <name>` after an upgrade changed BenchFlow's
    default at max effort from 600 s to 1800 s: nothing the user set changed,
    so the job resumes (named or not) and its remaining trials keep 600 s."""
    from benchflow.evaluation import (
        _check_resume_mismatch,
        _keep_recorded_idle_limit,
        _recorded_run_differs,
    )

    job = _older_max_job(tmp_path, given=given)
    config = _same_command()
    assert config.agent_idle_timeout == 1800
    assert _recorded_run_differs(job, config, None) is None
    _keep_recorded_idle_limit(job, config)
    assert (config.agent_idle_timeout, config.idle_timeout_source()) == (600, "given")
    assert _check_resume_mismatch(job, config) is None


def test_a_limit_the_older_run_set_still_counts_as_a_setting(tmp_path):
    """An explicit 600 s the job ran with, dropped on the rerun at max: the
    run now asks for BenchFlow's 1800 s, another setting, as before."""
    from benchflow.evaluation import _keep_recorded_idle_limit, _recorded_run_differs

    job = _older_max_job(tmp_path, given=True)
    config = _same_command()
    _keep_recorded_idle_limit(job, config)
    assert config.agent_idle_timeout == 1800
    assert "agent_idle_timeout" in (_recorded_run_differs(job, config, None) or "")


def test_a_rollout_config_gets_the_efforts_idle_default_too(tmp_path):
    """bf.run(RolloutConfig(...)), the SDK's run and bench eval branch build a
    RolloutConfig: it gets the effort's default as bench run does, and keeps
    a limit its caller set, 600 s at max included."""
    from benchflow.rollout import RolloutConfig

    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n[agent]\ntimeout_sec = 60\n')
    (task / "instruction.md").write_text("Solve it.\n")
    common = {"task_path": task, "agent": "claude-agent-acp", "reasoning_effort": "max"}
    assert RolloutConfig(**common).agent_idle_timeout == 1800
    kept = RolloutConfig(
        **common, agent_idle_timeout=600, agent_idle_timeout_given=True
    )
    assert kept.agent_idle_timeout == 600
    assert RolloutConfig(**common, agent_idle_timeout=900).agent_idle_timeout == 900
    low = {**common, "reasoning_effort": "low"}
    assert RolloutConfig(**low).agent_idle_timeout == 600
