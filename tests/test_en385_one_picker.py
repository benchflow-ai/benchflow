"""Every reader asks one judge how a trial ended (EN-385, 0.9.3).

The run decides: when result.json names its ending (an ``error_category``),
every reader reads it as written. The read-time zero-activity check
(``integration_health.diagnose_trial_dir``) runs only for a result written
without a category, and for an idle timeout: the run keeps an idle timeout
so the trial can be retried, and 0.9.3 keeps no attempt records to tell it
the final attempt, so its readers judge a final idle timeout whose agent did
nothing (0.10's run judges it itself, on the attempt its R2 attempt record
marks final). bf.load_trial and bf.load_job, the job export,
``bench eval inspect``, collect_metrics (``bench metrics``) and the
summary.json a run writes all call ``judged_integration_failure``, so they
agree. A wall-clock timeout whose agent did nothing is judged by the run, as
on 0.10 (tests/test_integration_health.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

IDLE = "Agent idle for 600s with no new tool call (0 tool calls so far)"
BASE = {
    "task_name": "slow-task",
    "agent": "codex-acp",
    "model": "gpt-5.5",
    "rewards": {"reward": 0.0},
    "n_tool_calls": 0,
    "n_prompts": 1,
    "error": IDLE,
    "error_category": "idle_timeout",
    "trajectory_source": "acp",
    "agent_result": {"n_input_tokens": None, "n_output_tokens": None},
    "timing": {"agent_execution": 600.0},
    "started_at": "2026-11-13T22:28:01+00:00",
    "finished_at": "2026-11-13T22:38:30+00:00",
}
PROMPT = {"type": "user_message", "text": "Create a file called `hello.txt`."}
QUIET = [PROMPT, {"type": "agent_timeout", "reason": "idle"}]
BUSY = [
    PROMPT,
    {"type": "agent_message", "text": "I'll write the file."},
    {"type": "tool_call", "tool_call_id": "t1", "title": "write hello.txt"},
    {"type": "agent_timeout", "reason": "idle"},
]


def _trial(job: Path, events: list[dict], *, exports: bool = False, **result) -> Path:
    trial = job / "slow-task__0001"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "agent").mkdir()
    (trial / "result.json").write_text(
        json.dumps({"rollout_name": trial.name, **BASE, **result})
    )
    (trial / "trajectory" / "acp_trajectory.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )
    if exports:
        _write_exports(trial, events)
    return trial


def _write_exports(trial: Path, events: list[dict]) -> None:
    """The records the rollout writes when the trial ends, from the reward
    the verifier gave (rollout/_results.py _build_rollout_result)."""
    from datetime import datetime

    from benchflow.diagnostics import RolloutDiagnostics
    from benchflow.rollout._results import (
        _write_results_jsonl,
        _write_rewards_jsonl,
        _write_trainer_artifact,
    )

    data = json.loads((trial / "result.json").read_text())
    prompts = [PROMPT["text"]]
    (trial / "prompts.json").write_text(json.dumps(prompts))
    _write_rewards_jsonl(trial, data["rewards"], datetime.now())
    _write_trainer_artifact(
        trial,
        task_name=data["task_name"],
        rollout_name=data["rollout_name"],
        agent_name=data["agent"],
        prompts=prompts,
        trajectory=events,
        rewards=data["rewards"],
        model=data["model"],
        verifier_error=None,
        error=data["error"],
    )
    _write_results_jsonl(
        trial,
        task_name=data["task_name"],
        rollout_name=data["rollout_name"],
        agent=data["agent"],
        agent_name="",
        model=data["model"],
        n_tool_calls=data["n_tool_calls"],
        prompts=prompts,
        trajectory=events,
        partial_trajectory=False,
        rewards=data["rewards"],
        error=data["error"],
        verifier_error=None,
        error_category=data["error_category"],
        verifier_error_category=None,
        diagnostics=RolloutDiagnostics(),
        export_error=None,
        timing=data["timing"],
        agent_result=data["agent_result"],
        scoring=None,
        purpose="task",
        parent_rollout=None,
    )
    for name in ("results.jsonl", "rewards.jsonl", "trainer/verifiers.jsonl"):
        assert (trial / name).is_file(), name


def _job(tmp_path: Path, events: list[dict], **result) -> Path:
    job = tmp_path / "job"
    _trial(job, events, **result)
    return job


def _verdict(failed: bool, reward: float | None) -> object:
    return "withheld" if failed else reward


def _load_trial(job: Path) -> object:
    import benchflow as bf

    (trial_dir,) = [p for p in job.iterdir() if p.is_dir()]
    trial = bf.load_trial(trial_dir)
    return _verdict(trial.integration_failure is not None, trial.reward)


def _load_job(job: Path) -> object:
    import benchflow as bf

    (trial,) = bf.load_job(job).trials
    return _verdict(trial.integration_failure is not None, trial.reward)


def _export(job: Path) -> object:
    import benchflow as bf
    from benchflow.job_export import trial_export

    (trial,) = bf.load_job(job).trials
    exported = trial_export(trial)
    return _verdict(exported.integration_failure is not None, exported.reward)


def _bench_eval_inspect(job: Path) -> object:
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    shown = CliRunner().invoke(app, ["eval", "inspect", str(job), "--json"])
    assert shown.exit_code == 0, shown.output
    (trial,) = json.loads(shown.stdout)["trials"]
    return _verdict(trial["execution"] == "integration_failed", trial["reward"])


def _collect_metrics(job: Path) -> object:
    from benchflow.metrics import collect_metrics

    summary = collect_metrics(job).summary()
    return _verdict(
        summary["integration_failures"]["total"] == 1,
        summary["mean_reward_scored_only"],
    )


def _summary_json(job: Path) -> object:
    """The rows a run's summary.json counts (Evaluation, after its trials)."""
    from benchflow._utils.scoring import extract_reward
    from benchflow.evaluation import _rows_as_read

    (trial_dir,) = [p for p in job.iterdir() if p.is_dir()]
    raw = json.loads((trial_dir / "result.json").read_text())
    (row,) = _rows_as_read({"slow-task": raw}, job).values()
    return _verdict("integration_failure_info" in row, extract_reward(row))


def _viewer(job: Path) -> object:
    """The viewer's catalog row (its group pass rates count these)."""
    from benchflow.trajectories.viewer.catalog import _rollout_summary

    row = _rollout_summary(job, "slow-task__0001")
    return _verdict(
        row["execution_detail"] == "agent integration"
        and row["assessment"] == "unscored",
        row["reward"],
    )


def _health_summary(job: Path) -> object:
    """eval_artifacts' health summary (the canonical selection and the run
    README build on it)."""
    from benchflow.eval_artifacts import build_health_summary

    health = build_health_summary(job)
    (row,) = health["rows"]
    withheld = row["error_category"] == "agent_integration"
    assert health["unscored_rows"] == (1 if withheld else 0)
    return _verdict(withheld and not row["scored"], row["reward"])


def _lift(job: Path) -> object:
    """The skill-lift report's rollouts."""
    from benchflow.eval_lift import _load_rollouts

    (rollout,) = _load_rollouts(job)
    return _verdict(
        rollout.reward is None and rollout.excluded_reason is not None,
        rollout.reward,
    )


READERS = {
    "load_trial": _load_trial,
    "load_job": _load_job,
    "export": _export,
    "bench eval inspect": _bench_eval_inspect,
    "collect_metrics": _collect_metrics,
    "summary.json": _summary_json,
    "viewer": _viewer,
    "health summary": _health_summary,
    "lift": _lift,
}


def _readers(job: Path) -> dict[str, object]:
    """What each reader says of the job's one trial: "withheld" (an unscored
    integration failure) or the reward it scores."""
    return {name: read(job) for name, read in READERS.items()}


TRIALS = {
    # A final idle timeout whose agent did nothing: withheld.
    "idle, did nothing": ({}, QUIET, "withheld"),
    # An idle timeout after real work: the idle timeout and its 0 stand.
    "idle, did work": ({"n_tool_calls": 1}, BUSY, 0.0),
    # A result written without a category: the read-time check decides.
    "no category": ({"error": None, "error_category": None}, QUIET, "withheld"),
    # Any other recorded ending is read as written, even with a quiet
    # trajectory: here a wall-clock timeout an earlier 0.9.3 build recorded.
    "recorded timeout": (
        {
            "error": "Agent prompt exceeded wall-clock budget 45s",
            "error_category": "timeout",
        },
        QUIET,
        0.0,
    ),
}


@pytest.mark.parametrize("name", list(TRIALS))
def test_every_reader_gives_the_same_answer(tmp_path, name):
    result, events, expected = TRIALS[name]
    read = _readers(_job(tmp_path, events, **result))
    assert read == dict.fromkeys(read, expected)


@pytest.mark.parametrize("reader", list(READERS))
def test_every_reader_calls_the_one_judge(tmp_path, monkeypatch, reader):
    import benchflow.integration_health as health

    judge = health.judged_integration_failure
    calls: list[str] = []

    def spy(trial_dir, result, *, control=False):
        calls.append(str(trial_dir))
        return judge(trial_dir, result, control=control)

    monkeypatch.setattr(health, "judged_integration_failure", spy)
    job = _job(tmp_path, QUIET)
    assert READERS[reader](job) == "withheld"
    assert calls, f"{reader} did not ask the judge"


CONTROLS = {
    # A control copy by name is never judged on read, whatever agent ran it.
    "empty copy by name": ({"task_name": "slow-task__e"}, None, 0.0),
    # A control by its config's task variant.
    "oracle variant": ({}, {"task_variant": "oracle"}, 0.0),
    # A result that names no agent is an agent run, judged like any other.
    "no agent recorded": ({"agent": ""}, None, "withheld"),
}


@pytest.mark.parametrize("name", list(CONTROLS))
def test_every_reader_reads_controls_by_one_rule(tmp_path, name):
    result, config, expected = CONTROLS[name]
    job = _job(tmp_path, QUIET, **result)
    if config is not None:
        (job / "slow-task__0001" / "config.json").write_text(json.dumps(config))
    read = _readers(job)
    assert read == dict.fromkeys(read, expected)


def test_a_finding_the_run_recorded_is_kept(tmp_path):
    finding = {"cause": "no_activity", "evidence": "0 tool calls"}
    read = _readers(
        _job(
            tmp_path,
            QUIET,
            rewards=None,
            error="agent integration failure [no_activity]: 0 tool calls",
            error_category="agent_integration",
            integration_failure_info=finding,
        )
    )
    assert set(read.values()) == {"withheld"}


def test_an_idle_timeout_is_still_retried():
    from benchflow.evaluation import RetryConfig

    assert RetryConfig().should_retry(IDLE, category="idle_timeout")


# Through a real Evaluation.run(), the trial fresh and then resumed (R093 M1):
# a fresh trial's summary row is rebuilt from the RolloutResult, which carries
# neither the agent nor the prompt count, so the judge has to read the trial's
# own result.json, as bf.load_job does.


async def _evaluate(tmp_path: Path, name: str, ran: list[str]) -> Path:
    result = await _run(tmp_path, name, ran)
    assert result.job_dir is not None
    return result.job_dir


async def _run(
    tmp_path: Path,
    name: str,
    ran: list[str],
    on_plan: object = None,
    exports: bool = False,
) -> object:
    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
    from benchflow.models import RolloutResult

    result, events, _ = TRIALS[name]
    tasks = tmp_path / "tasks"
    if not tasks.exists():
        (tasks / "slow-task").mkdir(parents=True)
        (tasks / "slow-task" / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    jobs = tmp_path / "jobs"
    evaluation = Evaluation(
        tasks_dir=tasks,
        jobs_dir=jobs,
        config=EvaluationConfig(retry=RetryConfig(max_retries=0)),
        job_name="one-picker",
        on_plan=on_plan,
    )

    async def run_single_task(task_dir: Path, cfg: object) -> RolloutResult:
        ran.append(task_dir.name)
        trial = _trial(jobs / "one-picker", events, exports=exports, **result)
        data = json.loads((trial / "result.json").read_text())
        return RolloutResult.from_dict(data, rollout_dir=trial)

    evaluation._run_single_task = run_single_task
    return await evaluation.run()


def _summary_file(job: Path) -> object:
    """What the summary.json the run wrote says of its one trial."""
    summary = json.loads((job / "summary.json").read_text())
    failures = summary["integration_failures"]
    if failures["total"]:
        assert (summary["errored"], summary["failed"]) == (1, 0), summary
        assert summary["error_categories"] == {"agent_integration": 1}, summary
        return "withheld", failures["by_cause"]
    assert summary["errored"] == 0, summary
    return summary["mean_reward_scored_only"], {}


def _job_as_read(job: Path) -> object:
    import benchflow as bf

    (trial,) = bf.load_job(job).trials
    info = trial.integration_failure
    return _verdict(info is not None, trial.reward), (
        {info["cause"]: 1} if info is not None else {}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("name", list(TRIALS))
async def test_a_run_writes_the_summary_every_reader_reads(tmp_path, name):
    expected = TRIALS[name][2]
    ran: list[str] = []
    for run in ("fresh", "resumed"):
        job = await _evaluate(tmp_path, name, ran)
        assert ran == ["slow-task"], run  # the resumed run reused the trial
        loaded = _job_as_read(job)
        assert loaded[0] == expected, run
        assert _summary_file(job) == loaded, run
        assert _export(job) == expected, run


# The keys of summary.json the judge decides.
JUDGED_KEYS = (
    "passed",
    "failed",
    "errored",
    "error_categories",
    "idle_timeout",
    "integration_failures",
    "score_excl_errors",
    "mean_reward_scored_only",
)


@pytest.mark.parametrize("name", list(TRIALS))
def test_a_rescore_keeps_the_counts(tmp_path, monkeypatch, name):
    """``bench eval score`` rebuilds summary.json from the trials' result.json
    files through the same judge (R093 M3)."""
    import asyncio
    from unittest.mock import AsyncMock

    from typer.testing import CliRunner

    from benchflow.cli import rescore
    from benchflow.cli.main import app

    job = asyncio.run(_evaluate(tmp_path, name, []))
    before = json.loads((job / "summary.json").read_text())
    trial = job / "slow-task__0001"
    # The reviewer returns the trial's result unchanged; only the job's
    # derived files are rebuilt.
    monkeypatch.setattr(
        rescore,
        "resume_review",
        AsyncMock(return_value=json.loads((trial / "result.json").read_text())),
    )
    scored = CliRunner().invoke(
        app, ["eval", "score", str(trial), "--tasks-root", str(tmp_path / "tasks")]
    )
    assert scored.exit_code == 0, scored.output
    after = json.loads((job / "summary.json").read_text())
    assert {k: after[k] for k in JUDGED_KEYS} == {k: before[k] for k in JUDGED_KEYS}
    assert _summary_file(job) == _job_as_read(job)


@pytest.mark.parametrize("name", list(TRIALS))
def test_the_cli_outcomes_count_what_summary_json_counts(tmp_path, name):
    """EvaluationResult.results, which the CLI's Outcomes line counts, carries
    the judge's answer, fresh and resumed, as summary.json and the Score line
    do."""
    import asyncio

    from benchflow.cli._shared import _report_eval_result, console

    for run in ("fresh", "resumed"):
        result = asyncio.run(_run(tmp_path, name, []))
        summary = json.loads((result.job_dir / "summary.json").read_text())
        with console.capture() as shown:
            _report_eval_result(result, result.job_dir)
        text = shown.get()
        scored = summary["passed"] + summary["failed"]
        assert (
            f"Outcomes: {scored} scored ({summary['passed']} passed, "
            f"{summary['failed']} failed), {summary['verifier_errored']} unscored, "
            f"{summary['errored']} errored"
        ) in text, (run, text)
        withheld = TRIALS[name][2] == "withheld"
        assert ("the agent integration broke: " in text) == withheld, (run, text)
        assert (result.results["slow-task"].reward is None) == withheld, run


@pytest.mark.parametrize("name", list(TRIALS))
def test_a_resumed_run_counts_its_reused_trials_as_read(tmp_path, name):
    """The live dashboard's breakdown of the trials a resumed run reuses
    (on_plan) counts them as the run's summary does."""
    import asyncio

    asyncio.run(_run(tmp_path, name, []))
    plans: list[tuple] = []
    result = asyncio.run(_run(tmp_path, name, [], on_plan=lambda *a: plans.append(a)))
    ((total, done, to_run, breakdown),) = plans
    assert (total, done, to_run) == (1, 1, 0)
    assert breakdown == (
        result.passed,
        result.failed,
        result.errored + result.verifier_errored,
    )


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _export_records(job: Path) -> dict[str, object]:
    """What the job's and the trial's export records say of its one trial:
    "withheld" (no reward) or the reward."""
    import benchflow as bf

    trial = job / "slow-task__0001"
    (job_row,) = bf.load_results_jsonl(job / "results.jsonl")
    (trial_row,) = bf.load_results_jsonl(trial / "results.jsonl")
    (job_trainer,) = _jsonl(job / "verifiers.jsonl")
    (trial_trainer,) = _jsonl(trial / "trainer" / "verifiers.jsonl")
    terminal = (
        [
            e["value"]
            for e in _jsonl(trial / "rewards.jsonl")
            if e.get("granularity") == "terminal"
        ]
        if (trial / "rewards.jsonl").is_file()
        else [None]
    )
    read = {
        "results.jsonl": job_row.reward,
        "trial results.jsonl": trial_row.reward,
        "verifiers.jsonl": job_trainer["reward"],
        "trial verifiers.jsonl": trial_trainer["reward"],
        "rewards.jsonl": terminal[0],
    }
    return {k: "withheld" if v is None else v for k, v in read.items()}


@pytest.mark.parametrize("name", list(TRIALS))
def test_the_export_records_follow_the_judge(tmp_path, monkeypatch, name):
    """results.jsonl, the trainer's verifiers.jsonl and rewards.jsonl give a
    trial the judge withholds no reward, never a 0, after the run, a resumed
    run and `bench eval score` (R093 L4, EN-544)."""
    import asyncio
    from unittest.mock import AsyncMock

    from typer.testing import CliRunner

    from benchflow.cli import rescore
    from benchflow.cli.main import app

    expected = TRIALS[name][2]
    for run in ("fresh", "resumed"):
        result = asyncio.run(_run(tmp_path, name, [], exports=True))
        read = _export_records(result.job_dir)
        assert read == dict.fromkeys(read, expected), run
    trial = result.job_dir / "slow-task__0001"
    monkeypatch.setattr(
        rescore,
        "resume_review",
        AsyncMock(return_value=json.loads((trial / "result.json").read_text())),
    )
    scored = CliRunner().invoke(
        app, ["eval", "score", str(trial), "--tasks-root", str(tmp_path / "tasks")]
    )
    assert scored.exit_code == 0, scored.output
    read = _export_records(result.job_dir)
    assert read == dict.fromkeys(read, expected), "bench eval score"
    # result.json stays as the run wrote it.
    assert json.loads((trial / "result.json").read_text())["rewards"] == {"reward": 0.0}
