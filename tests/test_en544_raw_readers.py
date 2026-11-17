"""The training readers and every records writer follow the judge (EN-544).

A trial the judge withholds on read, a final zero-activity idle timeout or a
result written without a category whose agent did nothing, is unscored to
every reader (``integration_health.scored_as_read``). The readers a trainer
uses read result.json themselves: the rollout stream (``bench train
stream``), the group advantages (``training_signal``) and the Prime and TRL
SFT converters. Each now leaves such a trial out (``excluded: unscored``,
a null reward with ``scored`` false, no row), never a 0.0 in its group.
Every entry point that writes export records rewrites them through the judge
(results.jsonl, trainer/verifiers.jsonl, rewards.jsonl): a run, a stopped
run, ``bf.run`` and ``run_batch``, and a branch job.
"""

from __future__ import annotations

import asyncio
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from tests.test_en385_one_picker import BASE, BUSY, PROMPT, QUIET
from tests.trajectories.test_export_prime_sft import _exchange

NEVER = (
    "agent integration failure [no_activity]: 0 tool calls, 0 messages, "
    "0 thoughts after the prompt in 45.0 s"
)
# How each task's attempts end: result.json fields and the ACP trajectory.
ENDINGS: dict[str, tuple[dict, list[dict]]] = {
    # Scored 1 and 0 after real work.
    "pass": (
        {
            "rewards": {"reward": 1.0},
            "n_tool_calls": 1,
            "error": None,
            "error_category": None,
        },
        BUSY,
    ),
    "wrong": (
        {
            "rewards": {"reward": 0.0},
            "n_tool_calls": 1,
            "error": None,
            "error_category": None,
        },
        BUSY,
    ),
    # A zero-activity idle timeout: retried once, then kept. The run keeps
    # its 0.0 so it can be retried; the judge withholds it on read.
    "idle": ({}, QUIET),
    # A timeout whose model never answered: the writer withholds it.
    "never": (
        {
            "rewards": None,
            "error": NEVER,
            "error_category": "agent_integration",
            "integration_failure_info": {
                "cause": "no_activity",
                "evidence": "0 tool calls, 0 messages, 0 thoughts",
            },
        },
        QUIET,
    ),
    # A result an older writer left without a category, whose agent did
    # nothing: the judge withholds it on read.
    "nocat": ({"error": None, "error_category": None}, QUIET),
}
# An ending outside the shared job: the writer withholds an answer with no
# content (empty_trajectory), as it withholds a never-answered timeout.
MORE_ENDINGS: dict[str, tuple[dict, list[dict]]] = {
    "nocontent": (
        {
            "rewards": None,
            "error": "agent integration failure [empty_trajectory]: the agent "
            "answered with no content",
            "error_category": "agent_integration",
            "integration_failure_info": {
                "cause": "empty_trajectory",
                "evidence": "no content after the prompt",
            },
        },
        [PROMPT],
    ),
}
WITHHELD = ("idle-task__0002", "never-task__0001", "nocat-task__0001")
# GRPO over the two scored rewards (1, 0): sample std, eps 1e-4.
ADVANTAGE = 0.5 / (math.sqrt(0.5) + 1e-4)


def _task(tasks: Path, name: str) -> Path:
    path = tasks / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
    return path


def _write_trial(trial: Path, kind: str) -> dict:
    """One attempt's folder as the rollout writes it: result.json, its
    trajectories, and its export records from the reward the verifier gave
    (rollout/_results.py's writers)."""
    from benchflow.diagnostics import RolloutDiagnostics
    from benchflow.rollout._results import (
        _write_results_jsonl,
        _write_rewards_jsonl,
        _write_trainer_artifact,
    )

    fields, events = ENDINGS[kind] if kind in ENDINGS else MORE_ENDINGS[kind]
    task = trial.name.split("__")[0]
    data = {**BASE, "task_name": task, "rollout_name": trial.name, **fields}
    (trial / "trajectory").mkdir(parents=True, exist_ok=True)
    (trial / "agent").mkdir(exist_ok=True)
    (trial / "result.json").write_text(json.dumps(data))
    (trial / "trajectory" / "acp_trajectory.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )
    if not data["error"]:
        # What the gateway captured, which the SFT converters read.
        (trial / "trajectory" / "llm_trajectory.jsonl").write_text(
            "".join(
                json.dumps(e) + "\n"
                for e in (_exchange(final=False), _exchange(final=True))
            )
        )
    prompts = [PROMPT["text"]]
    (trial / "prompts.json").write_text(json.dumps(prompts))
    _write_rewards_jsonl(trial, data["rewards"], datetime.now())
    _write_trainer_artifact(
        trial,
        task_name=task,
        rollout_name=trial.name,
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
        task_name=task,
        rollout_name=trial.name,
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
        diagnostics=RolloutDiagnostics.from_result_fields(data),
        export_error=None,
        timing=data["timing"],
        agent_result=data["agent_result"],
        scoring=None,
        purpose="task",
        parent_rollout=None,
    )
    return data


async def _evaluate(
    tmp_path: Path,
    kinds,
    *,
    retries: int = 1,
    stop: str | None = None,
    results: list | None = None,
) -> Path:
    """A real Evaluation.run() whose rollouts write the attempts ENDINGS
    names; the task ``stop`` is stopped (Ctrl-C) when it starts."""
    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
    from benchflow.models import RolloutResult

    tasks = tmp_path / "tasks"
    for kind in kinds:
        _task(tasks, f"{kind}-task")
    jobs = tmp_path / "jobs"
    attempts: dict[str, int] = {}

    async def run_single_task(task_dir: Path, cfg: object) -> RolloutResult:
        kind = task_dir.name.removesuffix("-task")
        if kind == stop:
            raise asyncio.CancelledError
        attempts[kind] = attempts.get(kind, 0) + 1
        trial = jobs / "readers" / f"{task_dir.name}__{attempts[kind]:04d}"
        data = _write_trial(trial, kind)
        return RolloutResult.from_dict(data, rollout_dir=trial)

    evaluation = Evaluation(
        tasks_dir=tasks,
        jobs_dir=jobs,
        config=EvaluationConfig(concurrency=1, retry=RetryConfig(max_retries=retries)),
        job_name="readers",
    )
    evaluation._run_single_task = run_single_task
    result = await evaluation.run()
    if results is not None:
        results.append(result)
    return jobs / "readers"


def _records(trial: Path) -> dict[str, object]:
    """What a trial's own export records say: its reward, or "withheld"."""
    import benchflow as bf

    (row,) = bf.load_results_jsonl(trial / "results.jsonl")
    trainer = json.loads((trial / "trainer" / "verifiers.jsonl").read_text())
    terminal = (
        [
            json.loads(line)["value"]
            for line in (trial / "rewards.jsonl").read_text().splitlines()
            if line and json.loads(line).get("granularity") == "terminal"
        ]
        if (trial / "rewards.jsonl").is_file()
        else [None]
    )
    read = {
        "results.jsonl": row.reward,
        "trainer/verifiers.jsonl": trainer["reward"],
        "rewards.jsonl": terminal[0],
    }
    return {k: "withheld" if v is None else v for k, v in read.items()}


@pytest.fixture
def job(tmp_path: Path) -> Path:
    return asyncio.run(_evaluate(tmp_path, ENDINGS))


def test_the_job_holds_every_ending(job):
    names = sorted(p.name for p in job.iterdir() if p.is_dir())
    assert names == [
        "idle-task__0001",
        "idle-task__0002",
        "never-task__0001",
        "nocat-task__0001",
        "pass-task__0001",
        "wrong-task__0001",
    ]
    # The run keeps an idle timeout's 0.0 in result.json, for its retry.
    kept = json.loads((job / "idle-task__0002" / "result.json").read_text())
    assert (kept["error_category"], kept["rewards"]) == (
        "idle_timeout",
        {"reward": 0.0},
    )


def test_the_stream_gives_a_withheld_trial_no_reward(job):
    """bench train stream: a null reward with scored false, left out of its
    group's baseline; the replaced idle attempt stays retried."""
    from benchflow.trajectories.rollout_stream import stream_rollouts

    records = {
        r.rollout: r
        for r in stream_rollouts(job, follow=False, group_by="model", group_size=5)
    }
    first = records["idle-task__0001"]
    assert first.retried
    assert first.advantage is None
    for name in WITHHELD:
        record = records[name]
        assert (record.reward, record.scored, record.advantage) == (
            None,
            False,
            None,
        ), name
    assert records["pass-task__0001"].advantage == pytest.approx(ADVANTAGE)
    assert records["wrong-task__0001"].advantage == pytest.approx(-ADVANTAGE)


def test_the_group_advantages_exclude_a_withheld_trial(job):
    """training_signal (rollout_training_signals, --group-advantage)."""
    from benchflow.trajectories.training_signal import rollout_training_signals

    signals = rollout_training_signals(job, group_by="model")
    (group,) = signals.manifest()["groups"]
    members = {m["rollout"]: m for m in group["members"]}
    for name in WITHHELD:
        assert members[name]["reward"] is None, name
        assert members[name]["advantage"] is None, name
        assert members[name]["excluded"] == "unscored", name
    assert group["scored"] == 2
    assert members["pass-task__0001"]["advantage"] == pytest.approx(ADVANTAGE)
    assert members["wrong-task__0001"]["advantage"] == pytest.approx(-ADVANTAGE)


@pytest.mark.parametrize("converter", ["prime-sft", "trl-sft"])
def test_the_sft_converters_write_no_row_for_a_withheld_trial(job, converter):
    from benchflow.trajectories.export_prime_sft import (
        convert_benchflow_rollouts_to_prime_sft_rows,
    )
    from benchflow.trajectories.export_trl_sft import (
        convert_benchflow_rollouts_to_trl_sft_rows,
    )

    if converter == "prime-sft":
        rows, _ = convert_benchflow_rollouts_to_prime_sft_rows(
            job, group_advantage="grpo", group_by="model"
        )
    else:
        rows, _ = convert_benchflow_rollouts_to_trl_sft_rows(
            job, row_mode="rollout", group_advantage="grpo", group_by="model"
        )
    by = {}
    for row in rows:
        path = row.get("source_rollout_dir") or str(Path(row["source_path"]).parents[1])
        by.setdefault(Path(path).name, []).append(row["advantage"])
    assert sorted(by) == ["pass-task__0001", "wrong-task__0001"]
    assert by["pass-task__0001"] == [pytest.approx(ADVANTAGE)] * len(
        by["pass-task__0001"]
    )
    assert by["wrong-task__0001"] == [pytest.approx(-ADVANTAGE)] * len(
        by["wrong-task__0001"]
    )


def test_the_runs_records_follow_the_judge(job):
    """Evaluation.run: the trials' own export records (and so the job's)."""
    for name in WITHHELD:
        read = _records(job / name)
        assert read == dict.fromkeys(read, "withheld"), name
    assert set(_records(job / "pass-task__0001").values()) == {1.0}
    assert set(_records(job / "wrong-task__0001").values()) == {0.0}


def test_a_stopped_run_rewrites_its_trials_records(tmp_path):
    """A run stopped (Ctrl-C, SIGTERM) after a trial finished: that trial's
    records are rewritten before the stop goes on."""
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_evaluate(tmp_path, ["idle", "stop"], retries=0, stop="stop"))
    trial = tmp_path / "jobs" / "readers" / "idle-task__0001"
    read = _records(trial)
    assert read == dict.fromkeys(read, "withheld")


@pytest.mark.parametrize("entry", ["bf.run", "run_batch"])
def test_bf_run_and_run_batch_rewrite_a_withheld_trials_records(
    tmp_path, monkeypatch, entry
):
    """bf.run (bf.arun, bf.run_sync) and as_completed/run_batch end in
    benchflow.runtime.run, which rewrites the trial's records."""
    import benchflow as bf
    from benchflow import batch, runtime
    from benchflow.models import RolloutResult
    from benchflow.rollout import RolloutConfig

    task = _task(tmp_path / "tasks", "idle-task")

    class Finished:
        def __init__(self, config: RolloutConfig) -> None:
            self.config = config

        async def run(self) -> RolloutResult:
            job = Path(self.config.jobs_dir) / (self.config.job_name or "job")
            trial = job / "idle-task__0001"
            return RolloutResult.from_dict(
                _write_trial(trial, "idle"), rollout_dir=trial
            )

    async def create(config: RolloutConfig) -> Finished:
        return Finished(config)

    monkeypatch.setattr("benchflow.rollout.Rollout.create", create)
    monkeypatch.setattr(batch, "_check_configs", lambda items: None)
    monkeypatch.setattr(runtime, "check_host", lambda items: None)
    config = RolloutConfig.from_legacy(
        task_path=task, agent="codex-acp", model="gpt-5.5", jobs_dir=tmp_path / "jobs"
    )
    if entry == "bf.run":
        result = bf.run_sync(config)
    else:
        (result,) = bf.run_batch([config], concurrency=1)
    trial = Path(result.rollout_dir)
    assert json.loads((trial / "result.json").read_text())["rewards"] == {"reward": 0.0}
    read = _records(trial)
    assert read == dict.fromkeys(read, "withheld")


@pytest.fixture
def branch_task(tmp_path: Path) -> Path:
    path = tmp_path / "task"
    path.mkdir()
    (path / "instruction.md").write_text("Do it.")
    return path


async def test_a_branch_job_rewrites_a_withheld_trials_records(tmp_path, branch_task):
    """bench eval branch: the parent trial ended as a zero-activity idle
    timeout; the branch job's own records and results.jsonl withhold it."""
    from benchflow.branch_run import run_branch_trial, write_branch_job
    from tests.test_branch_run import ScriptedRollout, _plan

    plan = _plan(tmp_path, task_paths=[branch_task], checkpoint_after=0)
    outcome = await run_branch_trial(plan, branch_task, rollout_factory=ScriptedRollout)
    trial = ScriptedRollout.last._rollout_dir
    for name in ("result.json", "results.jsonl", "rewards.jsonl"):
        (trial / name).unlink(missing_ok=True)
    shutil.rmtree(trial / "trainer", ignore_errors=True)
    _write_trial(trial, "idle")
    job_dir = write_branch_job(plan, [outcome])
    read = _records(trial)
    assert read == dict.fromkeys(read, "withheld")
    rows = [
        json.loads(line)
        for line in (job_dir / "results.jsonl").read_text().splitlines()
        if line
    ]
    assert [row["reward"] for row in rows] == [None]


# What the SDK hands its caller (R195 F1, F2; the lead's terms 2026-11-14):
# every public entry point gives a withheld trial reward None, its explicit
# status (error_category "agent_integration" with its integration_failure_info,
# score_outcome "errored"), and leaves it out of n_scored and the scored-only
# mean; the every-trial mean counts it as 0, as it counts any unscored trial.


def _withheld(result) -> None:
    assert result.rewards is None
    assert result.reward is None
    assert result.score_outcome == "errored"
    assert result.error_category == "agent_integration"
    assert result.integration_failure_info["cause"] == "no_activity"


def _sdk(tmp_path: Path, monkeypatch) -> list:
    """Stub the rollout's lifecycle (Rollout.create), so the real
    benchflow.runtime.run runs: each task's trial is written as ENDINGS says
    for the task's kind."""
    from benchflow import batch, runtime
    from benchflow.models import RolloutResult

    class Finished:
        def __init__(self, config) -> None:
            self.config = config

        async def run(self) -> RolloutResult:
            job = Path(self.config.jobs_dir) / (self.config.job_name or "job")
            name = Path(self.config.task_path).name
            trial = job / f"{name}__0001"
            data = _write_trial(trial, name.removesuffix("-task"))
            return RolloutResult.from_dict(data, rollout_dir=trial)

    async def create(config):
        return Finished(config)

    monkeypatch.setattr("benchflow.rollout.Rollout.create", create)
    monkeypatch.setattr(batch, "_check_configs", lambda items: None)
    monkeypatch.setattr(runtime, "check_host", lambda items: None)
    from benchflow.rollout import RolloutConfig

    return [
        RolloutConfig.from_legacy(
            task_path=_task(tmp_path / "tasks", f"{kind}-task"),
            agent="codex-acp",
            model="gpt-5.5",
            jobs_dir=tmp_path / "jobs",
        )
        for kind in ("pass", "idle")
    ]


def test_bf_run_sync_hands_back_a_withheld_trial_as_withheld(tmp_path, monkeypatch):
    import benchflow as bf

    _, idle = _sdk(tmp_path, monkeypatch)
    result = bf.run_sync(idle)
    _withheld(result)
    # The run keeps its record as it wrote it, for a retry.
    recorded = json.loads((Path(result.rollout_dir) / "result.json").read_text())
    assert recorded["rewards"] == {"reward": 0.0}


def test_run_batch_results_leave_a_withheld_trial_out(tmp_path, monkeypatch):
    import benchflow as bf

    results = bf.run_batch(_sdk(tmp_path, monkeypatch), concurrency=1)
    passed, idle = results
    assert passed.reward == 1.0
    _withheld(idle)
    assert results.n_scored == 1
    assert results.mean_reward_scored_only == 1.0
    assert results.mean_reward == 0.5  # every trial; an unscored one counts as 0
    records = {r["task_name"]: r for r in results.to_records()}
    assert records["idle-task"]["reward"] is None
    assert records["idle-task"]["score_outcome"] == "errored"
    assert records["idle-task"]["error_category"] == "agent_integration"
    out = results.to_jsonl(tmp_path / "results.jsonl")
    rows = {
        row["task_name"]: row
        for row in map(json.loads, Path(out).read_text().splitlines())
    }
    assert (rows["idle-task"]["reward"], rows["idle-task"]["error_category"]) == (
        None,
        "agent_integration",
    )
    assert rows["pass-task"]["reward"] == 1.0


def test_evaluation_stream_and_on_result_hand_back_a_withheld_trial(tmp_path):
    """Evaluation.stream() and on_result: each finished task as every reader
    scores it (the kept idle timeout; the run retried the first one)."""
    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
    from benchflow.models import RolloutResult

    tasks = tmp_path / "tasks"
    for kind in ("pass", "idle"):
        _task(tasks, f"{kind}-task")
    jobs = tmp_path / "jobs"
    attempts: dict[str, int] = {}
    reported: list[tuple] = []

    def make() -> Evaluation:
        evaluation = Evaluation(
            tasks_dir=tasks,
            jobs_dir=jobs,
            config=EvaluationConfig(concurrency=1, retry=RetryConfig(max_retries=1)),
            job_name="stream",
            on_result=lambda name, r: reported.append((name, r)),
        )

        async def run_single_task(task_dir: Path, cfg: object) -> RolloutResult:
            kind = task_dir.name.removesuffix("-task")
            attempts[kind] = attempts.get(kind, 0) + 1
            trial = jobs / "stream" / f"{task_dir.name}__{attempts[kind]:04d}"
            return RolloutResult.from_dict(_write_trial(trial, kind), rollout_dir=trial)

        evaluation._run_single_task = run_single_task
        return evaluation

    asyncio.run(make().run())
    on_result = dict(reported)
    _withheld(on_result["idle-task"])
    assert on_result["pass-task"].reward == 1.0

    async def streamed() -> dict:
        return {name: r async for name, r in make().stream()}

    shutil.rmtree(jobs)
    attempts.clear()
    by_task = asyncio.run(streamed())
    _withheld(by_task["idle-task"])
    assert by_task["pass-task"].reward == 1.0


def test_traj_upload_manifest_has_no_reward_for_a_withheld_trial(tmp_path):
    """bench traj upload: the manifest's run block, as the broker validates it."""
    from benchflow.publish.traj_capture import stage_trajectory_capture
    from services.trajectory_upload.validation import validate_local_capture

    rewards = {}
    for kind in ("pass", "idle"):
        trial = tmp_path / "job" / f"{kind}-task__0001"
        _write_trial(trial, kind)
        with stage_trajectory_capture(trial, source_id=f"en544-{kind}") as staged:
            rewards[kind] = staged.manifest["run"]["reward"]
            paths = {item.relname: item.local_path for item in staged.files[:-1]}
            validate_local_capture(staged.files[-1].local_path.read_bytes(), paths)
    assert rewards == {"pass": 1.0, "idle": None}


# R196B's lows: the batch records read a withheld trial as bf.load_trial does,
# bf.run hands it back withheld even when the records rewrite fails, and
# RolloutResult.load hands it back withheld.


def test_run_batch_records_read_a_withheld_trial_as_load_trial_does(
    tmp_path, monkeypatch
):
    import csv

    import benchflow as bf

    results = bf.run_batch(_sdk(tmp_path, monkeypatch), concurrency=1)
    _, idle = results
    loaded = bf.load_trial(idle.rollout_dir)
    records = {r["task_name"]: r for r in results.to_records()}
    record = records["idle-task"]
    assert record["execution"] == loaded.execution == "integration_failed"
    assert record["assessment"] == loaded.assessment == "unscored"
    assert record["integration_failure_info"]["cause"] == "no_activity"
    assert results.denominators(include_controls=True).integration_failures == 1
    assert records["pass-task"]["execution"] == "completed"
    assert records["pass-task"]["integration_failure_info"] is None
    rows = {
        row["task_name"]: row
        for row in map(
            json.loads, results.to_jsonl(tmp_path / "r.jsonl").read_text().splitlines()
        )
    }
    assert rows["idle-task"]["execution"] == "integration_failed"
    assert rows["idle-task"]["integration_failure_info"]["cause"] == "no_activity"
    with results.to_csv(tmp_path / "r.csv").open() as handle:
        table = {row["task_name"]: row for row in csv.DictReader(handle)}
    assert table["idle-task"]["execution"] == "integration_failed"
    info = json.loads(table["idle-task"]["integration_failure_info"])
    assert info["cause"] == "no_activity"
    assert table["pass-task"]["integration_failure_info"] == ""


def test_bf_run_hands_back_withheld_when_the_records_rewrite_fails(
    tmp_path, monkeypatch
):
    import benchflow as bf
    from benchflow.rollout import _results

    _, idle = _sdk(tmp_path, monkeypatch)

    def full_disk(trial_dir: Path) -> bool:
        raise OSError("No space left on device")

    monkeypatch.setattr(_results, "rewrite_exports_as_read", full_disk)
    _withheld(bf.run_sync(idle))


def test_rollout_result_load_hands_back_a_withheld_trial_as_withheld(tmp_path):
    from benchflow.models import RolloutResult

    trial = tmp_path / "job" / "idle-task__0001"
    _write_trial(trial, "idle")
    _withheld(RolloutResult.load(trial))
    _withheld(RolloutResult.load(trial / "result.json"))
    assert RolloutResult.load(trial).trajectory == ENDINGS["idle"][1]
    passed = tmp_path / "job" / "pass-task__0001"
    _write_trial(passed, "pass")
    assert RolloutResult.load(passed).reward == 1.0


def test_a_failed_runs_result_keeps_its_record_for_the_retry(tmp_path):
    """Guard: a run that raised hands its retry loop the record as written
    (a retryable idle timeout), not the judged copy load() gives callers."""
    from benchflow.rollout import Rollout, RolloutConfig

    task = _task(tmp_path / "tasks", "idle-task")
    rollout = Rollout(
        RolloutConfig.from_legacy(
            task_path=task,
            agent="codex-acp",
            model="gpt-5.5",
            jobs_dir=tmp_path / "jobs",
        )
    )
    trial = tmp_path / "jobs" / "job" / "idle-task__0001"
    _write_trial(trial, "idle")
    rollout._rollout_dir = trial
    rollout._rollout_name = trial.name
    rollout._started_at = datetime.now()
    result = rollout.result_after_failure(RuntimeError("sandbox lost"))
    assert (result.error_category, result.rewards) == ("idle_timeout", {"reward": 0.0})


def test_evaluation_result_reads_every_withheld_kind_as_load_trial_does(tmp_path):
    """EvaluationResult (Evaluation.run, run_sync, resume): its results, to_csv
    and to_jsonl give every withheld kind its finding and integration_failed,
    as bf.load_trial does: judged on read (idle, no category) and withheld by
    the run (never-answered, no content) alike (R213 L-A)."""
    import csv

    import benchflow as bf

    kinds = ["pass", "idle", "never", "nocontent", "nocat"]
    out: list = []
    asyncio.run(_evaluate(tmp_path, kinds, retries=0, results=out))
    (result,) = out
    with result.to_csv(tmp_path / "r.csv").open() as handle:
        table = {row["task_name"]: row for row in csv.DictReader(handle)}
    rows = {
        row["task_name"]: row
        for row in map(
            json.loads, result.to_jsonl(tmp_path / "r.jsonl").read_text().splitlines()
        )
    }
    for kind in kinds[1:]:
        task = f"{kind}-task"
        typed = result.results[task]
        loaded = bf.load_trial(typed.rollout_dir)
        cause = loaded.integration_failure["cause"]
        assert loaded.execution == "integration_failed", kind
        assert typed.integration_failure_info["cause"] == cause, kind
        assert table[task]["execution"] == "integration_failed", kind
        assert json.loads(table[task]["integration_failure_info"])["cause"] == cause
        assert rows[task]["execution"] == "integration_failed", kind
        assert rows[task]["integration_failure_info"]["cause"] == cause, kind
    assert table["pass-task"]["execution"] == "completed"
    assert result.results["pass-task"].integration_failure_info is None
