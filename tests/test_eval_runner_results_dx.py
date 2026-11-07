"""An eval runner's results: pass@k from the CLI, budgets, interrupted jobs,
control jobs, pooled folders and folders of jobs (DF-p3-eval-runner-3 to -8).

Each test is the dogfooder's repro on files: the job folders a run writes
(result.json, config.json, evaluation.json, summary.json) or a run with the
rollout faked, and the CLI's own words checked, never only a field.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

import benchflow as bf
from benchflow.cli.main import app
from tests.test_python_sdk_load_job import _trial

WIDE = {"COLUMNS": "240"}


def _invoke(*args: str):
    return CliRunner().invoke(app, list(args), env=WIDE)


def _oracle_trials(root: Path, per_trial: list[dict[str, float]], agent="oracle"):
    """``root/trial-NN/job`` folders of control runs, as --trials writes them."""
    for i, rewards in enumerate(per_trial, start=1):
        job = root / f"trial-{i:02d}" / "job"
        for task, reward in rewards.items():
            _trial(job, task, reward=reward, agent=agent, model=None)
    return root


ORACLE_3 = [
    {"fix-git": 1.0, "regex-log": 1.0, "cancel-async-tasks": 1.0},
    {"fix-git": 1.0, "regex-log": 1.0, "cancel-async-tasks": 0.0},
    {"fix-git": 1.0, "regex-log": 1.0, "cancel-async-tasks": 1.0},
]


# ── DF-p3-eval-runner-3: pass@k for control runs from the CLI ──────────────


def test_metrics_gives_pass_at_k_for_a_folder_of_control_trials(tmp_path: Path):
    root = _oracle_trials(tmp_path / "m3" / "oracle", ORACLE_3)
    out = _invoke("eval", "metrics", str(root), "--k", "1", "--k", "2", "--k", "3")
    assert out.exit_code == 0, out.output
    # 3 tasks x 3 trials: cancel-async-tasks solved 2 of 3.
    assert "no task has" not in out.output
    assert "pass@3" in out.output and "100.0%" in out.output
    assert "88.9%" in out.output  # pass@1 = (1 + 1 + 2/3) / 3
    assert "so they are counted" in out.output
    assert "include_controls" not in out.output
    # cancel-async-tasks failed once: it is listed under Failed as well.
    assert "Failed:" in out.output and "cancel-async-tasks" in out.output


def test_metrics_include_controls_counts_them_in_a_mixed_folder(tmp_path: Path):
    job = tmp_path / "mixed"
    _trial(job, "t1", reward=1.0)
    _trial(job, "t1", reward=1.0, agent="oracle", model=None, suffix="0000oracl")
    left_out = _invoke("eval", "metrics", str(job))
    assert left_out.exit_code == 0, left_out.output
    assert "--include-controls counts them" in left_out.output
    assert "include_controls=True" not in left_out.output
    counted = _invoke("eval", "metrics", str(job), "--include-controls", "--json")
    assert counted.exit_code == 0, counted.output
    rates = json.loads(counted.stdout)["solve_rates"]
    assert rates["controls_excluded"] == 0 and rates["trials"] == 2
    assert "--include-controls" in _invoke("eval", "metrics", "--help").output


def test_pass_at_k_rows_say_the_trials_are_control_runs(tmp_path: Path):
    root = _oracle_trials(tmp_path / "o", ORACLE_3)
    rates = bf.load_job(root).solve_rates(ks=[1, 2, 3])
    rows = dict(rates.rows())
    assert rows["pass@3"] == (
        "n/a: the 9 trial(s) are control runs (include_controls=True counts them)"
    )
    assert all("no task has" not in value for value in rows.values())


def test_a_matrix_names_a_control_by_its_agent_and_refuses_none(tmp_path: Path):
    from benchflow.cli.eval_artifacts import _load_eval_matrix

    good = tmp_path / "good.yaml"
    good.write_text("models:\n  haiku: claude-haiku-4-5\n  oracle: {agent: oracle}\n")
    models = _load_eval_matrix(good)
    assert models["oracle"] == {"agent": "oracle", "model": None}
    assert models["haiku"] == {"model": "claude-haiku-4-5"}
    for text in ("models:\n  oracle: none\n", "models:\n  oracle:\n"):
        bad = tmp_path / "bad.yaml"
        bad.write_text(text)
        with pytest.raises(ValueError, match=r"is not a model name.*agent: oracle"):
            _load_eval_matrix(bad)
    shapeless = tmp_path / "list.yaml"
    shapeless.write_text("models:\n  - oracle\n")
    with pytest.raises(ValueError, match=r"models: maps each alias"):
        _load_eval_matrix(shapeless)


def test_matrix_and_trials_take_a_dataset_or_repository_and_refuse_a_config(
    tmp_path: Path,
):
    from benchflow.eval_plan import EvalCreateRequest, EvalPlanError, build_eval_plan

    matrix = tmp_path / "m.yaml"
    matrix.write_text("models:\n  oracle: {agent: oracle}\n")

    build_eval_plan(
        EvalCreateRequest(dataset="terminal-bench@2.0", agent="oracle", trials=3)
    )
    build_eval_plan(
        EvalCreateRequest(dataset="terminal-bench@2.0", matrix=matrix, trials=3)
    )
    # A task named by repository repeats too.
    build_eval_plan(
        EvalCreateRequest(
            source_repo="benchflow-ai/skillsbench", agent="oracle", trials=3
        )
    )
    with pytest.raises(EvalPlanError, match="needs a task folder, a task named by"):
        build_eval_plan(EvalCreateRequest(config_file=Path("run.yaml"), trials=3))


def _write_task(task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )


def _fake_rollouts(monkeypatch, reward: float = 1.0, tokens: int | None = None):
    from benchflow.models import RolloutResult

    async def fake_rollout_create(config):
        trial = AsyncMock()
        trial.run = AsyncMock(
            return_value=RolloutResult(
                task_name=config.task_path.name,
                rewards={"reward": reward},
                total_tokens=tokens,
            )
        )
        return trial

    monkeypatch.setattr("benchflow.rollout.Rollout.create", fake_rollout_create)


def test_trials_alone_repeats_the_agent_into_trial_folders(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    for name in ("a", "b"):
        _write_task(tasks / name)
    _fake_rollouts(monkeypatch)
    jobs = tmp_path / "jobs"
    run = _invoke(
        "eval", "run", "--tasks-dir", str(tasks), "--agent", "oracle",
        "--sandbox", "docker", "--trials", "2", "--jobs-dir", str(jobs),
    )  # fmt: skip
    assert run.exit_code == 0, run.output
    assert (jobs / "oracle" / "trial-01").is_dir()
    assert (jobs / "oracle" / "trial-02").is_dir()
    summary = json.loads((jobs / "matrix-summary.json").read_text())
    assert [r["trial"] for r in summary["runs"]] == [1, 2]
    rates = bf.load_job(jobs / "oracle").solve_rates(include_controls=True)
    assert rates.min_trials_per_task == 2 and rates.get(2) is not None


# ── DF-p3-eval-runner-5: a budget-capped job ───────────────────────────────


def test_a_capped_job_says_what_did_not_run_and_the_gate_counts_it(
    tmp_path, monkeypatch
):
    tasks = tmp_path / "tasks"
    for i in range(4):
        _write_task(tasks / f"task-{i}")
    _fake_rollouts(monkeypatch, tokens=100)
    jobs = tmp_path / "jobs"
    run = _invoke(
        "eval", "run", "--tasks-dir", str(tasks), "--agent", "oracle",
        "--sandbox", "docker", "--concurrency", "1", "--max-tokens", "150",
        "--fail-under", "0.9", "--jobs-dir", str(jobs),
        "--summary-out", str(tmp_path / "run.json"),
    )  # fmt: skip
    assert run.exit_code == 1, run.output
    assert "✓ Score" not in run.output
    assert "Score: 2/2 run (100.0%)" in run.output
    assert "2 of 4 planned tasks not run (budget)" in run.output
    assert "2/4 (50.0%) of the plan" in run.output
    assert "Gate failed: pass rate 0.50 < --fail-under 0.9" in run.output
    assert json.loads((tmp_path / "run.json").read_text())["not_run"] == 2
    [job] = [d for d in jobs.iterdir() if d.is_dir()]
    summary = json.loads((job / "summary.json").read_text())
    # score keeps its meaning; the plan goes beside it.
    assert summary["score_ratio"] == 1.0
    assert (summary["planned"], summary["ran"], summary["not_run"]) == (4, 2, 2)
    assert summary["score_of_plan"] == 0.5
    listed = _invoke("eval", "list", str(jobs))
    assert "2/2 run (100.0%), 2/4 (50.0%) of the plan; 2 not run" in listed.output, (
        listed.output
    )
    metrics = _invoke("eval", "metrics", str(job))
    # The budget kept 2 from starting (DFR-p3-10: not "not run").
    assert "4 tasks, 2 ran, 2 not started" in metrics.output, metrics.output
    assert "50.0% (2 of 4 planned tasks passed)" in metrics.output
    doc = json.loads(_invoke("eval", "metrics", str(job), "--json").stdout)
    assert (doc["planned"], doc["ran"], doc["score_of_plan"]) == (4, 2, 0.5)


@pytest.mark.asyncio
async def test_an_uncapped_job_keeps_its_score_and_summary(tmp_path):
    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
    from benchflow.models import RolloutResult

    tasks = tmp_path / "tasks"
    for i in range(2):
        _write_task(tasks / f"task-{i}")
    job = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(agent="oracle", retry=RetryConfig(max_retries=0)),
    )
    job._run_single_task = AsyncMock(
        side_effect=lambda task_path, _cfg: RolloutResult(
            task_name=task_path.name, rewards={"reward": 1.0}
        )
    )
    result = await job.run()
    assert result.not_run == 0 and result.score_of_plan == result.score == 1.0
    job_dir = tmp_path / "jobs" / job._job_name
    summary = json.loads((job_dir / "summary.json").read_text())
    assert (summary["planned"], summary["ran"], summary["score_of_plan"]) == (2, 2, 1.0)
    assert "not_run" not in summary
    record = json.loads((job_dir / "evaluation.json").read_text())
    assert record["planned_tasks"] == ["task-0", "task-1"]


# ── DF-p3-eval-runner-6: interrupted jobs in list and inspect ──────────────


def _interrupted_job(job: Path, planned: list[str], finished: list[str]) -> Path:
    job.mkdir(parents=True)
    (job / "evaluation.json").write_text(
        json.dumps({"schema_version": 1, "planned_tasks": planned, "config": {}})
    )
    for task in finished:
        _trial(job, task, reward=1.0, agent="oracle", model=None)
    # A trial that started and was cut off: config.json, no result.json.
    cut = job / f"{planned[len(finished)]}__deadbeef"
    cut.mkdir()
    (cut / "config.json").write_text(
        json.dumps({"agent": "oracle", "task_path": f"tasks/{planned[len(finished)]}"})
    )
    return job


PLANNED = [f"task-{i}" for i in range(10)]


def test_inspect_names_the_unfinished_tasks_and_how_to_resume(tmp_path: Path):
    job = _interrupted_job(
        tmp_path / "intr" / "2026-10-18__22-36-54", PLANNED, PLANNED[:3]
    )
    out = _invoke("eval", "inspect", str(job))
    assert out.exit_code == 0, out.output
    assert (
        "Not finished: 3 of 10 planned tasks finished, 1 interrupted, 6 not started"
        in out.output
    )
    assert f"bench eval resume {job}" in out.output
    # Only control runs: counted, without "0 left out (… counted)".
    assert "0 control runs left out" not in out.output
    assert "0 unscored, 0 errored, only control runs, so they are counted" in out.output


def test_list_shows_an_interrupted_job_against_its_plan(tmp_path: Path):
    jobs = tmp_path / "jobs"
    _interrupted_job(jobs / "intr-ctrlc" / "2026-10-18__22-36-54", PLANNED, PLANNED[:3])
    out = _invoke("eval", "list", str(jobs))
    assert out.exit_code == 0, out.output
    assert "intr-ctrlc/2026-10-18__22-36-54" in out.output
    assert "not finished: 3 of 10 tasks have a result" in out.output
    assert "bench eval resume" in out.output


# ── DF-p3-eval-runner-7: one row per job ───────────────────────────────────


def _finished_job(job: Path, passed: int, total: int) -> None:
    for i in range(total):
        _trial(
            job, f"t{i}", reward=1.0 if i < passed else 0.0, agent="oracle", model=None
        )
    summary = {"total": total, "passed": passed, "score": f"{passed / total:.1%}"}
    (job / "summary.json").write_text(json.dumps(summary))
    (job.parent / "summary.json").write_text(json.dumps(summary))  # the root copy


def test_list_shows_every_job_in_a_folder_of_jobs(tmp_path: Path):
    jobs = tmp_path / "jobs"
    _finished_job(jobs / "oracle-c4" / "2026-10-18__22-05-00", 9, 10)
    _finished_job(jobs / "oracle-c4" / "2026-10-18__22-40-00", 10, 10)
    for listed in (jobs, jobs / "oracle-c4"):
        out = _invoke("eval", "list", str(listed))
        assert out.exit_code == 0, out.output
        assert "9/10 (90.0%)" in out.output and "10/10 (100.0%)" in out.output
        # No job has a memory score, so no empty Memory column (DFR-p1-17).
        assert "Jobs" in out.output and "Memory" not in out.output


# ── DF-p3-eval-runner-4: comparing control jobs ────────────────────────────


def test_compare_of_two_control_jobs_pairs_their_tasks(tmp_path: Path):
    a, b = tmp_path / "c4" / "job", tmp_path / "c8" / "job"
    for job, reward in ((a, 1.0), (b, 0.0)):
        _trial(job, "fix-git", reward=1.0, agent="oracle", model=None)
        _trial(job, "cancel-async-tasks", reward=reward, agent="oracle", model=None)
    out = _invoke("eval", "compare", str(a), str(b), "--labels", "c4", "c8")
    assert out.exit_code == 0, out.output
    assert "Both sides hold only control runs, so they are counted." in out.output
    assert "2 tasks paired" in out.output
    assert "| cancel-async-tasks | 1 | 0 | -1 | paired |" in out.output


def test_compare_with_one_control_side_says_so_with_the_cli_flag(tmp_path: Path):
    a, b = tmp_path / "agent" / "job", tmp_path / "oracle" / "job"
    _trial(a, "fix-git", reward=1.0)
    _trial(b, "fix-git", reward=1.0, agent="oracle", model=None)
    out = _invoke("eval", "compare", str(a), str(b), "--labels", "agent", "oracle")
    assert out.exit_code == 0, out.output
    assert out.stdout.lstrip().startswith(
        "oracle holds only control runs (1), which are left out, so none of its "
        "tasks pair; --include-controls counts them."
    )
    assert "include_controls=True" not in out.output


# ── DF-p3-eval-runner-8: metrics on a folder that pools different runs ────


def _set_config(job: Path, **values) -> None:
    for config in job.glob("*/config.json"):
        data = json.loads(config.read_text())
        config.write_text(json.dumps({**data, **values}))


def test_metrics_warns_when_pooled_jobs_differ_in_a_setting(tmp_path: Path):
    folder = tmp_path / "oracle-c4"
    for name, user, reward in (("j1", "agent", 0.0), ("j2", None, 1.0)):
        _trial(
            folder / name,
            "git-leak-recovery",
            reward=reward,
            agent="oracle",
            model=None,
        )
        _set_config(folder / name, sandbox_user=user)
    out = _invoke("eval", "metrics", str(folder))
    assert out.exit_code == 0, out.output
    assert (
        "differ in sandbox_user (on git-leak-recovery: 'agent' vs None)" in out.output
    )
    assert "bench eval metrics <job folder>" in out.output
    # One agent: no split table, and the warning does not promise one.
    assert "By agent" not in out.output
    doc = json.loads(_invoke("eval", "metrics", str(folder), "--json").stdout)
    assert doc["settings_differ"][0]["setting"] == "sandbox_user"


def test_metrics_splits_oracle_and_nop_by_agent(tmp_path: Path):
    jobs = tmp_path / "jobs"
    for i in range(3):
        _trial(jobs / "oracle", f"t{i}", reward=1.0, agent="oracle", model=None)
        _trial(jobs / "nop", f"t{i}", reward=0.0, agent="nop", model=None)
    out = _invoke("eval", "metrics", str(jobs))
    assert out.exit_code == 0, out.output
    assert "differ in agent (nop · no model, oracle · no model)" in out.output
    assert "By agent" in out.output
    doc = json.loads(_invoke("eval", "metrics", str(jobs), "--json").stdout)
    split = {row["agent"]: row["pass_rate"] for row in doc["by_agent"]}
    assert split == {"nop": 0.0, "oracle": 1.0}


def test_list_reaches_the_jobs_of_a_trials_run(tmp_path: Path):
    jobs = tmp_path / "jobs"
    for i in (1, 2):
        _finished_job(
            jobs / "m3" / "oracle" / f"trial-{i:02d}" / "2026-10-18__22-44-00", 2, 3
        )
    out = _invoke("eval", "list", str(jobs))
    assert out.exit_code == 0, out.output
    assert "m3/oracle/trial-01/2026-10-18__22-44-00" in out.output
    assert "m3/oracle/trial-02/2026-10-18__22-44-00" in out.output
    assert "no job" not in out.output


def test_a_matrix_run_checks_its_entries_agents_not_the_default(tmp_path, monkeypatch):
    """A matrix of {agent: oracle} with no --agent was refused for the default
    agent's missing model key before any trial (the dogfooder's m4 repro)."""
    import benchflow.cli.doctor as doctor

    checked: list[tuple[str | None, str | None]] = []
    monkeypatch.setattr(
        doctor,
        "eval_preflight",
        lambda **kw: checked.append((kw.get("agent"), kw.get("model"))),
    )
    tasks = tmp_path / "tasks"
    _write_task(tasks / "a")
    _fake_rollouts(monkeypatch)
    matrix = tmp_path / "m4.yaml"
    matrix.write_text("models:\n  oracle: {agent: oracle}\n  nop: {agent: nop}\n")
    run = _invoke(
        "eval", "run", "--tasks-dir", str(tasks), "--matrix", str(matrix),
        "--sandbox", "docker", "--jobs-dir", str(tmp_path / "jobs"),
    )  # fmt: skip
    assert run.exit_code == 0, run.output
    assert checked == [("oracle", None), ("nop", None)]
    assert (tmp_path / "jobs" / "nop" / "trial-01").is_dir()


# ── Round 2, review F-1 (DX-F-MATRIX-STOP): every trial runs, one gate ─────


def _rollouts_by_call(monkeypatch, outcome):
    """Fake rollouts whose result is ``outcome(n, config)`` for the n-th
    rollout started (1-based): a reward, or None for an infrastructure error."""
    from benchflow.models import RolloutResult

    started = []

    async def fake_rollout_create(config):
        started.append(config.task_path.name)
        reward = outcome(len(started), config)
        trial = AsyncMock()
        trial.run = AsyncMock(
            return_value=RolloutResult(
                task_name=config.task_path.name,
                rewards=None if reward is None else {"reward": reward},
                error="sandbox setup failed: docker daemon went away"
                if reward is None
                else None,
                error_category="sandbox_setup" if reward is None else None,
            )
        )
        return trial

    monkeypatch.setattr("benchflow.rollout.Rollout.create", fake_rollout_create)
    return started


def test_trials_5_with_an_infrastructure_error_in_trial_01_runs_every_trial(
    tmp_path, monkeypatch
):
    tasks = tmp_path / "tasks"
    _write_task(tasks / "t")
    started = _rollouts_by_call(monkeypatch, lambda n, _c: None if n == 1 else 1.0)
    jobs = tmp_path / "jobs"
    run = _invoke(
        "eval", "run", "--tasks-dir", str(tasks), "--agent", "oracle",
        "--sandbox", "docker", "--trials", "5", "--retry-attempts", "0",
        "--jobs-dir", str(jobs),
        "--summary-out", str(tmp_path / "run.json"),
    )  # fmt: skip
    # The errored trial still makes the run exit 1, but only after all five.
    assert run.exit_code == 1, run.output
    assert len(started) == 5
    for i in range(1, 6):
        assert (jobs / "oracle" / f"trial-{i:02d}").is_dir()
    summary = json.loads((jobs / "matrix-summary.json").read_text())
    assert summary["complete"] is True
    assert [r["trial"] for r in summary["runs"]] == [1, 2, 3, 4, 5]
    assert summary["runs"][0]["errored"] == 1
    # P4's rule: the errored trial counts as a trial that did not pass.
    rates = summary["solve_rates"]["oracle"]
    assert rates["trials"] == 5 and rates["solve_rate"] == pytest.approx(0.8)
    assert "Matrix: 5 jobs (5 repeats), 4/5 trials passed, errors=1" in run.output
    doc = json.loads((tmp_path / "run.json").read_text())
    assert (doc["total"], doc["passed"], doc["errored"]) == (5, 4, 1)


def _matrix_gate_run(tmp_path, monkeypatch, models: str):
    tasks = tmp_path / "tasks"
    _write_task(tasks / "t")
    _rollouts_by_call(
        monkeypatch, lambda _n, config: 0.0 if config.agent == "nop" else 1.0
    )
    matrix = tmp_path / "m.yaml"
    matrix.write_text(models)
    jobs = tmp_path / "jobs"
    run = _invoke(
        "eval", "run", "--tasks-dir", str(tasks), "--matrix", str(matrix),
        "--trials", "2", "--sandbox", "docker", "--fail-under", "0.9",
        "--retry-attempts", "0", "--jobs-dir", str(jobs),
    )  # fmt: skip
    return run, jobs


def test_a_matrix_gate_leaves_its_nop_control_out(tmp_path, monkeypatch):
    """Control trials are out of --fail-under's numerator and
    denominator; the gate line names them; matrix-summary.json counts them."""
    # The control that would fail the gate comes first.
    run, jobs = _matrix_gate_run(
        tmp_path,
        monkeypatch,
        "models:\n  nop: {agent: nop}\n  haiku: claude-haiku-4-5\n",
    )
    assert run.exit_code == 0, run.output
    for alias in ("nop", "haiku"):
        for i in (1, 2):
            assert (jobs / alias / f"trial-{i:02d}").is_dir(), run.output
    summary = json.loads((jobs / "matrix-summary.json").read_text())
    assert len(summary["runs"]) == 4 and summary["complete"] is True
    assert summary["gate"]["controls_left_out"] == 2
    assert summary["gate"]["control_kinds"] == ["nop"]
    assert summary["gate"]["trials"] == 2
    assert (
        "Gate: 1.00 over 2 trials of claude-agent-acp; 2 control trials (nop) left out"
        in run.output
    )
    assert "Gate failed" not in run.output
    # The Matrix line still totals every trial.
    assert "Matrix: 4 jobs (2 aliases, 2 repeats each), 2/4 trials passed" in run.output


def test_a_matrix_of_only_controls_gates_on_them_once(tmp_path, monkeypatch):
    run, jobs = _matrix_gate_run(
        tmp_path,
        monkeypatch,
        "models:\n  nop: {agent: nop}\n  oracle: {agent: oracle}\n",
    )
    assert run.exit_code == 1, run.output
    for alias in ("nop", "oracle"):
        for i in (1, 2):
            assert (jobs / alias / f"trial-{i:02d}").is_dir(), run.output
    summary = json.loads((jobs / "matrix-summary.json").read_text())
    assert summary["gate"]["controls_counted"] is True
    assert summary["gate"]["controls_left_out"] == 0
    # One gate, over the whole matrix: 2 of 4 trials passed.
    assert run.output.count("Gate failed:") == 1
    assert "Gate failed: pass rate 0.50 < --fail-under 0.9" in run.output
    assert "(every trial is a control run, so they are counted)" in run.output


# ── Round 2, review L-F1: list and inspect name tasks alike ────────────────


def test_list_and_inspect_count_an_unfinished_jobs_tasks_alike(tmp_path: Path):
    job = tmp_path / "jobs" / "intr" / "2026-10-19__09-00-00"
    job.mkdir(parents=True)
    (job / "evaluation.json").write_text(
        json.dumps({"schema_version": 1, "planned_tasks": ["hello", "b"], "config": {}})
    )
    # Two attempts of one task, in folders whose names are not the task's.
    for folder in ("hello-1__abcd1234", "hello-2__efgh5678"):
        trial = _trial(job, "hello", reward=1.0, agent="oracle", model=None)
        trial.rename(job / folder)
    inspected = _invoke("eval", "inspect", str(job))
    assert "Not finished: 1 of 2 planned tasks finished" in inspected.output
    listed = _invoke("eval", "list", str(tmp_path / "jobs"))
    assert "not finished: 1 of 2 tasks have a result" in listed.output, listed.output
