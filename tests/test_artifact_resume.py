"""Guard RAW-ABS-CONF-1: the 13f0d6a2/7466a059 naming fix must retain resume."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from benchflow.cli.eval_artifacts import matrix_solve_rates
from benchflow.cli.main import app
from benchflow.evaluation import Evaluation, EvaluationConfig
from benchflow.loop_sweep import run_sweep


def _old_sweep_name(model, loop):
    """The persisted layout written by 8c1d06c6, not the candidate's mapper."""
    label = loop or "single-shot"
    safe_loop = label.replace(":", "_").replace(",", "_").replace("=", "")
    safe_model = model.replace("/", "_")
    digest = hashlib.sha256(f"{model}\0{label}".encode()).hexdigest()[:8]
    return f"model={safe_model}__loop={safe_loop}__{digest}"


def _job(tasks, jobs, name, model, loop=None):
    evaluation = Evaluation(
        tasks_dir=tasks,
        jobs_dir=jobs,
        job_name=name,
        config=EvaluationConfig(agent="oracle", model=model, loop_strategy=loop),
    )
    evaluation._write_evaluation_record()
    trial = evaluation.job_dir / "task-a__abc"
    trial.mkdir()
    (trial / "result.json").write_text(
        json.dumps({"task_name": "task-a", "rewards": {"reward": 1.0}})
    )
    return evaluation.job_dir


def _matrix(tmp_path, alias, *, summary=True):
    tasks = tmp_path / "tasks"
    (tasks / "task-a").mkdir(parents=True)
    (tasks / "task-a" / "task.toml").write_text('version = "1.0"\n')
    jobs = tmp_path / "jobs"
    trial = jobs / alias / "trial-01"
    job = _job(tasks, trial, "2026-10-16__01-02-03", "dummy-model")
    if summary:
        (jobs / "matrix-summary.json").write_text(
            json.dumps({"runs": [{"alias": alias, "trial": 1, "jobs_dir": str(trial)}]})
        )
    return tasks, jobs, job


@pytest.mark.parametrize(
    "alias",
    [
        "base",
        "Base-2",
        "org_model.v1",
        "base model",
        "matrix-ownership.json",
        ".base",
        "-base",
        "模型😀",
        "a" * 240,
    ],
)
@pytest.mark.parametrize("summary", [True, False])
def test_matrix_existing_base_layout_is_resumed_and_included(
    tmp_path, monkeypatch, alias, summary
):
    """Guard RAW-ABS-CONF-1 for completed and interrupted 8c1d06c6 matrices."""
    tasks, jobs, job = _matrix(tmp_path, alias, summary=summary)
    before = (job / "task-a__abc" / "result.json").read_bytes()
    matrix = tmp_path / "models.yaml"
    matrix.write_text(
        "models:\n  " + json.dumps(alias, ensure_ascii=False) + ": dummy-model\n"
    )
    calls = []

    async def resume_only(self):
        # Production __init__ has already resolved the existing timestamp job.
        assert self.job_dir == job, "completed base job was silently restarted"
        assert (self.job_dir / "task-a__abc" / "result.json").read_bytes() == before
        calls.append(self.job_dir)
        return SimpleNamespace(
            job_name=self._job_name,
            passed=1,
            failed=0,
            errored=0,
            verifier_errored=0,
            total=1,
            score=1.0,
            score_excl_errors=1.0,
        )

    monkeypatch.setattr(Evaluation, "run", resume_only)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(tasks),
            "--jobs-dir",
            str(jobs),
            "--agent",
            "oracle",
            "--matrix",
            str(matrix),
            "--run-config-out",
            str(tmp_path / "configs" / "run.json"),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == [job]
    assert [p.name for p in jobs.iterdir() if p.is_dir()] == [alias]
    rates = matrix_solve_rates(jobs, [alias])[alias]
    assert rates is not None and rates["tasks"] == 1
    written = json.loads((jobs / "matrix-summary.json").read_text())
    assert written["runs"][0]["jobs_dir"] == str(job.parent)
    assert written["solve_rates"][alias] == rates
    assert (tmp_path / "configs" / alias / "trial-01" / "run.json").is_file()


@pytest.mark.parametrize("model", ["model", "org/model.v1", "foo_bar"])
@pytest.mark.parametrize("loop", [None, "verify-retry:k=3"])
async def test_sweep_existing_base_layout_keeps_both_directory_levels(
    tmp_path, monkeypatch, model, loop
):
    """Guard RAW-ABS-CONF-1: the explicit inner job name must resume too."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    jobs = tmp_path / "jobs"
    legacy = _old_sweep_name(model, loop)
    old = jobs / legacy
    job = _job(tasks, old, legacy, model, loop)
    summary = {"passed": 1, "total": 1, "score_ratio": 1.0}
    (old / "summary.json").write_text(json.dumps(summary))
    (job / "summary.json").write_text(json.dumps(summary))
    seen = []

    async def resume_only(self):
        assert self.job_dir == job, "completed sweep was silently restarted"
        assert (self.job_dir / "task-a__abc" / "result.json").is_file()
        seen.append(self.job_dir)

    monkeypatch.setattr(Evaluation, "run", resume_only)
    matrix = await run_sweep(
        tasks_dir=tasks,
        jobs_dir=jobs,
        base_config=EvaluationConfig(agent="oracle"),
        models=[model],
        loops=[loop],
    )
    assert seen == [job]
    assert matrix["cells"][0]["passed"] == 1
    assert [p.name for p in jobs.iterdir() if p.is_dir()] == [legacy]
