"""Guard the fix and 98eec233's RAW-ABS-CONF3-1/2 regressions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import benchflow as bf
from benchflow.cli.eval_artifacts import matrix_solve_rates
from benchflow.cli.main import app
from benchflow.evaluation import Evaluation, EvaluationConfig

LEGACY_ALIASES = [
    "base",
    ".",
    "",
    "./",
    "nested/alias",
    "./nested//alias",
    "base--cae662172fd450bb",
    "matrix-ownership.json",
    "模型😀",
]


def _tasks(tmp_path):
    tasks = tmp_path / "tasks"
    (tasks / "task-a").mkdir(parents=True)
    (tasks / "task-a" / "task.toml").write_text('version = "1.0"\n')
    return tasks


def _result(evaluation):
    evaluation._write_evaluation_record()
    rollout = evaluation.job_dir / "task-a__abc"
    if not rollout.exists():
        rollout.mkdir()
        (rollout / "result.json").write_text(
            json.dumps({"task_name": "task-a", "rewards": {"reward": 1.0}})
        )
    return SimpleNamespace(
        job_name=evaluation._job_name,
        passed=1,
        failed=0,
        errored=0,
        verifier_errored=0,
        total=1,
        score=1.0,
        score_excl_errors=1.0,
    )


def _invoke(tasks, jobs, matrix):
    return CliRunner().invoke(
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
            "--trials",
            "2",
        ],
    )


@pytest.mark.parametrize("alias", LEGACY_ALIASES)
@pytest.mark.parametrize(
    "root_form", ["relative", "absolute", "normalized", "relocated"]
)
@pytest.mark.parametrize("completed", [True, False])
def test_base_layout_resumes_repeats_and_reads_unchanged(
    tmp_path, monkeypatch, alias, root_form, completed
):
    """Guard CONF3-1: root spelling or missing summary cannot change a base layout."""
    monkeypatch.chdir(tmp_path)
    tasks = _tasks(tmp_path)
    jobs = Path("jobs")
    expected = {}
    aliases = [alias, "other-arm", "third-arm"]
    for arm in aliases:
        for trial in (1, 2):
            evaluation = Evaluation(
                tasks_dir=tasks,
                jobs_dir=jobs / arm / f"trial-{trial:02d}",
                job_name="2026-10-16__01-02-03",
                config=EvaluationConfig(agent="oracle", model="dummy-model"),
            )
            _result(evaluation)
            expected[(arm, trial)] = evaluation.job_dir
    if completed:
        (jobs / "matrix-summary.json").write_text(
            json.dumps(
                {
                    "runs": [
                        {"alias": arm, "trial": trial, "jobs_dir": str(job.parent)}
                        for (arm, trial), job in expected.items()
                    ]
                }
            )
        )
    expected_rates = {
        arm: bf.load_job(jobs / arm).solve_rates().to_dict() for arm in aliases
    }
    if root_form == "relocated":
        jobs.rename("moved-jobs")
        jobs = Path("moved-jobs")
        expected = {
            key: jobs / path.relative_to("jobs") for key, path in expected.items()
        }
    root = jobs
    if root_form == "absolute":
        root = jobs.absolute()
    elif root_form == "normalized":
        root = jobs.absolute() / ".." / jobs.name
    before = {
        path.resolve(): (path / "task-a__abc/result.json").read_bytes()
        for path in expected.values()
    }
    seen = []

    async def resume_only(self):
        path = self.job_dir.resolve()
        assert path in before, "saved base job was silently replaced"
        assert (path / "task-a__abc/result.json").read_bytes() == before[path]
        seen.append(path)
        return _result(self)

    monkeypatch.setattr(Evaluation, "run", resume_only)
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "models:\n"
        + "".join(
            f"  {json.dumps(arm, ensure_ascii=False)}: dummy-model\n" for arm in aliases
        )
    )
    for _ in range(2):
        result = _invoke(tasks, root, matrix)
        assert result.exit_code == 0, (result.output, result.exception)
        rates = matrix_solve_rates(root, aliases)
        assert rates == expected_rates
        assert not (root / "matrix-ownership.json").is_file()
    assert sorted(seen) == sorted(list(before) * 2)
    assert {
        p.resolve(): (p / "task-a__abc/result.json").read_bytes()
        for p in expected.values()
    } == before


def test_add_old_encoded_looking_alias_after_completed_arm(tmp_path, monkeypatch):
    """Guard CONF3-2: adding a literal old encoded-looking name cannot steal base."""
    tasks = _tasks(tmp_path)
    jobs = tmp_path / "jobs"
    matrix = tmp_path / "matrix.yaml"
    seen = []

    async def run(self):
        seen.append(self.job_dir)
        return _result(self)

    monkeypatch.setattr(Evaluation, "run", run)
    matrix.write_text("models:\n  base: dummy-model\n")
    first = _invoke(tasks, jobs, matrix)
    assert first.exit_code == 0, (first.output, first.exception)
    original = list(seen)
    matrix.write_text(
        "models:\n  base: dummy-model\n  base--cae662172fd450bb: dummy-model\n"
    )
    for _ in range(2):
        seen.clear()
        result = _invoke(tasks, jobs, matrix)
        assert result.exit_code == 0, (result.output, result.exception)
        assert seen[:2] == original
        assert all(p.parent.parent == jobs / "base--cae662172fd450bb" for p in seen[2:])
        assert all(
            v["trials"] == 2
            for v in matrix_solve_rates(
                jobs, ["base", "base--cae662172fd450bb"]
            ).values()
        )
