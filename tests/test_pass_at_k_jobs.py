"""pass@k / pass^k / solve rates in bf.load_job, bf.compare, `bench eval
metrics`, `bench eval compare`, summary.json and matrix-summary.json.

Repeated trials of one model live in sibling job folders (``--matrix
--trials`` writes ``<alias>/trial-NN/<job>/``), so a task's samples are its
trials across those folders; retries inside one job stay one sample (the
``attempts="best"`` rule). Controls are left out and unscored trials do not
count as failures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import benchflow as bf
from benchflow.cli.main import app
from tests.test_python_sdk_load_job import _trial


def _trials(root: Path, per_trial: list[dict[str, float | None]]) -> Path:
    """``root/trial-NN/job`` folders, one per dict of task -> reward."""
    for i, rewards in enumerate(per_trial, start=1):
        job = root / f"trial-{i:02d}" / "job"
        for task, reward in rewards.items():
            _trial(
                job, task, reward=reward, error=None if reward is not None else "boom"
            )
        _trial(job, "zz-oracle", agent="oracle", model=None, suffix="0000oracl")
    return root


ARM = [
    {"t1": 1.0, "t2": 0.0},
    {"t1": 0.0, "t2": 0.0},
    {"t1": 1.0, "t2": None},
]


def test_job_solve_rates_pool_trials_across_job_folders(tmp_path: Path) -> None:
    job = bf.load_job(_trials(tmp_path / "arm", ARM))
    rates = job.solve_rates()
    # t1: n=3 c=2; t2: n=2 c=0 (its unscored trial is not a failure)
    assert rates.min_trials_per_task == 2 and rates.max_trials_per_task == 3
    assert [p.k for p in rates.at_k] == [1, 2]
    assert rates.get(1).pass_at_k == pytest.approx((2 / 3 + 0) / 2)
    assert rates.get(2).pass_at_k == pytest.approx((1.0 + 0) / 2)
    assert rates.get(2).pass_hat_k == pytest.approx((1 / 3 + 0) / 2)
    assert rates.unscored == 1
    assert rates.controls_excluded == 3
    three = job.solve_rates(ks=[3])
    assert three.get(3).tasks == 1 and three.get(3).tasks_short == 1


def test_job_solve_rates_threshold_and_controls(tmp_path: Path) -> None:
    job = bf.load_job(_trials(tmp_path / "arm", [{"t1": 0.6}, {"t1": 0.2}]))
    assert job.solve_rates().solve_rate == 0.0
    assert job.solve_rates(solve_threshold=0.5).solve_rate == pytest.approx(0.5)
    with_controls = job.solve_rates(include_controls=True)
    assert with_controls.tasks == 2 and with_controls.controls_excluded == 0


def test_job_document_carries_solve_rates(tmp_path: Path) -> None:
    doc = bf.load_job(_trials(tmp_path / "arm", ARM)).to_json_dict()
    assert doc["solve_rates"]["pass_at_k"]["2"] == pytest.approx(0.5)


def test_compare_reports_both_sides(tmp_path: Path) -> None:
    a = _trials(tmp_path / "a", ARM)
    b = _trials(tmp_path / "b", [{"t1": 1.0, "t2": 1.0}] * 3)
    cmp = bf.compare(a, b, ks=[1, 2], on_mismatch="ignore")
    assert cmp.solve_rates_a.get(2).pass_at_k == pytest.approx(0.5)
    assert cmp.solve_rates_b.get(2).pass_hat_k == 1.0
    md = cmp.to_markdown()
    assert "pass@2" in md and "pass^2" in md
    doc = cmp.to_json_dict()
    assert doc["b"]["solve_rates"]["pass_hat_k"]["2"] == 1.0


def test_cli_metrics_prints_pass_at_k(tmp_path: Path) -> None:
    root = _trials(tmp_path / "arm", ARM)
    runner = CliRunner()
    out = runner.invoke(
        app, ["eval", "metrics", str(root), "--json", "--k", "1", "--k", "3"]
    )
    assert out.exit_code == 0, out.output
    rates = json.loads(out.stdout)["solve_rates"]
    assert rates["ks"] == [1, 3]
    assert rates["tasks_short_at_k"]["3"] == 1
    assert any("n < k" in c for c in rates["caveats"])
    table = runner.invoke(
        app, ["eval", "metrics", str(root), "--solve-threshold", "0.5"]
    )
    assert table.exit_code == 0, table.output
    assert "pass@2" in table.output and "reward >= 0.5" in table.output


def test_cli_compare_takes_k_and_threshold(tmp_path: Path) -> None:
    a = _trials(tmp_path / "a", ARM)
    b = _trials(tmp_path / "b", ARM)
    out = CliRunner().invoke(
        app,
        [
            "eval",
            "compare",
            str(a),
            str(b),
            "--json",
            "--k",
            "2",
            "--solve-threshold",
            "0.5",
        ],
    )
    assert out.exit_code == 0, out.output
    doc = json.loads(out.stdout)
    assert doc["a"]["solve_rates"]["ks"] == [2]
    assert doc["a"]["solve_rates"]["success_rule"] == "reward >= 0.5"


def test_summary_json_has_solve_rates() -> None:
    from benchflow._utils.evaluation_results import solve_rate_summary

    block = solve_rate_summary(
        {
            "a": {"task_name": "a", "rewards": {"reward": 1.0}},
            "b": {"task_name": "b", "rewards": {"reward": 0.0}},
            "c": {"task_name": "c", "rewards": None, "error": "boom"},
        }
    )["solve_rates"]
    assert block["pass_at_k"] == {"1": 0.5}
    assert block["unscored"] == 1
    assert block["trials"] == 2


def test_matrix_summary_pools_trials_per_alias(tmp_path: Path) -> None:
    from benchflow.cli.eval_artifacts import matrix_solve_rates

    _trials(tmp_path / "haiku", ARM)
    block = matrix_solve_rates(tmp_path, ["haiku", "missing"])
    assert block["haiku"]["pass_at_k"]["2"] == pytest.approx(0.5)
    assert block["missing"] is None
