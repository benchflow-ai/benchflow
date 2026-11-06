"""Every reader counts by the one scoring rule, on the conformance fixture.

summary.json, the end-of-run line, report.md, ``bench eval metrics`` (text
and ``--json``), ``bench results`` (one job, a folder of jobs), pass@k,
``print(job)`` and the run card's after-run lines read the same trials. Each
must give the fixture's figures: report.md's interval treated a task's
trials as independent, the job listing used summary.json's audit count, and
none of them split trials that ran different settings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import benchflow as bf
from benchflow._utils.scoring import score_summary_fields
from benchflow.cli.main import app
from benchflow.report import build_report
from benchflow.run_checks import after_run_lines

FIXTURE = Path(__file__).parent / "fixtures" / "scoring" / "benchflow.score.v1.json"
CASES = [
    "every outcome, with repeated trials",
    "provider and harness endings left after retries count 0, by who ended them",
    "a reward decides, whatever error the trial also records",
]
WIDE = {"COLUMNS": "300"}


def _case(name: str) -> dict:
    doc = json.loads(FIXTURE.read_text())
    return next(c for c in doc["cases"] if c["name"] == name)


def _batch(root: Path, case: dict) -> tuple[Path, list[dict]]:
    """The case's trials as a folder of independent rollouts (every trial a
    sample, as bf.run_batch writes them), each with result.json and
    config.json."""
    folder = root / "batch"
    rows = []
    for i, trial in enumerate(case["trials"]):
        path = folder / f"{trial['task']}__{i:08x}"
        path.mkdir(parents=True)
        result = {
            "task_name": trial["task"],
            "rollout_name": path.name,
            "agent": "claude-agent-acp",
            "model": "claude-haiku-4-5",
            "error": None,
            "verifier_error": None,
            "n_tool_calls": 2,
            "started_at": "2026-11-04T12:00:00+00:00",
            "finished_at": "2026-11-04T12:01:00+00:00",
            **trial["result"],
        }
        path.joinpath("result.json").write_text(json.dumps(result))
        path.joinpath("config.json").write_text(
            json.dumps(
                {
                    "benchflow_version": "0.10.0",
                    "agent": "claude-agent-acp",
                    "model": "claude-haiku-4-5",
                    "reasoning_effort": "high",
                    "harness_mode": "acp",
                }
            )
        )
        rows.append(result)
    return folder, rows


def _plain(text: str) -> str:
    return " ".join(re.sub("[─-╿]", " ", text).split())


def _interval(want: dict) -> str:
    low, high = want["ci95"]
    how = "clustered by task" if want["ci_method"] == "wilson-clustered" else ""
    return f"95% interval {low:.1%} to {high:.1%}, {how}".rstrip(", ")


@pytest.mark.parametrize("name", CASES)
def test_every_reader_gives_the_fixture_s_figures(tmp_path: Path, name: str) -> None:
    case = _case(name)
    want = case["expected"]
    folder, rows = _batch(tmp_path, case)
    interval = _interval(want)
    line = f"{want['passed']} of {want['n']} trials"

    # summary.json's block.
    summary = score_summary_fields(rows)
    assert summary["scoring"]["n"] == want["n"]
    assert summary["score_ratio"] == pytest.approx(want["score"])
    assert summary["scoring"]["ci95"] == pytest.approx(want["ci95"])
    assert summary["scoring"]["endings"] == want["endings"]

    # bench eval metrics --json: the same figures, pass@k beside them.
    ks = [arg for k in want["pass_at_k"] for arg in ("--k", k)]
    doc = json.loads(
        CliRunner().invoke(app, ["eval", "metrics", str(folder), "--json", *ks]).stdout
    )
    assert doc["scoring_rule"]["id"] == "benchflow.score.v1"
    scoring = doc["scoring"]
    assert (scoring["n"], scoring["passed"], scoring["n_tasks"]) == (
        want["n"],
        want["passed"],
        want["n_tasks"],
    )
    assert scoring["score"] == pytest.approx(want["score"])
    assert scoring["score_excl_errors"] == pytest.approx(want["score_excl_errors"])
    assert scoring["se"] == pytest.approx(want["se"])
    assert scoring["ci95"] == pytest.approx(want["ci95"])
    assert scoring["outcomes"] == want["outcomes"]
    assert scoring["endings"] == want["endings"]
    assert doc["score_ratio"] == pytest.approx(want["score"])
    assert scoring["settings"]["mixed"] is False
    for k, value in want["pass_at_k"].items():
        assert doc["solve_rates"]["pass_at_k"][k] == pytest.approx(value)
    # The solve rate's interval is the rule's.
    assert doc["solve_rates"]["solve_rate_interval"] == pytest.approx(want["ci95"])

    # bench eval metrics, as text.
    text = _plain(
        CliRunner().invoke(app, ["eval", "metrics", str(folder)], env=WIDE).output
    )
    assert f"Score {want['score']:.1%} ({line}); {interval}" in text
    assert "Scoring rule benchflow.score.v1" in text

    # report.md.
    report = build_report([folder], "fixture")
    assert f"{want['passed']} passed ({want['score']:.1%}; {interval})" in report
    assert "Counted by the scoring rule benchflow.score.v1" in report

    # bench results on the folder (bench eval inspect).
    shown = _plain(CliRunner().invoke(app, ["results", str(folder)], env=WIDE).output)
    assert (
        f"Score by benchflow.score.v1: {want['score']:.1%} ({line}; {interval}), "
        f"{want['n_tasks']} tasks"
    ) in shown

    # pass@k through the library, and print(job).
    job = bf.load_job(folder)
    rates = job.solve_rates(ks=[int(k) for k in want["pass_at_k"]])
    for k, value in want["pass_at_k"].items():
        assert rates.get(int(k)).pass_at_k == pytest.approx(value)
    assert rates.interval == pytest.approx(tuple(want["ci95"]))

    # The run card's after-run lines.
    lines = after_run_lines({"scoring": summary["scoring"]})
    assert (
        f"  Score:     {line} passed ({want['score']:.1%}; {interval}) by "
        "benchflow.score.v1"
    ) in lines


def test_the_job_listing_counts_the_rule_s_passes(tmp_path: Path) -> None:
    """A scored trial whose verifier then recorded an error passed by the
    rule; the listing read summary.json's audit count and said 0/2."""
    job = tmp_path / "jobs" / "j"
    rows = [
        {
            "task_name": "a",
            "rewards": {"reward": 1.0},
            "verifier_error": "[solver-preserved] post-solver cleanup failed: x",
        },
        {"task_name": "b", "rewards": {"reward": 0.0}},
    ]
    for row in rows:
        trial = job / f"{row['task_name']}__1"
        trial.mkdir(parents=True)
        trial.joinpath("result.json").write_text(json.dumps(row))
    job.joinpath("summary.json").write_text(
        json.dumps({"job_name": "j", **score_summary_fields(rows)})
    )
    out = CliRunner().invoke(app, ["eval", "list", str(tmp_path / "jobs")], env=WIDE)
    assert out.exit_code == 0, out.output
    assert "1/2 (50.0%)" in out.output and "0/2" not in out.output


def test_a_mixed_job_is_split_by_every_reader(tmp_path: Path) -> None:
    """The resumed job of two routes, two Claude Code releases and two
    BenchFlow releases under one model label: no reader pools it."""
    from tests.test_setting_rows import BEDROCK, DIRECT, _resumed_job

    job = _resumed_job(tmp_path)
    doc = json.loads(
        CliRunner().invoke(app, ["eval", "metrics", str(job), "--json"]).stdout
    )
    assert doc["score"] is None and doc["score_ratio"] is None
    assert doc["solve_rates"] is None and "mixed settings" in doc["headline_withheld"]
    rows = doc["scoring"]["settings"]["rows"]
    assert [r["route"] for r in rows] == [DIRECT, BEDROCK]
    assert [r["scoring"]["n"] for r in rows] == [3, 3]
    assert all(r["solve_rates"]["pass_at_k"]["1"] is not None for r in rows)

    result = CliRunner().invoke(app, ["eval", "metrics", str(job)], env=WIDE)
    text = _plain(result.output)
    assert "none for the whole folder: it holds 2 settings" in text
    assert f"via {DIRECT}, effort high, BenchFlow 0.10.0: 1 of 3 trials passed" in text
    assert "mixed settings: these trials ran 2 settings" in text

    report = build_report([job], "mixed")
    assert "6 trials of 6 tasks in 2 settings." in report
    assert (
        f"via {BEDROCK}, effort high, BenchFlow 0.10.1: 1 of 3 trials passed" in report
    )

    shown = _plain(CliRunner().invoke(app, ["results", str(job)], env=WIDE).output)
    assert "By setting, each counted by benchflow.score.v1:" in shown
    assert "mixed settings" in shown

    markdown = bf.load_job(job).to_markdown()
    assert "- Warning: mixed settings: these trials ran 2 settings" in markdown

    from benchflow.setting_groups import summary_settings

    summary = score_summary_fields(
        [json.loads(p.read_text()) for p in sorted(job.glob("*/result.json"))]
    )
    summary["scoring"]["settings"] = summary_settings(job, [summary])
    lines = after_run_lines({"scoring": summary["scoring"]})
    assert any(line.startswith("  Settings:  mixed settings:") for line in lines)


def test_the_run_and_bench_results_name_the_same_trials_alike(tmp_path: Path) -> None:
    """A verifier-errored trial was "1 unscored" on the run's Outcomes line
    and "1 verifier errors, 0 unscored" in bench results: the two lines split
    the same trials differently. Both now give the rule's classes with the
    same counts."""
    import io
    from types import SimpleNamespace

    from rich.console import Console

    import benchflow.cli._shared as shared

    job = tmp_path / "jobs" / "run"
    rows = {
        "right": {"rewards": {"reward": 1.0}},
        "wrong": {"rewards": {"reward": 0.0}},
        "crashed": {
            "rewards": None,
            "verifier_error": "verifier crashed: No reward file found at reward.txt",
        },
        "stopped": {
            "rewards": None,
            "error": "Agent prompt exceeded wall-clock budget 600s",
            "error_category": "timeout",
        },
    }
    results = {}
    for name, fields in rows.items():
        trial = job / f"{name}__0001"
        trial.mkdir(parents=True)
        row = {
            "task_name": name,
            "rollout_name": trial.name,
            "agent": "claude-agent-acp",
            "model": "claude-haiku-4-5",
            "n_tool_calls": 2,
            "error": None,
            "verifier_error": None,
            **fields,
        }
        trial.joinpath("result.json").write_text(json.dumps(row))
        results[name] = row
    words = "1 unscored (1 verifier error), 1 errored"

    console = Console(file=io.StringIO(), width=300)
    original = shared.console
    shared.console = console
    try:
        shared._report_outcomes(SimpleNamespace(results=results), job)
    finally:
        shared.console = original
    assert (
        f"Outcomes: 2 scored (1 passed, 1 failed), {words}" in console.file.getvalue()
    )

    shown = _plain(CliRunner().invoke(app, ["results", str(job)], env=WIDE).output)
    assert f"1/4 attempted passed (25%), 1/2 scored passed (50%), {words}" in shown


def test_a_trial_folder_that_cannot_be_read_is_named_not_dropped(
    tmp_path: Path,
) -> None:
    """bf.load_job skipped a trial whose result.json would not parse with a
    log line only, so every figure silently left it out."""
    folder, _ = _batch(tmp_path, _case(CASES[2]))
    broken = folder / "lost__0001"
    broken.mkdir()
    broken.joinpath("result.json").write_text("{not json")
    job = bf.load_job(folder)
    assert job.unreadable == [broken] and len(job.trials) == 5
    warning = f"1 trial folder could not be read (first: {broken}), so it is in none"
    assert warning in job.to_markdown()
    assert warning in build_report([folder], "x")
    doc = json.loads(
        CliRunner().invoke(app, ["eval", "metrics", str(folder), "--json"]).stdout
    )
    assert doc["scoring"]["unreadable"] == [str(broken)]
    text = CliRunner().invoke(app, ["eval", "metrics", str(folder)], env=WIDE).output
    assert warning in " ".join(text.split())
    shown = CliRunner().invoke(app, ["results", str(folder)], env=WIDE).output
    assert warning in " ".join(shown.split())


def test_the_scored_only_row_counts_a_refusal_in_its_n(tmp_path: Path) -> None:
    """bench eval metrics printed the error-excluding figure, which keeps a
    refusal as a failure (50.0%), beside "2 of 3 scored trials", which left
    it out: the percentage and its count disagreed (seen on a real run whose
    verifier wrote no usable reward)."""
    folder, _ = _batch(tmp_path, {"trials": [
        {"task": "a", "result": {"rewards": {"reward": 1.0}}},
        {"task": "b", "result": {"rewards": {"reward": 1.0}}},
        {"task": "c", "result": {"rewards": {"reward": 0.0}}},
        {"task": "d", "result": {"rewards": None, "verifier_error":
            "verifier wrote no usable reward: Failed to parse rewards"}},
        {"task": "e", "result": {"rewards": None, "error": "ACP error -32603"}},
    ]})  # fmt: skip
    text = _plain(
        CliRunner().invoke(app, ["eval", "metrics", str(folder)], env=WIDE).output
    )
    assert "Score, scored only 50.0% (2 of 4 scored or refused trials)" in text
