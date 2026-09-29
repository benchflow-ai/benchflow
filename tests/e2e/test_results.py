"""Reading results: metrics, inspect, compare, pass@k; bf.load_job/load_trial/compare.

Solve rates leave control runs (oracle, nop) out, so the repeated trials here
use ``claude-agent-acp`` driven by the scripted fake model: a real agent run
with a real ACP session, scored by the task verifier, with no model provider.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import benchflow as bf
from tests.e2e import harness as h

PARTIAL_TEST = """#!/bin/bash
if [ "$(tr -d '\\n' < /app/hello.txt 2>/dev/null)" = "Hello, world!" ]; then
  echo 0.5 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""
TRIALS = 3


def job_dirs(matrix_dir: Path) -> list[Path]:
    """The job folders of a ``--matrix`` alias: ``trial-NN/<job>/``."""
    return sorted(p.parent for p in matrix_dir.glob("trial-*/*/results.jsonl"))


@pytest.fixture(scope="module")
def matrix_dir(
    sandbox: str, tasks_root: Path, jobs_root: Path, ledger: h.Ledger
) -> Path:
    """``--matrix --trials 3`` over a passing, a failing and a half-credit task."""
    root = tasks_root / "passk"
    h.write_fake_llm_task(root, "fake-pass", script="hello-pass")
    h.write_fake_llm_task(root, "fake-wrong", script="hello-wrong")
    h.write_fake_llm_task(root, "fake-half", script="hello-pass", test=PARTIAL_TEST)
    matrix = jobs_root / "matrix.yaml"
    matrix.write_text(f"models:\n  fake:\n    model: {h.FAKE_MODEL}\n")
    out = jobs_root / "matrix"
    if h.reuse() and (out / "matrix-summary.json").is_file():
        return out / "fake"
    run = h.bench(
        "eval", "run", "--tasks-dir", root, *h.fake_agent_args("proxy"),
        "--sandbox", sandbox, "--jobs-dir", out, "--matrix", matrix,
        "--trials", str(TRIALS), "--concurrency", "3",
        "--max-sandbox-seconds", str(h.cap_seconds()), "--max-cost-usd", "1", "--quiet",
        log=jobs_root / "matrix.log", timeout=2400,
    )  # fmt: skip
    ledger.record(
        f"eval run --matrix --trials {TRIALS} (fake agent, 3 tasks)",
        surface="CLI",
        seconds=run.seconds,
        result="pass" if run.returncode in (0, 1) else f"exit {run.returncode}",
    )
    h.assert_exit(run, 0, 1)
    return out / "fake"


def test_cli_metrics_pass_at_k(matrix_dir: Path, ledger: h.Ledger):
    started = time.monotonic()
    run, data = h.bench_json(
        "eval", "metrics", matrix_dir, "--k", "1", "--k", "3", "--k", "4", "--json"
    )
    h.assert_exit(run, 0)
    rates = data["solve_rates"]
    assert rates["tasks"] == 3
    assert rates["trials"] == 3 * TRIALS, rates
    assert rates["min_trials_per_task"] == TRIALS
    # fake-pass solves 3/3, fake-wrong 0/3, fake-half never reaches reward 1.
    assert rates["pass_at_k"]["1"] == pytest.approx(1 / 3)
    assert rates["pass_at_k"]["3"] == pytest.approx(1 / 3)
    assert rates["pass_hat_k"]["3"] == pytest.approx(1 / 3)
    # n = 3 < 4: every task is left out of k = 4, with a caveat.
    assert rates["pass_at_k"]["4"] is None
    assert rates["tasks_short_at_k"]["4"] == 3
    assert any("n < k" in c or "(n <" in c for c in rates["caveats"]), rates["caveats"]
    run, half = h.bench_json(
        "eval", "metrics", matrix_dir, "--k", "1", "--solve-threshold", "0.5", "--json"
    )
    h.assert_exit(run, 0)
    assert half["solve_rates"]["solve_rate"] == pytest.approx(2 / 3)
    ledger.record("eval metrics pass@k, pass^k, --solve-threshold", surface="CLI",
                  seconds=time.monotonic() - started)  # fmt: skip


def test_python_solve_rates_match_cli(matrix_dir: Path, ledger: h.Ledger):
    started = time.monotonic()
    job = bf.load_job(job_dirs(matrix_dir))
    rates = job.solve_rates(ks=[1, 3])
    assert rates.get(1).pass_at_k == pytest.approx(1 / 3)
    assert rates.get(3).pass_hat_k == pytest.approx(1 / 3)
    half = job.solve_rates(ks=[1], solve_threshold=0.5)
    assert half.solve_rate == pytest.approx(2 / 3)
    ledger.record(
        "Job.solve_rates", surface="Python", seconds=time.monotonic() - started
    )


def test_inspect_and_load_job_documents(matrix_dir: Path, ledger: h.Ledger):
    started = time.monotonic()
    job_dir = job_dirs(matrix_dir)[0]
    run, doc = h.bench_json("eval", "inspect", job_dir, "--json")
    h.assert_exit(run, 0)
    h.validate(doc, "benchflow-job.v1.schema.json")
    assert doc["kind"] == "benchflow.job"
    py = bf.load_job(job_dir).to_json_dict()
    h.validate(py, "benchflow-job.v1.schema.json")
    assert py["denominators"] == doc["denominators"]
    assert py["denominators"]["attempted"] == 3
    # One trial document, CLI and Python.
    trial_dir = h.trial_of(job_dir, "fake-pass")
    run, tdoc = h.bench_json("eval", "inspect", trial_dir, "--json")
    h.assert_exit(run, 0)
    h.validate(tdoc, "benchflow-trial.v1.schema.json")
    trial = bf.load_trial(trial_dir)
    assert trial.reward == 1.0 and trial.passed
    assert trial.agent == h.FAKE_AGENT
    tpy = trial.to_json_dict()
    h.validate(tpy, "benchflow-trial.v1.schema.json")
    # The proxy route records usage and a price.
    assert trial.total_tokens and trial.total_tokens > 0
    assert trial.cost_usd and trial.cost_usd > 0
    ledger.record("inspect --json / load_job / load_trial vs schemas", surface="CLI+Python",
                  seconds=time.monotonic() - started)  # fmt: skip


def test_compare_documents_and_mismatch(
    sandbox: str, matrix_dir: Path, tasks_root: Path, jobs_root: Path, ledger: h.Ledger
):
    started = time.monotonic()
    a, b = job_dirs(matrix_dir)[:2]
    run, doc = h.bench_json("eval", "compare", a, b, "--json", "--k", "1")
    h.assert_exit(run, 0)
    h.validate(doc, "benchflow-comparison.v1.schema.json")
    assert len(doc["rows"]) == 3
    assert (doc["a"]["label"], doc["b"]["label"]) == ("trial-01", "trial-02")
    report = bf.compare(a, b, ks=[1])
    py = report.to_json_dict()
    h.validate(py, "benchflow-comparison.v1.schema.json")
    assert {r["task"] for r in py["rows"]} == {r["task"] for r in doc["rows"]}
    # The same tasks with another harness (oracle) is a settings mismatch.
    oracle_job = jobs_root / "passk-oracle"
    if not (h.reuse() and (oracle_job / "summary.json").is_file()):
        run = h.bench(
            "eval", "run", "--tasks-dir", tasks_root / "passk", "--agent", "oracle",
            "--sandbox", sandbox, "--jobs-dir", jobs_root, "--job-name", oracle_job.name,
            "--concurrency", "3", "--max-sandbox-seconds", str(h.cap_seconds()), "--quiet",
            log=jobs_root / "passk-oracle.log",
        )  # fmt: skip
        ledger.record("eval run oracle over the pass@k tasks", surface="CLI",
                      seconds=run.seconds, job_dir=oracle_job)  # fmt: skip
    run = h.bench(
        "eval", "compare", a, oracle_job, "--include-controls", "--on-mismatch", "raise",
    )  # fmt: skip
    h.assert_exit(run, 1)
    assert "harness" in run.output
    run = h.bench(
        "eval", "compare", a, oracle_job, "--include-controls", "--on-mismatch", "raise",
        "--vary", "harness", "--vary", "model",
    )  # fmt: skip
    h.assert_exit(run, 0)
    assert "| passk-oracle |" in run.output or "passk-oracle" in run.output
    with pytest.raises(Exception, match="harness"):
        bf.compare(a, oracle_job, include_controls=True, on_mismatch="raise")
    # Control runs are left out by default: the oracle side has nothing scored.
    report = bf.compare(a, oracle_job, vary=("harness", "model"))
    assert report.to_json_dict()["b"]["denominators"]["attempted"] == 0
    # A missing path is a usage error.
    run = h.bench("eval", "compare", a, jobs_root / "does-not-exist")
    h.assert_exit(run, 2)
    ledger.record("compare --json / bf.compare / --on-mismatch raise / --vary", surface="CLI+Python",
                  seconds=time.monotonic() - started)  # fmt: skip
