"""Hard per-job budget: --max-sandbox-seconds / --max-cost-usd and bf.Budget."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import benchflow as bf
from tests.e2e import harness as h


@pytest.fixture(scope="module")
def three_oracle_tasks(tasks_root: Path) -> Path:
    root = tasks_root / "budget-oracle"
    for i in range(3):
        h.write_task(root, f"budget-{i}")
    return root


def test_cli_sandbox_seconds_cap_cancels_and_skips(
    sandbox: str, three_oracle_tasks: Path, jobs_root: Path, ledger: h.Ledger
):
    """A 20 s cap stops a one-at-a-time job during its first trial."""
    job = jobs_root / "budget-seconds"
    run = h.bench(
        "eval", "run", "--tasks-dir", three_oracle_tasks, "--agent", "oracle",
        "--sandbox", sandbox, "--jobs-dir", jobs_root, "--job-name", job.name,
        "--concurrency", "1", "--max-sandbox-seconds", "20",
        "--summary-out", jobs_root / "budget-seconds.run-summary.json", "--quiet",
        log=jobs_root / "budget-seconds.log",
    )  # fmt: skip
    ledger.record("--max-sandbox-seconds 20 over 3 tasks, concurrency 1", surface="CLI",
                  seconds=run.seconds, job_dir=job)  # fmt: skip
    summary = h.read_json(job / "summary.json")
    budget = summary["budget"]
    assert budget["stopped"] is True, budget
    assert "sandbox" in (budget["reason"] or "")
    stopped = len(budget["cancelled"]) + len(budget["not_started"])
    assert stopped >= 2, budget
    # Cancelled and not-started trials are not failures.
    assert summary["failed"] == 0 and summary["errored"] == 0, summary
    for name in budget["cancelled"]:
        assert not list(job.glob(f"{name}__*/result.json"))
    doc = h.read_json(jobs_root / "budget-seconds.run-summary.json")
    h.validate(doc, "benchflow-run-summary.v1.schema.json")
    assert doc["budget"]["stopped"] is True
    # Resume runs the rest (under a generous cap) and counts what was spent.
    run = h.bench(
        "eval", "run", "--tasks-dir", three_oracle_tasks, "--agent", "oracle",
        "--sandbox", sandbox, "--jobs-dir", jobs_root, "--job-name", job.name,
        "--concurrency", "3", "--max-sandbox-seconds", str(h.cap_seconds()), "--quiet",
        log=jobs_root / "budget-seconds.resume.log",
    )  # fmt: skip
    ledger.record(
        "resume after a budget stop", surface="CLI", seconds=run.seconds, job_dir=job
    )
    h.assert_exit(run, 0)
    summary = h.read_json(job / "summary.json")
    assert summary["passed"] == 3, summary
    assert summary["budget"]["spent"]["sandbox_seconds"] > 20


def test_cli_cost_cap_with_a_priced_agent(
    sandbox: str, tasks_root: Path, jobs_root: Path, ledger: h.Ledger
):
    """The fake model's calls are priced by the gateway; a tiny USD cap stops the job."""
    root = tasks_root / "budget-cost"
    for i in range(3):
        h.write_fake_llm_task(root, f"cost-{i}", script="hello-pass")
    job = jobs_root / "budget-cost"
    run = h.bench(
        "eval", "run", "--tasks-dir", root, *h.fake_agent_args("proxy"),
        "--sandbox", sandbox, "--jobs-dir", jobs_root, "--job-name", job.name,
        "--concurrency", "1", "--max-cost-usd", "0.0001",
        "--max-sandbox-seconds", str(h.cap_seconds()), "--quiet",
        log=jobs_root / "budget-cost.log",
    )  # fmt: skip
    ledger.record("--max-cost-usd 0.0001 over 3 fake-agent tasks", surface="CLI",
                  seconds=run.seconds, job_dir=job)  # fmt: skip
    budget = h.read_json(job / "summary.json")["budget"]
    assert budget["stopped"] is True, budget
    assert (
        "cost" in (budget["reason"] or "") or "usd" in (budget["reason"] or "").lower()
    )
    assert budget["spent"]["cost_usd"] > 0.0001
    assert len(budget["not_started"]) == 2, budget


def test_python_budget(
    sandbox: str, three_oracle_tasks: Path, jobs_root: Path, ledger: h.Ledger
):
    started = time.monotonic()
    ev = bf.Evaluation(
        tasks_dir=three_oracle_tasks,
        jobs_dir=jobs_root,
        job_name="py-budget",
        config=bf.EvaluationConfig(agent="oracle", environment=sandbox, concurrency=1),
        budget=bf.Budget(max_sandbox_seconds=20),
    )
    result = ev.run_sync()
    ledger.record("Evaluation(budget=Budget(max_sandbox_seconds=20))", surface="Python",
                  seconds=time.monotonic() - started, job_dir=Path(result.job_dir))  # fmt: skip
    budget = result.budget
    assert budget is not None
    data = budget if isinstance(budget, dict) else budget.to_dict()
    assert data["stopped"] is True, data
    assert result.failed == 0
