"""Frozen-workspace restore, job listing and metrics over a shared jobs folder.

Regression scenarios for these defects:

- a frozen workspace lost the solver's file modes, so ``bench eval regrade``
  with the unchanged verifier failed a key the solver had made private;
- a verdict that changed although the task did not was reported as a plain
  ``pass->fail``;
- a batch trial records only its task folder's name, so ``regrade`` without
  ``--tasks-dir`` found no task although the job records its tasks folder;
- ``bench eval list jobs/`` showed one row for a folder of several jobs;
- ``bench eval metrics jobs/`` let one agent's pass hide another agent's fail;
- a task with a rubric and no reviewer model was refused with the solver's
  flag (``--model``) instead of ``--reviewer-model``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e import harness as h

PRIVATE_KEY_SOLVE = """#!/bin/bash
set -e
mkdir -p /app/ssl
printf 'not a real key\\n' > /app/ssl/server.key
chmod 600 /app/ssl/server.key
chmod 700 /app/ssl
"""
PRIVATE_KEY_TEST = """#!/bin/bash
mode=$(stat -c %a /app/ssl/server.key 2>/dev/null)
dir=$(stat -c %a /app/ssl 2>/dev/null)
echo "key mode $mode, dir mode $dir"
if [ "$mode" = "600" ] && [ "$dir" = "700" ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
"""
OUTSIDE_SOLVE = """#!/bin/bash
set -e
printf 'Hello, world!\\n' > /app/hello.txt
printf 'written outside the workspace\\n' > /opt/e2e-outside-marker
"""
OUTSIDE_TEST = """#!/bin/bash
if [ -f /opt/e2e-outside-marker ] && [ -f /app/hello.txt ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
"""
RUBRIC = {
    "criteria": [
        {
            "name": "greeting_present",
            "blocker": 1,
            "weight": 1,
            "description": "hello.txt exists",
            "guidance": "PASS when /app/hello.txt exists; FAIL otherwise.",
        },
        # A weighted rubric needs at least one scored (blocker: 0) criterion,
        # or the task is refused as invalid before the reviewer check runs.
        {
            "name": "greeting_exact",
            "blocker": 0,
            "weight": 1,
            "description": "hello.txt holds exactly the requested greeting",
            "guidance": "Score 1 for exactly 'Hello, world!', 0 otherwise.",
        },
    ]
}


@pytest.fixture(scope="module")
def restore_tasks(tasks_root: Path) -> Path:
    root = tasks_root / "restore"
    h.write_task(
        root, "e2e-private-key", solve=PRIVATE_KEY_SOLVE, test=PRIVATE_KEY_TEST
    )
    h.write_task(root, "e2e-outside-state", solve=OUTSIDE_SOLVE, test=OUTSIDE_TEST)
    return root


@pytest.fixture(scope="module")
def restore_jobs(
    sandbox: str, restore_tasks: Path, e2e_out: Path, ledger: h.Ledger
) -> Path:
    """An oracle job and a nop job over the same two tasks, in one jobs folder."""
    jobs = e2e_out / "jobs-restore"
    for agent in ("oracle", "nop"):
        job = jobs / f"restore-{agent}"
        if not h.needs_run(job):
            continue
        run = h.bench(
            "eval", "run",
            "--tasks-dir", restore_tasks,
            "--agent", agent,
            "--sandbox", sandbox,
            "--jobs-dir", jobs,
            "--job-name", job.name,
            "--concurrency", "2",
            "--freeze-workspace",
            "--max-sandbox-seconds", str(h.cap_seconds()),
            "--quiet",
            log=jobs / f"{job.name}.log",
        )  # fmt: skip
        ledger.record(
            f"eval run {agent} (2 tasks, --freeze-workspace, shared jobs folder)",
            surface="CLI",
            seconds=run.seconds,
            job_dir=job,
            result="pass" if run.returncode in (0, 1) else f"exit {run.returncode}",
        )
        h.assert_exit(run, 0, 1)
    return jobs


def test_oracle_scores_both_tasks_in_its_own_sandbox(restore_jobs: Path):
    job = restore_jobs / "restore-oracle"
    for task in ("e2e-private-key", "e2e-outside-state"):
        result = h.read_json(h.trial_of(job, task) / "result.json")
        assert result["rewards"] == {"reward": 1.0}, (task, result.get("error"))
    manifest = h.read_json(
        h.trial_of(job, "e2e-private-key") / "evidence/manifest.json"
    )
    modes = {e["path"]: e.get("mode") for e in manifest["entries"]}
    assert modes["ssl/server.key"] == 0o600 and modes["ssl"] == 0o700


def test_regrade_restores_modes_finds_tasks_and_marks_unrestored_state(
    restore_jobs: Path, ledger: h.Ledger
):
    job = restore_jobs / "restore-oracle"
    # No --tasks-dir: each trial finds its task through the job's evaluation.json.
    run, summary = h.bench_json(
        "eval", "regrade", job, "--reason", "unchanged verifier", "--json",
        timeout=1200,
    )  # fmt: skip
    ledger.record(
        "eval regrade, unchanged tasks, no --tasks-dir",
        surface="CLI",
        seconds=run.seconds,
    )
    h.assert_exit(run, 0)
    rows = {r["task"]: r for r in summary["trials"]}
    assert set(rows) == {"e2e-private-key", "e2e-outside-state"}, rows
    key = rows["e2e-private-key"]
    assert key["status"] == "regraded" and key["change"] == "same", key
    outside = rows["e2e-outside-state"]
    assert outside["change"] == "pass->fail", outside
    assert outside["task_changed"] is False
    assert "unchanged" in outside["reason"]


def test_list_and_metrics_see_every_job_and_agent(restore_jobs: Path):
    listing = h.bench("eval", "list", restore_jobs, env={"COLUMNS": "200"})
    h.assert_exit(listing, 0)
    assert "restore-oracle" in listing.output and "restore-nop" in listing.output
    run, metrics = h.bench_json("eval", "metrics", restore_jobs, "--json")
    h.assert_exit(run, 0)
    assert (metrics["total"], metrics["passed"], metrics["failed"]) == (4, 2, 2), (
        metrics
    )


def test_rubric_task_without_a_reviewer_model_names_the_reviewer_flag(
    sandbox: str, tasks_root: Path, e2e_out: Path
):
    root = tasks_root / "rubric-refusal"
    task = h.write_task(root, "e2e-rubric")
    (task / "verifier" / "rubric.json").write_text(json.dumps(RUBRIC))
    jobs = e2e_out / "jobs-rubric-refusal"
    run = h.bench(
        "eval", "run",
        "--tasks-dir", task,
        "--agent", "oracle",
        "--sandbox", sandbox,
        "--jobs-dir", jobs,
        "--max-sandbox-seconds", str(h.cap_seconds()),
        timeout=300,
    )  # fmt: skip
    assert run.returncode != 0, run.tail()
    assert "--reviewer-model" in run.output, run.tail()
    assert "pass --model" not in run.output
    # Refused before any trial: no trial folder exists.
    assert not list(jobs.rglob("result.json"))


def test_doctor_json_reports_the_backend_without_secrets(sandbox: str):
    import os

    run, report = h.bench_json("doctor", "--sandbox", sandbox, "--json", timeout=300)
    h.assert_exit(run, 0)
    checks = {c["id"]: c for c in report["checks"]}
    assert checks[sandbox]["status"] == "pass", checks[sandbox]
    assert report["sandbox"] == sandbox
    for name, value in os.environ.items():
        if name.endswith(("_API_KEY", "_TOKEN")) and len(value) >= 12:
            assert value not in run.output, f"{name} value printed by doctor"
