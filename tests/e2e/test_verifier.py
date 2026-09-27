"""Verifier features: regrade (CLI and Python), separate verifier sandboxes,
verifier recovery after a lost sandbox."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

import benchflow as bf
from tests.e2e import harness as h

# ---------------------------------------------------------------------------
# Regrade
# ---------------------------------------------------------------------------

LENIENT_TEST = """#!/bin/bash
# Fixed verifier: also accepts the farewell the first verifier rejected.
case "$(tr -d '\\n' < /app/hello.txt 2>/dev/null)" in
  "Hello, world!"|"Goodbye") echo 1 > /logs/verifier/reward.txt ;;
  *) echo 0 > /logs/verifier/reward.txt ;;
esac
"""


def _fixed_tasks(batch_tasks: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(batch_tasks, dest)
    for name in ("e2e-fail", "e2e-verifier-error"):
        test = dest / name / "verifier" / "test.sh"
        test.write_text(LENIENT_TEST)
        test.chmod(0o755)
    return dest


def test_cli_regrade_changed_verifier(
    batch_job: Path,
    batch_tasks: Path,
    tasks_root: Path,
    jobs_root: Path,
    ledger: h.Ledger,
):
    fixed = _fixed_tasks(batch_tasks, tasks_root / "batch-fixed")
    run, summary = h.bench_json(
        "eval", "regrade", batch_job, "--tasks-dir", fixed, "--reason", "accept farewell",
        "--json", timeout=1200,
    )  # fmt: skip
    ledger.record("eval regrade (3 frozen trials, changed verifier)", surface="CLI",
                  seconds=run.seconds)  # fmt: skip
    h.assert_exit(run, 0)
    rows = {r["task"]: r for r in summary["trials"]}
    assert rows["e2e-pass"]["change"] == "same"
    assert rows["e2e-fail"]["change"] == "fail->pass"
    assert rows["e2e-verifier-error"]["change"] == "scored"
    counts = summary["counts"]
    assert counts["regraded"] == 3 and counts["fail_to_pass"] == 1, counts
    # The original result.json is untouched; regrade.json sits beside it.
    trial = h.trial_of(batch_job, "e2e-fail")
    assert h.read_json(trial / "result.json")["rewards"] == {"reward": 0.0}
    record = h.read_json(trial / "regrade.json")
    (block,) = [b for b in record["regrades"] if b["id"] == record["latest"]]
    assert block["new_reward"] == 1.0 and block["reason"] == "accept farewell"
    assert block["task_changed"] is True
    assert (batch_job / "regrade-summary.json").is_file()


def test_python_regrade_and_not_regradable(
    sandbox: str, batch_job: Path, batch_tasks: Path, tasks_root: Path, jobs_root: Path,
    ledger: h.Ledger,
):  # fmt: skip
    started = time.monotonic()
    trial = h.trial_of(batch_job, "e2e-pass")
    summary = bf.regrade(trial, tasks_dir=batch_tasks, reason="unchanged verifier")
    ledger.record("bf.regrade one trial, unchanged verifier", surface="Python",
                  seconds=time.monotonic() - started)  # fmt: skip
    counts = summary.counts()
    assert counts["regraded"] == 1 and counts["changed"] == 0, counts
    # A run without --freeze-workspace is reported, never scored.
    job = jobs_root / "no-freeze"
    if h.needs_run(job):
        h.bench(
            "eval", "run", "--tasks-dir", batch_tasks / "e2e-pass", "--agent", "oracle",
            "--sandbox", sandbox, "--jobs-dir", jobs_root, "--job-name", job.name, "--quiet",
            "--max-sandbox-seconds", str(h.cap_seconds()), log=jobs_root / "no-freeze.log",
        )  # fmt: skip
    run, data = h.bench_json(
        "eval", "regrade", job, "--tasks-dir", batch_tasks, "--json"
    )
    h.assert_exit(run, 0)
    assert data["counts"]["not_regradable"] == 1, data["counts"]
    for path in job.glob("*/regrade.json"):
        assert all(b.get("new_reward") is None for b in h.read_json(path)["regrades"])


# ---------------------------------------------------------------------------
# Separate verifier sandboxes
# ---------------------------------------------------------------------------

PLANTING_SOLVE = """#!/bin/bash
printf 'Hello, world!\\n' > /app/hello.txt
# Outside the workspace: must never reach a separate verifier.
echo planted > /tmp/agent-planted
mkdir -p /logs/verifier && echo 1 > /logs/verifier/reward.txt
"""
SEEING_TEST = """#!/bin/bash
if [ -e /tmp/agent-planted ]; then echo "SAW_PLANTED_FILE"; else echo "NO_PLANTED_FILE"; fi
if [ "$(tr -d '\\n' < /app/hello.txt 2>/dev/null)" = "Hello, world!" ] && [ ! -e /tmp/agent-planted ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""


@pytest.mark.parametrize(
    "variant",
    [
        {"sandbox_mode": "separate"},
        {"sandbox_mode": "separate", "sandbox": {"docker_image": h.PY_IMAGE}},
    ],
    ids=["task-dockerfile", "verifier-image"],
)
def test_separate_verifier_sees_only_the_workspace(
    sandbox: str, tasks_root: Path, jobs_root: Path, ledger: h.Ledger, variant
):
    name = "e2e-separate-" + ("image" if "sandbox" in variant else "dockerfile")
    task = h.write_task(
        tasks_root / "separate", name, solve=PLANTING_SOLVE, test=SEEING_TEST,
        frontmatter={"verifier": {"timeout_sec": 90, **variant}},
    )  # fmt: skip
    run = h.bench("tasks", "check", task, "--sandbox", sandbox)
    h.assert_exit(run, 0)
    job = jobs_root / name
    h.clear_job(job)
    run = h.bench(
        "eval", "run", "--tasks-dir", task, "--agent", "oracle", "--sandbox", sandbox,
        "--jobs-dir", jobs_root, "--job-name", job.name, "--quiet",
        "--max-sandbox-seconds", str(h.cap_seconds()), log=jobs_root / f"{name}.log",
    )  # fmt: skip
    ledger.record(f"separate verifier sandbox ({name})", surface="CLI",
                  seconds=run.seconds, job_dir=job)  # fmt: skip
    h.assert_exit(run, 0)
    trial = h.trial_of(job, name)
    stdout = (trial / "verifier" / "test-stdout.txt").read_text()
    assert "NO_PLANTED_FILE" in stdout, stdout
    assert h.read_json(trial / "result.json")["rewards"] == {"reward": 1.0}
    record = h.read_json(trial / "verifier-sandbox" / "verifier-sandbox.json")
    assert record["status"] == "complete", record
    assert record["sandbox_id"]
    expected_source = "verifier" if "sandbox" in variant else "environment"
    assert expected_source in str(record["image_source"]), record["image_source"]
    timing = h.read_json(trial / "timing.json")
    for key in (
        "verifier_sandbox_setup",
        "verifier_transfer",
        "verifier_sandbox_total",
    ):
        assert key in timing, timing


# ---------------------------------------------------------------------------
# Verifier recovery after a lost sandbox
# ---------------------------------------------------------------------------

RECOVERY_SOLVE = """#!/bin/bash
printf 'Hello, world!\\n' > /app/hello.txt
# A marker outside the workspace: present only in the original sandbox.
echo original > /tmp/original-sandbox
"""
RECOVERY_TEST = """#!/bin/bash
if [ -e /tmp/original-sandbox ]; then
  echo "original sandbox: waiting to be lost"
  sleep 80
fi
if [ "$(tr -d '\\n' < /app/hello.txt 2>/dev/null)" = "Hello, world!" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""
PINNED_PY = (
    "python@sha256:6771159cd4fa5d9bba1258caf0b82e6b73458c694d178ad97c5e925c2d0e1a91"
)


def _delete_owner_sandboxes_created_after(started_epoch: float) -> list[str]:
    from datetime import datetime

    from benchflow.sandbox.daytona import build_sync_client

    client = build_sync_client()
    owner = os.environ["BENCHFLOW_DAYTONA_OWNER"]
    deleted = []
    for sb in client.list().items if hasattr(client.list(), "items") else client.list():
        labels = getattr(sb, "labels", {}) or {}
        if owner not in labels.values():
            continue
        created = getattr(sb, "created_at", None)
        if isinstance(created, str):
            created = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
        if created and created >= started_epoch - 5:
            client.delete(sb)
            deleted.append(sb.id)
    return deleted


def test_verifier_recovery_after_lost_sandbox(
    sandbox: str, tasks_root: Path, jobs_root: Path, ledger: h.Ledger
):
    task = h.write_task(
        tasks_root / "recovery", "e2e-recovery", solve=RECOVERY_SOLVE, test=RECOVERY_TEST,
        frontmatter={
            "verifier": {"timeout_sec": 150, "workspace_recovery": True},
            "sandbox": {"docker_image": PINNED_PY, "workdir": "/app"},
        },
    )  # fmt: skip
    # The package still needs environment/Dockerfile; docker_image pins the
    # image that recovery restarts from.
    (task / "environment" / "Dockerfile").write_text(
        f"FROM {PINNED_PY}\nWORKDIR /app\nRUN mkdir -p /logs/verifier /logs/agent /logs/artifacts\n"
    )
    job = jobs_root / "recovery"
    h.clear_job(job)
    log = jobs_root / "recovery.log"
    argv = [
        h.bench_executable(), "eval", "run", "--tasks-dir", str(task), "--agent", "oracle",
        "--sandbox", sandbox, "--jobs-dir", str(jobs_root), "--job-name", job.name,
        "--max-sandbox-seconds", str(h.cap_seconds()), "--retry-attempts", "0",
    ]  # fmt: skip
    started = time.monotonic()
    started_epoch = time.time()
    with log.open("w") as fh:
        proc = subprocess.Popen(
            argv,
            cwd=h.REPO_ROOT,
            env=h.clean_env(),
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
        deleted: list[str] = []
        deadline = time.monotonic() + 600
        while proc.poll() is None and time.monotonic() < deadline:
            if not deleted and "Running verifier" in log.read_text():
                time.sleep(15)  # let the verifier reach its sleep
                deleted = _delete_owner_sandboxes_created_after(started_epoch)
            time.sleep(3)
        proc.wait(timeout=600)
    ledger.record("verifier recovery after the sandbox is lost mid-verifier", surface="CLI",
                  seconds=time.monotonic() - started, job_dir=job)  # fmt: skip
    assert deleted, f"never saw the verifier start:\n{log.read_text()[-3000:]}"
    trial = h.trial_of(job, "e2e-recovery")
    assert (trial / "solver-complete.json").is_file() or (
        trial / "solver.json"
    ).is_file()
    verification = h.read_json(trial / "verification.json")
    attempts = sorted((trial / "verifier-recovery").glob("*/recovery.json"))
    assert attempts, "no recovery attempt recorded"
    receipt = h.read_json(attempts[-1])
    assert receipt.get("solver_replayed") in (False, None), receipt
    result = h.read_json(trial / "result.json")
    assert result["rewards"] == {"reward": 1.0}, (
        result.get("verifier_error"),
        verification,
    )
    # The start receipt lives under /run/benchflow, never in the verifier outputs.
    assert not list((trial / "verifier").glob("*receipt*"))
