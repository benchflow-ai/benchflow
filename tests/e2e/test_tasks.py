"""Task authoring: bench tasks init/check, canaries, the equivalence level,
lenient legacy task.toml on the run path."""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

import pytest

from tests.e2e import harness as h

GUID = re.compile(r"canary GUID ([0-9a-f-]{36})")


def test_init_writes_canaries_and_check_warns_without_one(
    tmp_path: Path, ledger: h.Ledger
):
    started = time.monotonic()
    run = h.bench("tasks", "init", "e2e-scaffold", "--dir", tmp_path)
    h.assert_exit(run, 0)
    task = tmp_path / "e2e-scaffold"
    files = ["task.md", "verifier/test.sh", "oracle/solve.sh", "environment/Dockerfile"]
    guids = {f: GUID.findall((task / f).read_text()) for f in files}
    assert all(len(g) == 1 for g in guids.values()), guids
    assert len({g[0] for g in guids.values()}) == 1, "one GUID per task"
    # The scaffold is not runnable until its placeholders are replaced.
    run = h.bench("tasks", "check", task)
    h.assert_exit(run, 1)
    assert "placeholder" in run.output
    # A second task gets a different GUID.
    h.bench("tasks", "init", "e2e-scaffold-2", "--dir", tmp_path)
    other = GUID.findall((tmp_path / "e2e-scaffold-2" / "task.md").read_text())
    assert other and other[0] != guids["task.md"][0]
    # A complete task without a canary passes with a warning.
    bare = h.write_task(tmp_path, "e2e-no-canary", canary=False)
    run = h.bench("tasks", "check", bare)
    h.assert_exit(run, 0)
    assert "no canary string" in run.output
    ledger.record("tasks init canaries; check warns without one", surface="CLI",
                  seconds=time.monotonic() - started)  # fmt: skip


ANSWER_SOLVE = """#!/bin/bash
printf '{"total": 42.5, "unit": "kg"}\\n' > /app/answer.json
"""
STRICT_TEST = """#!/bin/bash
# Compares bytes: rejects a re-indented but equal answer.
if [ "$(cat /app/answer.json 2>/dev/null)" = '{"total": 42.5, "unit": "kg"}' ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""
ROBUST_TEST = """#!/bin/bash
# Compares values: reads /app/answer.json as JSON.
python3 - <<'PY'
import json, math
try:
    data = json.load(open("/app/answer.json"))
    ok = math.isclose(float(data["total"]), 42.5, rel_tol=1e-6) and data["unit"] == "kg"
except Exception:
    ok = False
open("/logs/verifier/reward.txt", "w").write("1" if ok else "0")
PY
"""


@pytest.mark.parametrize("kind", ["strict", "robust"])
def test_equivalence_level(
    sandbox: str, tmp_path: Path, jobs_root: Path, ledger: h.Ledger, kind
):
    task = h.write_task(
        tmp_path, f"e2e-equivalence-{kind}", prompt="Write the total mass to `/app/answer.json`.",
        solve=ANSWER_SOLVE, test=STRICT_TEST if kind == "strict" else ROBUST_TEST,
    )  # fmt: skip
    report = jobs_root / f"equivalence-{kind}.json"
    run = h.bench(
        "tasks", "check", task, "--level", "equivalence", "--sandbox", sandbox,
        "--report-output", report, log=jobs_root / f"equivalence-{kind}.log", timeout=900,
    )  # fmt: skip
    ledger.record(f"tasks check --level equivalence ({kind} verifier)", surface="CLI",
                  seconds=run.seconds)  # fmt: skip
    data = h.read_json(report)
    text = run.output
    if kind == "strict":
        h.assert_exit(run, 1)
        assert "false negative" in text
    else:
        h.assert_exit(run, 0)
        assert "false negative" not in text and "false positive" not in text, text[
            -3000:
        ]
    assert data


def test_python_check_equivalence(sandbox: str, tmp_path: Path, ledger: h.Ledger):
    from benchflow.task import check_equivalence

    task = h.write_task(
        tmp_path, "e2e-equivalence-py", prompt="Write the total mass to `/app/answer.json`.",
        solve=ANSWER_SOLVE, test=STRICT_TEST,
    )  # fmt: skip
    started = time.monotonic()
    report = check_equivalence(task, sandbox_type=sandbox)
    ledger.record("check_equivalence (strict verifier)", surface="Python",
                  seconds=time.monotonic() - started)  # fmt: skip
    assert report.false_negatives
    assert not report.false_positives
    assert report.to_json()


LEGACY_TOML = """version = "1.0"
unknown_future_key = true

[metadata]
author_name = "e2e"
difficulty = "easy"
category = "sanity"

[agent]
timeout_sec = 180.0
some_new_agent_option = 3

[verifier]
timeout_sec = 90.0

[environment]
cpus = 1
memory_mb = 1024
"""


def _legacy_task(root: Path, name: str, toml: str) -> Path:
    task = root / name
    if task.exists():
        shutil.rmtree(task)
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    (task / "task.toml").write_text(toml)
    (task / "instruction.md").write_text(h.HELLO_PROMPT + "\n")
    (task / "environment" / "Dockerfile").write_text(h.DOCKERFILE)
    for rel, text in (
        ("tests/test.sh", h.HELLO_TEST),
        ("solution/solve.sh", h.HELLO_SOLVE),
    ):
        (task / rel).write_text(text)
        (task / rel).chmod(0o755)
    return task


def test_lenient_legacy_task_toml_runs(
    sandbox: str, tasks_root: Path, jobs_root: Path, ledger: h.Ledger
):
    task = _legacy_task(tasks_root / "legacy", "e2e-legacy", LEGACY_TOML)
    run = h.bench("tasks", "check", task)
    h.assert_exit(run, 0)
    assert "unknown_future_key" in run.output and "some_new_agent_option" in run.output
    job = jobs_root / "legacy"
    h.clear_job(job)
    run = h.bench(
        "eval", "run", "--tasks-dir", task, "--agent", "oracle", "--sandbox", sandbox,
        "--jobs-dir", jobs_root, "--job-name", job.name,
        "--max-sandbox-seconds", str(h.cap_seconds()), log=jobs_root / "legacy.log",
    )  # fmt: skip
    ledger.record("legacy task.toml with unknown keys runs (oracle)", surface="CLI",
                  seconds=run.seconds, job_dir=job)  # fmt: skip
    h.assert_exit(run, 0)
    assert "unknown_future_key" in run.output
    assert h.read_json(h.trial_of(job, "e2e-legacy") / "result.json")["rewards"] == {
        "reward": 1.0
    }


def test_unhonourable_legacy_key_is_refused_before_launch(
    sandbox: str, tasks_root: Path, jobs_root: Path
):
    toml = LEGACY_TOML + '\n[[verifier.collect]]\npath = "/app/out"\n'
    task = _legacy_task(tasks_root / "legacy", "e2e-legacy-collect", toml)
    job = jobs_root / "legacy-collect"
    h.clear_job(job)
    run = h.bench(
        "eval", "run", "--tasks-dir", task, "--agent", "oracle", "--sandbox", sandbox,
        "--jobs-dir", jobs_root, "--job-name", job.name,
    )  # fmt: skip
    assert run.returncode != 0, run.tail()
    assert "collect" in run.output
    # Refused before any sandbox starts: an errored trial with no sandbox.
    (trial,) = h.trial_dirs(job)
    assert not (trial / "sandbox.json").exists()
    assert "verifier.collect" in h.read_json(trial / "result.json")["error"]
    # bench tasks check names it up front for the sandbox.
    run = h.bench("tasks", "check", task, "--sandbox", sandbox)
    h.assert_exit(run, 1)
    assert "verifier.collect" in run.output


def test_batch_with_a_symlinked_task_skips_it_and_keeps_the_rest(
    sandbox: str, tasks_root: Path, jobs_root: Path, ledger: h.Ledger
):
    """Regression: one task relying on symlinks cost an agent run per retry, then
    lost the whole job's summary. It must be skipped with its reason instead."""
    root = tasks_root / "symlinked-batch"
    if root.exists():
        shutil.rmtree(root)
    h.write_task(root, "e2e-plain")
    variant = root / "e2e-linked"
    variant.mkdir()
    shutil.copy(root / "e2e-plain" / "task.md", variant / "task.md")
    for name in ("environment", "oracle", "verifier"):
        (variant / name).symlink_to(Path("..") / "e2e-plain" / name)
    run = h.bench("tasks", "check", variant)
    h.assert_exit(run, 1)
    assert "symlinks" in run.output
    job = jobs_root / "symlinked-batch"
    h.clear_job(job)
    run = h.bench(
        "eval", "run", "--tasks-dir", root, "--agent", "oracle", "--sandbox", sandbox,
        "--jobs-dir", jobs_root, "--job-name", job.name,
        "--max-sandbox-seconds", str(h.cap_seconds()), log=jobs_root / "symlinked-batch.log",
    )  # fmt: skip
    ledger.record("batch with a symlinked task (skipped with reason)", surface="CLI",
                  seconds=run.seconds, job_dir=job)  # fmt: skip
    summary = h.read_json(job / "summary.json")
    assert summary["total"] == 1 and summary["passed"] == 1, summary
    assert not list(job.glob("e2e-linked__*"))
    assert "e2e-linked" in run.output and "symlink" in run.output
