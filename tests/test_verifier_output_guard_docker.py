"""EN-675 on real Docker: the task's problem and BenchFlow's own, told apart everywhere.

Three tasks through ``bench eval run`` (the oracle, retries on):

- ``pass``: a root image whose test.sh writes 1.
- ``task-problem``: a root image whose test.sh writes no reward. The
  verifier's error, a failure in every score (``verifier_no_usable_reward``).
- ``verifier-infra``: an image that ends ``USER nobody`` and names no
  ``[verifier] user``, so test.sh would run as nobody (the rc only: 0.9.3
  cannot set up the sandbox user in an image whose USER is not root).
- ``verifier-named``: a root image whose task names ``[verifier] user =
  "nobody"``.

In both, BenchFlow makes verifier/test-stdout.txt on the host first, as the
host's user, 0644, and nobody cannot open it. The check before the test
command ends the trial as a verifier setup failure: unscored, out of
``score_excl_errors``, named an infrastructure problem, never retried.
(Before EN-675 the shell's redirect failed with exit 2 and the trial read as
the task's: no usable reward.)

Each place a job reports from is checked: summary.json, the job's
results.jsonl and the CLI's lines (0.9.3 writes no report.md).

Deterministic tier (``tests/integration/README.md``): real Docker, no model.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()
BASE = (
    "FROM python:3.12-slim@sha256:"
    "a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1\n"
    "RUN mkdir -p /app && chmod 1777 /app\nWORKDIR /app\n"
)
WRITES_ONE = (
    "#!/bin/bash\n"
    'if [ "$(cat /app/answer.txt 2>/dev/null)" = ok ]; then r=1; else r=0; fi\n'
    'echo "$r" > /logs/verifier/reward.txt\n'
)
WRITES_NONE = "#!/bin/bash\necho 'graded, but no reward written'\n"
TASKS = {
    "pass": (BASE, WRITES_ONE, ""),
    "task-problem": (BASE, WRITES_NONE, ""),
    "verifier-named": (BASE, WRITES_ONE, 'user = "nobody"\n'),
}
INFRA = ("verifier-named",)
TASK_TOML = """version = "1.0"
[metadata]
author_name = "benchflow"
difficulty = "easy"
category = "fixture"
[verifier]
timeout_sec = 120.0
{verifier_user}[agent]
timeout_sec = 120.0
[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
storage_mb = 4096
"""


def _tasks(root: Path) -> Path:
    for name, (dockerfile, test_sh, verifier_user) in TASKS.items():
        task = root / name
        for sub in ("environment", "solution", "tests"):
            (task / sub).mkdir(parents=True)
        (task / "task.toml").write_text(TASK_TOML.format(verifier_user=verifier_user))
        (task / "instruction.md").write_text("Write ok to /app/answer.txt.\n")
        (task / "environment" / "Dockerfile").write_text(dockerfile)
        for path, text in (
            (
                task / "solution" / "solve.sh",
                "#!/bin/bash\necho ok > /app/answer.txt\n",
            ),
            (task / "tests" / "test.sh", test_sh),
        ):
            path.write_text(text)
            path.chmod(0o755)
    return root


@pytest.fixture(scope="module")
def job(tmp_path_factory):
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else f"Docker-only ({SANDBOX})")
    if os.geteuid() == 0:
        pytest.skip("BenchFlow on the host as root can write any file it makes")
    root = tmp_path_factory.mktemp("en675")
    tasks = _tasks(root / "tasks")
    jobs = root / "jobs"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchflow.cli.main",
            "eval",
            "run",
            "--tasks-dir",
            str(tasks),
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--jobs-dir",
            str(jobs),
            "--retry-attempts",
            "2",
            "--concurrency",
            "3",
        ],
        env={**os.environ, "HARBOR_TELEMETRY": "off", "NO_COLOR": "1"},
        capture_output=True,
        text=True,
        timeout=1800,
    )
    (job_dir,) = [p for p in jobs.iterdir() if p.is_dir()]
    return run, job_dir


def test_summary_json_counts_the_infra_trial_unscored(job):
    _, job_dir = job
    summary = json.loads((job_dir / "summary.json").read_text())
    assert summary["passed"] == 1
    assert summary["verifier_errored"] == 1 + len(INFRA)
    # Only the task's problem is a refusal (a failure in every score).
    assert summary["unusable_rewards"] == 1
    assert summary["verifier_error_categories"] == {
        "verifier_failure": 1,
        "verifier_infra": len(INFRA),
    }
    # The score is over every trial; the error-excluding score keeps passed,
    # failed and refused, so 1 of 2, not 1 of all.
    assert summary["score_excl_errors_ratio"] == pytest.approx(1 / 2)
    assert summary["score_ratio"] == pytest.approx(1 / len(TASKS))


@pytest.mark.parametrize("name", INFRA)
def test_the_infra_trial_is_never_retried(job, name):
    _, job_dir = job
    assert len(list(job_dir.glob(f"{name}__*"))) == 1
    (result,) = [
        json.loads(p.read_text()) for p in job_dir.glob(f"{name}__*/result.json")
    ]
    assert result["rewards"] is None
    assert result["verifier_error_category"] == "verifier_infra"
    error = result["verifier_error"]
    assert (
        "Verifier setup failed: the verifier's user cannot write its output " in error
    )
    assert "/logs/verifier/test-stdout.txt" in error
    assert "the verifier runs as uid 65534" in error
    assert f"the file is uid {os.getuid()}'s, mode 0644" in error


def test_results_jsonl_rows_carry_each_category(job):
    _, job_dir = job
    rows = {}
    for line in (job_dir / "results.jsonl").read_text().splitlines():
        row = json.loads(line)
        rows[row["info"]["task_name"]] = row
    assert rows["pass"]["reward"] == 1.0
    for name, category in (
        ("task-problem", "verifier_failure"),
        *((infra, "verifier_infra") for infra in INFRA),
    ):
        assert rows[name]["reward"] is None
        assert rows[name]["info"]["diagnostics"]["verifier_error_category"] == category


def test_the_cli_names_the_infra_trial_an_infrastructure_problem(job):
    run, _ = job
    out = run.stdout + run.stderr
    lines = out.splitlines()
    infra = [
        line
        for line in lines
        if any(name in line for name in INFRA) and "problem)" in line
    ]
    task = [line for line in lines if "task-problem" in line and "problem)" in line]
    assert infra and all("(infrastructure problem)" in line for line in infra), out
    assert not any("task problem" in line for line in infra), out
    assert task and all("(task problem)" in line for line in task), out
    # The scored-only figure leaves the infra trial out: 1 of 2.
    assert "scored only: 1/2" in out, out
    # The "check verifier scripts" warning counts only the task's problem.
    total = len(TASKS)
    assert f"1 tasks ({100 / total:.0f}%) had verifier errors" in out, out
    assert f"{1 + len(INFRA)} tasks (" not in out, out
    assert f"{len(INFRA)} task(s) were not scored" in out, out
