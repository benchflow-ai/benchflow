"""A one-shot Compose service done at start-up leaves a trial with a separate verifier scored, on real Docker.

The task's seeder runs while the start covers it, seeds, and exits 0 before the turn: its network namespace is gone,
so the re-proof at the end of the turn can't join it. It runs nothing, so its rules left unproven are no breach and
no outside cause. The in-place verifier already scored such a trial as test.sh says; the separate verifier ended it
"network policy not proven", unscored.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()

TASK_TOML = """version = "1.0"

[metadata]
author_name = "benchflow"
difficulty = "easy"
category = "sanity"

[verifier]
timeout_sec = 120.0
environment_mode = "separate"

[agent]
timeout_sec = 120.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 512
"""

COMPOSE = """services:
  seeder:
    image: ubuntu:24.04
    command: ["sh", "-c", "sleep 20; echo seeded"]
  main:
    depends_on:
      seeder:
        condition: service_completed_successfully
"""

IMAGE = "FROM ubuntu:24.04\nWORKDIR /app\n"


def test_a_one_shot_service_done_at_start_up_leaves_a_separate_verifier_scored(
    tmp_path,
):
    """The oracle's trial is scored as test.sh says (1), and result.json names the seeder under ``held_exited``."""
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else "Docker only")
    task = tmp_path / "tasks" / "oneshot-separate"
    for folder in ("environment", "solution", "tests"):
        (task / folder).mkdir(parents=True)
    (task / "task.toml").write_text(TASK_TOML)
    (task / "instruction.md").write_text("Write /app/done.txt.\n")
    (task / "environment" / "Dockerfile").write_text(IMAGE)
    (task / "environment" / "docker-compose.yaml").write_text(COMPOSE)
    (task / "tests" / "Dockerfile").write_text(IMAGE)
    for path, text in (
        (task / "solution" / "solve.sh", "#!/bin/bash\necho done > /app/done.txt\n"),
        (
            task / "tests" / "test.sh",
            "#!/bin/bash\n[ -f /app/done.txt ] && r=1 || r=0\necho $r > /logs/verifier/reward.txt\n",
        ),
    ):
        path.write_text(text)
        path.chmod(0o755)
    jobs = tmp_path / "jobs"
    done = subprocess.run(
        [
            h.bench_executable(),
            "eval",
            "run",
            "--tasks-dir",
            str(task.parent),
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--retry-attempts",
            "0",
            "--jobs-dir",
            str(jobs),
        ],
        cwd=h.REPO_ROOT,
        env={
            **os.environ,
            "BENCHFLOW_SKIP_UPDATE_CHECK": "1",
            "HARBOR_TELEMETRY": "off",
            "DO_NOT_TRACK": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1200,
    )
    results = list(jobs.glob("*/*__*/result.json"))
    assert len(results) == 1, done.stdout[-3000:]
    trial = json.loads(results[0].read_text())
    assert trial.get("error") is None and trial.get("verifier_error") is None, (
        trial.get("error") or trial.get("verifier_error")
    )
    assert (trial.get("rewards") or {}).get("reward") == 1.0, results[0].parent
    assert [r["service"] for r in trial.get("held_exited") or []] == ["seeder"], (
        trial.get("held_exited")
    )
