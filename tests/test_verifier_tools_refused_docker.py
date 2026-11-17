"""A CRUST-like image on real Docker: an open toolchain folder is locked, then graded.

The official rust image (CRUST-Bench's ``rust:1.83-slim``) leaves
``/usr/local/cargo``, ``/usr/local/cargo/bin`` and every program in it 0777
root:root, so any user can replace ``cargo``. A trial with a sandbox user
locks such PATH folders to root before the agent starts, and the verifier
keeps them. ``bench check --level equivalence`` and ``--sandbox-user none``
trials skipped that lock, so the verifier's PATH left the folder out and
hardening crashed ("verifier crashed: RuntimeError: Verifier hardening
failed: ... required_tools contract"): every CRUST-Bench oracle failed
equivalence on 0.10.0rc3 while ``bench run`` scored it 1.0.

Two tasks share one image whose ``/opt/toolchain/bin`` (on PATH, test.sh's
``crustc`` in it) is 0777 like the rust image's:

- ``open-toolchain``: the oracle writes the answer. Locked first, the folder
  is root's alone when the verifier runs: reward 1.0 in a root-agent trial
  and in the equivalence check, and the lock is recorded.
- ``reopened-toolchain``: the oracle (root) opens the folder again before it
  answers and puts nothing in it. The required_tools refusal is off in the
  default mode, and in strict (FAITHFUL G2, the faithful default 09:58Z), so
  the trial is graded as test.sh says: 1.0, from the image's own ``crustc``
  (test.sh prints its sha256), in the trial and in the equivalence check.

Deterministic tier (``tests/integration/README.md``): real Docker, the
oracle, no model.
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
DOCKERFILE = (
    "FROM python:3.12-slim@sha256:"
    "a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1\n"
    # As the official rust image leaves CARGO_HOME: 0777, root's.
    "RUN mkdir -p /opt/toolchain/bin /app && chmod 1777 /app"
    " && printf '#!/bin/sh\\necho ok\\n' > /opt/toolchain/bin/crustc"
    " && chmod -R a+rwx /opt/toolchain\n"
    "ENV PATH=/opt/toolchain/bin:$PATH\n"
    "WORKDIR /app\n"
)
TEST_SH = (
    "#!/bin/bash\n"
    "sha256sum /opt/toolchain/bin/crustc\n"
    'if [ "$(crustc)" = ok ] && [ "$(cat /app/answer.txt 2>/dev/null)" = ok ]; '
    "then r=1; else r=0; fi\n"
    'echo "$r" > /logs/verifier/reward.txt\n'
)
SOLVE = {
    "open-toolchain": "#!/bin/bash\necho ok > /app/answer.txt\n",
    # A root agent can undo the lock; the verifier must not then run what
    # anyone could have put there.
    "reopened-toolchain": (
        "#!/bin/bash\nchmod -R a+rwx /opt/toolchain\necho ok > /app/answer.txt\n"
    ),
}
TASK_TOML = """version = "1.0"
[metadata]
author_name = "benchflow"
difficulty = "easy"
category = "fixture"
[verifier]
timeout_sec = 120.0
[agent]
timeout_sec = 120.0
[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
storage_mb = 4096
"""
# sha256 of the image's own crustc (DOCKERFILE's printf).
IMAGE_CRUSTC_SHA256 = "b4d644d4279594903f1a9911956432d9473041f2984fc6014c14d7402c7d126c"
ENV = {**os.environ, "HARBOR_TELEMETRY": "off", "NO_COLOR": "1", "COLUMNS": "200"}


def _tasks(root: Path) -> Path:
    for name, solve in SOLVE.items():
        task = root / name
        for sub in ("environment", "solution", "tests"):
            (task / sub).mkdir(parents=True)
        (task / "task.toml").write_text(TASK_TOML)
        (task / "instruction.md").write_text("Write ok to /app/answer.txt.\n")
        (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
        for path, text in (
            (task / "solution" / "solve.sh", solve),
            (task / "tests" / "test.sh", TEST_SH),
        ):
            path.write_text(text)
            path.chmod(0o755)
    return root


def _bench(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "benchflow.cli.main", *args],
        env=ENV,
        capture_output=True,
        text=True,
        timeout=1800,
    )


@pytest.fixture(scope="module")
def tasks(tmp_path_factory) -> Path:
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else f"Docker-only ({SANDBOX})")
    return _tasks(tmp_path_factory.mktemp("crust-like") / "tasks")


@pytest.fixture(scope="module")
def root_job(tasks, tmp_path_factory):
    jobs = tmp_path_factory.mktemp("crust-like-jobs")
    run = _bench(
        "eval",
        "run",
        "--tasks-dir",
        str(tasks),
        "--agent",
        "oracle",
        "--sandbox",
        "docker",
        "--sandbox-user",
        "none",
        "--jobs-dir",
        str(jobs),
        "--retry-attempts",
        "2",
        "--concurrency",
        "2",
    )
    (job_dir,) = [p for p in jobs.iterdir() if p.is_dir()]
    return run, job_dir


def _trial(job_dir: Path, name: str) -> Path:
    (trial,) = [p for p in job_dir.glob(f"{name}__*") if p.is_dir()]
    return trial


def _equivalence(tasks: Path, name: str, out: Path) -> dict:
    run = _bench(
        "check",
        str(tasks / name),
        "--level",
        "equivalence",
        "--sandbox",
        "docker",
        "--report-output",
        str(out),
    )
    assert out.exists(), run.stdout + run.stderr
    return json.loads(out.read_text())


def test_a_root_agent_s_trial_locks_the_open_toolchain_and_scores(root_job):
    _, job_dir = root_job
    trial = _trial(job_dir, "open-toolchain")
    result = json.loads((trial / "result.json").read_text())
    assert result["verifier_error"] is None, result["verifier_error"]
    assert result["rewards"] == {"reward": 1.0}
    lock = json.loads((trial / "locked-system-path.json").read_text())
    assert lock["state"] == "enforced"
    assert "/opt/toolchain/bin" in lock["locked"]
    assert lock["note"].startswith("repaired:")


def test_a_root_agent_that_reopens_it_is_graded_as_test_sh_says(root_job):
    """FAITHFUL G2: no required_tools refusal in the default mode. The root
    agent reopened the folder and put nothing in it: test.sh ran the image's
    own crustc and its 1.0 is scored, once (never retried)."""
    _, job_dir = root_job
    trial = _trial(job_dir, "reopened-toolchain")
    result = json.loads((trial / "result.json").read_text())
    assert result["verifier_error"] is None, result["verifier_error"]
    assert result["rewards"] == {"reward": 1.0}
    said = (trial / "verifier" / "test-stdout.txt").read_text()
    assert IMAGE_CRUSTC_SHA256 in said, said
    config = json.loads((trial / "config.json").read_text())
    assert config["integrity"] == "audit", config

    summary = json.loads((job_dir / "summary.json").read_text())
    assert summary["passed"] == 2
    assert summary["verifier_errored"] == 0
    assert not summary.get("verifier_error_categories"), summary


def test_equivalence_locks_first_so_the_oracle_scores(tasks, tmp_path):
    report = _equivalence(tasks, "open-toolchain", tmp_path / "open.json")
    assert report["oracle_reward"] == 1.0, report["reason"]
    assert report["status"] not in ("oracle-fails", "unscored", "error")
    lock = report["locked_system_path"]
    assert lock["state"] == "enforced"
    assert "/opt/toolchain/bin" in lock["locked"]


def test_equivalence_grades_a_reopened_toolchain_as_test_sh_says(tasks, tmp_path):
    """FAITHFUL G2: the equivalence check grades the reopened toolchain too."""
    report = _equivalence(tasks, "reopened-toolchain", tmp_path / "reopened.json")
    assert report["status"] == "clean", report
    assert report["oracle_reward"] == 1.0, report["reason"]


# --- an absent PATH entry only root could make (lead ruling, 2026-11-16) -------
#
# SWE-bench Multilingual's task images put /root/.cargo/bin on PATH and never
# make it (uv installs to /root/.local/bin): hardening left it out, "not
# present when the verifier PATH was checked", and the trial needed a
# required_tools contract (preact-2757). An absent entry is kept when every
# part of it that exists is root's alone, with no link another user controls:
# only root could ever make it. The others were refused with the reason; with
# the required_tools refusal off (FAITHFUL G2) they are graded as test.sh says.

ABSENT_BASE = (
    "FROM python:3.12-slim@sha256:"
    "a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1\n"
    "RUN mkdir -p /app && chmod 1777 /app\n"
)
ABSENT = {
    # /root 0700 root:root, as in the Multilingual images: kept.
    "absent-under-root": ABSENT_BASE
    + "RUN chmod 0700 /root\nENV PATH=/root/.cargo/bin:$PATH\nWORKDIR /app\n",
    # /root open to everyone: anyone could make /root/.cargo/bin.
    "absent-under-open-root": ABSENT_BASE
    + "RUN chmod 0777 /root\nENV PATH=/root/.cargo/bin:$PATH\nWORKDIR /app\n",
    # A group-writable ancestor.
    "absent-under-group-folder": ABSENT_BASE
    + "RUN mkdir -p /opt/grp && chmod 0775 /opt/grp\n"
    "ENV PATH=/opt/grp/tools/bin:$PATH\nWORKDIR /app\n",
    # A root-owned link whose target has a `..` after a missing part (RUSAGE).
    "absent-through-dotdot-link": ABSENT_BASE
    + "RUN mkdir -p /opt/tw && chmod 0777 /opt/tw"
    " && ln -s /root/missing/../../opt/tw /opt/lnd\n"
    "ENV PATH=/opt/lnd/bin:$PATH\nWORKDIR /app\n",
    # A link another user owns, in a root-only folder.
    "absent-through-user-link": ABSENT_BASE
    + "RUN mkdir -p /opt/lnk && ln -s /opt/lnk-target /opt/lnk/tc"
    " && chown -h 1000:1000 /opt/lnk/tc\n"
    "ENV PATH=/opt/lnk/tc/bin:$PATH\nWORKDIR /app\n",
}
ABSENT_TEST_SH = (
    "#!/bin/bash\n"
    'if [ "$(cat /app/answer.txt 2>/dev/null)" = ok ]; then r=1; else r=0; fi\n'
    'echo "$r" > /logs/verifier/reward.txt\n'
)
# Absent entries a user other than root could make. The required_tools
# refusal named them; it is off in the default mode (FAITHFUL G2).
OTHERS_COULD_MAKE = (
    "absent-under-open-root",
    "absent-under-group-folder",
    "absent-through-dotdot-link",
    "absent-through-user-link",
)


@pytest.fixture(scope="module")
def absent_job(tmp_path_factory):
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else f"Docker-only ({SANDBOX})")
    root = tmp_path_factory.mktemp("absent-entries")
    for name, dockerfile in ABSENT.items():
        task = root / "tasks" / name
        for sub in ("environment", "solution", "tests"):
            (task / sub).mkdir(parents=True)
        (task / "task.toml").write_text(TASK_TOML)
        (task / "instruction.md").write_text("Write ok to /app/answer.txt.\n")
        (task / "environment" / "Dockerfile").write_text(dockerfile)
        for path, text in (
            (task / "solution" / "solve.sh", SOLVE["open-toolchain"]),
            (task / "tests" / "test.sh", ABSENT_TEST_SH),
        ):
            path.write_text(text)
            path.chmod(0o755)
    jobs = root / "jobs"
    # The default sandbox user: an agent that is not root.
    run = _bench(
        "eval",
        "run",
        "--tasks-dir",
        str(root / "tasks"),
        "--agent",
        "oracle",
        "--sandbox",
        "docker",
        "--jobs-dir",
        str(jobs),
        "--retry-attempts",
        "2",
        "--concurrency",
        "4",
    )
    (job_dir,) = [p for p in jobs.iterdir() if p.is_dir()]
    return run, job_dir


def test_an_absent_entry_only_root_could_make_is_kept_and_recorded(absent_job):
    _, job_dir = absent_job
    trial = _trial(job_dir, "absent-under-root")
    result = json.loads((trial / "result.json").read_text())
    assert result["verifier_error"] is None, result["verifier_error"]
    assert result["rewards"] == {"reward": 1.0}
    # trusted-path.json is the hardening's record; the default mode grades
    # test.sh with the image's PATH (FAITHFUL G2).
    assert not (trial / "verifier" / "trusted-path.json").exists()
    # The PATH judge, and its absent_root_only note, are strict's (FAITHFUL G1).
    lock = json.loads((trial / "locked-system-path.json").read_text())
    assert "absent_root_only" not in lock, lock


@pytest.mark.parametrize("name", OTHERS_COULD_MAKE)
def test_an_absent_entry_others_could_make_is_graded_as_test_sh_says(absent_job, name):
    """FAITHFUL G2: no required_tools refusal in the default mode. The entry
    is absent and test.sh reads only the answer the oracle wrote: its 1.0 is
    scored, once (never retried), and no hardening record is written."""
    _, job_dir = absent_job
    trial = _trial(job_dir, name)
    result = json.loads((trial / "result.json").read_text())
    assert result["verifier_error"] is None, result["verifier_error"]
    assert result["rewards"] == {"reward": 1.0}
    assert not (trial / "verifier" / "trusted-path.json").exists()


def test_the_absent_entry_job_counts_every_trial_passed(absent_job):
    """FAITHFUL G2: no trial of the job is refused for an absent PATH entry."""
    _, job_dir = absent_job
    summary = json.loads((job_dir / "summary.json").read_text())
    assert summary["passed"] == len(OTHERS_COULD_MAKE) + 1, summary
    assert summary["verifier_errored"] == 0, summary
    assert not summary.get("verifier_error_categories"), summary
