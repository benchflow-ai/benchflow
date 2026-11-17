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
  answers. Hardening leaves it out and the task declares no required_tools:
  the trial ends ``verifier tools refused``, unscored, never retried, a setup
  problem, and the equivalence check reports ``unscored`` with the reason.
  (The 0.9 line writes no report.md; summary.json, results.jsonl and the CLI
  are checked.)

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


def test_a_root_agent_that_reopens_it_gets_the_readable_refusal(root_job):
    run, job_dir = root_job
    # Never retried: one attempt.
    trial = _trial(job_dir, "reopened-toolchain")
    result = json.loads((trial / "result.json").read_text())
    assert result["rewards"] is None
    assert result["verifier_error_category"] == "verifier_tools_refused"
    error = result["verifier_error"]
    assert error.startswith("verifier tools refused: "), error
    assert "verifier crashed" not in error
    assert "/opt/toolchain/bin: root's, but writable by group or others" in error
    assert "required_tools" in error

    summary = json.loads((job_dir / "summary.json").read_text())
    assert summary["passed"] == 1
    assert summary["verifier_errored"] == 1
    assert summary["unusable_rewards"] == 0
    assert summary["verifier_error_categories"] == {"verifier_tools_refused": 1}
    # Out of the error-excluding score: 1 of 1, not 1 of 2.
    assert summary["score_excl_errors_ratio"] == pytest.approx(1.0)

    rows = {}
    for line in (job_dir / "results.jsonl").read_text().splitlines():
        row = json.loads(line)
        rows[row["info"]["task_name"]] = row
    assert rows["reopened-toolchain"]["reward"] is None
    assert (
        rows["reopened-toolchain"]["info"]["diagnostics"]["verifier_error_category"]
        == "verifier_tools_refused"
    )

    out = run.stdout + run.stderr
    lines = [
        line
        for line in out.splitlines()
        if "reopened-toolchain" in line and "problem)" in line
    ]
    assert lines and all("(setup problem)" in line for line in lines), out
    assert not any("task problem" in line for line in lines), out


def test_equivalence_locks_first_so_the_oracle_scores(tasks, tmp_path):
    report = _equivalence(tasks, "open-toolchain", tmp_path / "open.json")
    assert report["oracle_reward"] == 1.0, report["reason"]
    assert report["status"] not in ("oracle-fails", "unscored", "error")
    lock = report["locked_system_path"]
    assert lock["state"] == "enforced"
    assert "/opt/toolchain/bin" in lock["locked"]


def test_equivalence_reports_a_reopened_toolchain_unscored_with_its_reason(
    tasks, tmp_path
):
    report = _equivalence(tasks, "reopened-toolchain", tmp_path / "reopened.json")
    assert report["status"] == "unscored"
    assert report["oracle_reward"] is None
    reason = report["reason"]
    assert reason.startswith(
        "the oracle output was not scored: verifier tools refused: "
    ), reason
    assert "/opt/toolchain/bin" in reason
    assert "verifier crashed" not in reason


# --- an absent PATH entry only root could make (lead ruling, 2026-11-16) -------
#
# SWE-bench Multilingual's task images put /root/.cargo/bin on PATH and never
# make it (uv installs to /root/.local/bin): hardening left it out, "not
# present when the verifier PATH was checked", and the trial needed a
# required_tools contract (preact-2757). An absent entry is kept when every
# part of it that exists is root's alone, with no link another user controls:
# only root could ever make it. Otherwise it is refused with the reason.

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
REFUSED_WHY = {
    "absent-under-open-root": (
        "/root/.cargo/bin: not present when the verifier PATH was checked, "
        "and /root is writable by group or others"
    ),
    "absent-under-group-folder": (
        "/opt/grp/tools/bin: not present when the verifier PATH was checked, "
        "and /opt/grp is writable by group or others"
    ),
    "absent-through-dotdot-link": (
        "/opt/lnd/bin: not present when the verifier PATH was checked, "
        "and /opt/lnd links to a target with a .. part"
    ),
    "absent-through-user-link": (
        "/opt/lnk/tc/bin: not present when the verifier PATH was checked, "
        "and /opt/lnk/tc is a link root does not own (uid 1000)"
    ),
}


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
    trusted = json.loads((trial / "verifier" / "trusted-path.json").read_text())
    assert trusted["absent_root_only"] == ["/root/.cargo/bin"]
    assert trusted["dropped"] == []
    assert trusted["path"].split(":")[0] == "/root/.cargo/bin"
    lock = json.loads((trial / "locked-system-path.json").read_text())
    assert lock["absent_root_only"] == ["/root/.cargo/bin"]
    assert "only root could ever make it" in lock["absent_root_only_note"]


@pytest.mark.parametrize("name", sorted(REFUSED_WHY))
def test_an_absent_entry_others_could_make_is_refused_with_its_reason(absent_job, name):
    _, job_dir = absent_job
    trial = _trial(job_dir, name)  # one attempt: never retried
    result = json.loads((trial / "result.json").read_text())
    assert result["rewards"] is None
    assert result["verifier_error_category"] == "verifier_tools_refused"
    error = result["verifier_error"]
    assert error.startswith("verifier tools refused: "), error
    assert REFUSED_WHY[name] in error, error
    trusted = json.loads((trial / "verifier" / "trusted-path.json").read_text())
    assert "absent_root_only" not in trusted


def test_the_absent_entry_job_counts_four_unscored_one_passed(absent_job):
    _, job_dir = absent_job
    summary = json.loads((job_dir / "summary.json").read_text())
    assert summary["passed"] == 1
    assert summary["verifier_errored"] == 4
    assert summary["verifier_error_categories"] == {"verifier_tools_refused": 4}
    assert summary["score_excl_errors_ratio"] == pytest.approx(1.0)
