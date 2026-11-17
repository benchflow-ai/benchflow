"""EN-715 on real Docker: what test.sh needs, decided before the agent's turn.

One image family, with ``mytool`` at ``/opt/x/bin`` on PATH or not:

- ``needs-missing-tool``: ``set -e`` then ``mytool``, in an image without
  it. Refused before the agent's turn ("sandbox startup refused: test.sh
  needs `mytool` ..."): unscored, never retried, and the oracle never ran.
- ``tool-present``: the same test.sh with ``mytool`` on PATH scores 1.0, and
  records the check with no warning.
- ``agent-deletes-tool`` / ``agent-shadows-tool``: ``mytool`` resolved at
  preflight; the oracle (root) deletes it, or replaces it with one that
  fails (a shadow). The agent's doing: never refused, a failure as before
  ("verifier wrote no usable reward"), which closes deleting a tool to
  escape a 0.
- ``warned-unchanged``: no errexit, names a missing ``othertool``, writes 1.
  Warned in result.json and counted in the summary; the score stands.
- ``warned-wrong-zero``: no errexit, ``if othertool; then 1; else 0``. A 0
  whose last "not found" command is the warned one: listed in summary.json's
  wrong-0 candidates.

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
BASE = (
    "FROM python:3.12-slim@sha256:"
    "a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1\n"
    "RUN mkdir -p /app && chmod 1777 /app\n"
)
WITH_TOOL = (
    BASE + "RUN mkdir -p /opt/x/bin && printf '#!/bin/sh\\nexit 0\\n' > "
    "/opt/x/bin/mytool && chmod 755 /opt/x/bin/mytool\n"
    "ENV PATH=/opt/x/bin:$PATH\nWORKDIR /app\n"
)
WITHOUT_TOOL = BASE + "WORKDIR /app\n"
NEEDS_TOOL = "#!/bin/bash\nset -e\nmytool --check\necho 1 > /logs/verifier/reward.txt\n"
ANSWER = "echo ok > /app/answer.txt\n"
TASKS = {
    "needs-missing-tool": (WITHOUT_TOOL, NEEDS_TOOL, ANSWER),
    "tool-present": (WITH_TOOL, NEEDS_TOOL, ANSWER),
    "agent-deletes-tool": (WITH_TOOL, NEEDS_TOOL, "rm -f /opt/x/bin/mytool\n" + ANSWER),
    "agent-shadows-tool": (
        WITH_TOOL,
        NEEDS_TOOL,
        "printf '#!/bin/sh\\nexit 3\\n' > /opt/x/bin/mytool\n" + ANSWER,
    ),
    "warned-unchanged": (
        WITHOUT_TOOL,
        "#!/bin/bash\nothertool --version\necho 1 > /logs/verifier/reward.txt\n",
        ANSWER,
    ),
    # The task's tests are root's alone after their upload (RUSAGE): docker
    # cp writes the host's uid:gid. test.sh scores 1 only when every file
    # and folder under /tests is 0:0.
    "tests-are-roots": (
        WITHOUT_TOOL,
        "#!/bin/bash\n"
        'if [ -z "$(find /tests \\( ! -user 0 -o ! -group 0 \\) -print)" ]; '
        "then echo 1 > /logs/verifier/reward.txt; "
        "else echo 0 > /logs/verifier/reward.txt; fi\n",
        ANSWER,
    ),
    # test.sh (root) swaps its own output for a link to a host-looking file,
    # or a FIFO, then scores 0 after a warned command: the host reads only a
    # regular file through the held capability, so the trial ends and names
    # no candidate (RUSAGE).
    "stdout-swapped-link": (
        WITHOUT_TOOL,
        "#!/bin/bash\nothertool --run\nrm -f /logs/verifier/test-stdout.txt\n"
        "ln -s /etc/hostname /logs/verifier/test-stdout.txt\n"
        "echo 0 > /logs/verifier/reward.txt\n",
        ANSWER,
    ),
    "stdout-swapped-fifo": (
        WITHOUT_TOOL,
        "#!/bin/bash\nothertool --run\nrm -f /logs/verifier/test-stdout.txt\n"
        "mkfifo /logs/verifier/test-stdout.txt\n"
        "echo 0 > /logs/verifier/reward.txt\n",
        ANSWER,
    ),
    "warned-wrong-zero": (
        WITHOUT_TOOL,
        "#!/bin/bash\nif othertool --run; then echo 1 > /logs/verifier/reward.txt; "
        "else echo 0 > /logs/verifier/reward.txt; fi\n",
        ANSWER,
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
    root = tmp_path_factory.mktemp("preflight") / "tasks"
    for name, (dockerfile, test_sh, solve) in TASKS.items():
        task = root / name
        for sub in ("environment", "solution", "tests"):
            (task / sub).mkdir(parents=True)
        (task / "task.toml").write_text(TASK_TOML)
        (task / "instruction.md").write_text("Write ok to /app/answer.txt.\n")
        (task / "environment" / "Dockerfile").write_text(dockerfile)
        for path, text in (
            (task / "solution" / "solve.sh", "#!/bin/bash\n" + solve),
            (task / "tests" / "test.sh", test_sh),
        ):
            path.write_text(text)
            path.chmod(0o755)
    return root


@pytest.fixture(scope="module")
def job(tasks, tmp_path_factory):
    jobs = tmp_path_factory.mktemp("preflight-jobs")
    run = _bench(
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
    )
    (job_dir,) = [p for p in jobs.iterdir() if p.is_dir()]
    return run, job_dir


def _result(job_dir: Path, name: str) -> tuple[Path, dict]:
    (trial,) = [p for p in job_dir.glob(f"{name}__*") if p.is_dir()]
    return trial, json.loads((trial / "result.json").read_text())


def test_a_pristine_image_without_the_tool_is_refused_before_the_turn(job):
    _, job_dir = job
    trial, result = _result(job_dir, "needs-missing-tool")  # one attempt
    assert result["rewards"] is None
    error = str(result["error"])
    assert "sandbox startup refused: test.sh needs `mytool`" in error, error
    assert "not on the verifier's PATH" in error
    record = result["test_sh_preflight"]
    assert record["refused"] == "mytool"
    assert record["fatal"] == ["mytool"]
    from benchflow.failures import FAULT_WORDS, cause_of

    cause = cause_of(result)
    assert cause is not None and cause.key == "test_sh_command_missing"
    assert FAULT_WORDS[cause.fault] == "task problem"
    # The oracle never ran: no solve.sh output, no answer graded.
    assert not (trial / "agent" / "oracle.txt").exists()
    assert result["verifier_error"] is None


def test_the_tool_on_path_scores_and_records_the_check(job):
    _, job_dir = job
    _, result = _result(job_dir, "tool-present")
    assert result["rewards"] == {"reward": 1.0}, result.get("verifier_error")
    record = result["test_sh_preflight"]
    assert record["checked"] is True and record["missing"] == []
    assert not any(
        "not on the verifier PATH at preflight" in w
        for w in result.get("warnings") or []
    )


@pytest.mark.parametrize("name", ["agent-deletes-tool", "agent-shadows-tool"])
def test_a_tool_the_agent_removes_after_preflight_stays_a_failure(job, name):
    _, job_dir = job
    _, result = _result(job_dir, name)
    assert result["error"] is None or "startup refused" not in str(result["error"])
    assert result["test_sh_preflight"]["checked"] is True
    assert "refused" not in result["test_sh_preflight"]
    # As before EN-715: no usable reward, a failure in every score.
    assert result["rewards"] is None
    assert result["verifier_error"].startswith("verifier wrote no usable reward: ")


def test_a_named_missing_command_is_warned_and_the_score_stands(job):
    _, job_dir = job
    _, result = _result(job_dir, "warned-unchanged")
    assert result["rewards"] == {"reward": 1.0}
    assert "test.sh names `othertool`, not on the verifier PATH at preflight" in (
        result.get("warnings") or []
    )
    assert result["test_sh_preflight"]["missing"] == ["othertool"]
    assert "wrong_zero_candidate" not in result["test_sh_preflight"]


def test_the_warned_command_last_before_a_0_is_a_wrong_zero_candidate(job):
    _, job_dir = job
    _, result = _result(job_dir, "warned-wrong-zero")
    assert result["rewards"] == {"reward": 0.0}
    assert result["test_sh_preflight"]["wrong_zero_candidate"] == "othertool"


def test_the_summary_counts_and_lists_them(job):
    _, job_dir = job
    summary = json.loads((job_dir / "summary.json").read_text())
    block = summary["test_sh_preflight"]
    assert block["refused"] == 1
    assert block["warned"] == 4
    assert [c["task"] for c in block["wrong_zero_candidates"]] == ["warned-wrong-zero"]
    assert block["wrong_zero_candidates"][0]["command"] == "othertool"
    assert block["wrong_zero_candidates"][0]["trial"].startswith("warned-wrong-zero__")
    assert "test-stdout.txt" in block["wrong_zero_source"]
    assert summary["passed"] == 3  # tool-present, warned-unchanged, tests-are-roots


@pytest.mark.parametrize("name", ["stdout-swapped-link", "stdout-swapped-fifo"])
def test_a_swapped_test_stdout_is_never_followed(job, name):
    _, job_dir = job
    _, result = _result(job_dir, name)
    assert result["rewards"] == {"reward": 0.0}
    assert result["test_sh_preflight"]["missing"] == ["othertool"]
    assert "wrong_zero_candidate" not in result["test_sh_preflight"]


def test_the_uploaded_tests_are_root_s_alone(job):
    _, job_dir = job
    _, result = _result(job_dir, "tests-are-roots")
    assert result["rewards"] == {"reward": 1.0}, result.get("verifier_error")


def test_equivalence_reports_the_refusal_unscored(tasks, tmp_path):
    out = tmp_path / "equivalence.json"
    run = _bench(
        "check",
        str(tasks / "needs-missing-tool"),
        "--level",
        "equivalence",
        "--sandbox",
        "docker",
        "--report-output",
        str(out),
    )
    assert out.exists(), run.stdout + run.stderr
    report = json.loads(out.read_text())
    assert report["status"] == "unscored"
    assert "test.sh needs `mytool`" in report["reason"]
