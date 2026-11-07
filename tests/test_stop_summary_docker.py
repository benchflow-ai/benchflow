"""An orderly stop writes a partial summary, on real Docker (DFR-p3-11).

On 9931d7815 a run stopped by SIGTERM (or Ctrl-C) wrote no summary.json and
no --summary-out file. Now, once its trials' cleanup has run, it writes both,
atomically, over the trials that finished, with the job's completion state:
which planned tasks had started (interrupted) and which had not. While it
runs, `bench results` calls its started trials running, with no resume
line (DFR-p3-10). After it, `bench train stream` exits 1 with the resume
step, and `--summary-out` counts the trial it ran as ran (FIX-49 r3).

Two tasks at -j 2: one finishes, the other's oracle sleeps until the stop.
Each test's image carries its own ``target=FIX49STOP-…`` label, so the
cleanup touches only its own containers and images. Skipped when no Docker
daemon is available.
"""

from __future__ import annotations

import json
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from tests.test_dx_a_container import _docker_ready, _env, _task

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]


@pytest.fixture
def label():
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = "FIX49STOP-" + uuid.uuid4().hex[:8]
    yield name
    for kind in (("ps", "-aq"), ("image", "ls", "-q")):
        ids = subprocess.run(
            ["docker", *kind, "--filter", f"label=target={name}"],
            capture_output=True,
            text=True,
        ).stdout.split()
        if ids:
            rm = ["rm", "-f"] if kind[0] == "ps" else ["rmi", "-f"]
            subprocess.run(["docker", *rm, *ids], capture_output=True)


def _bench(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "benchflow.cli.main", *args],
        cwd=root,
        env=_env(root),
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize(
    ("signum", "code", "how"),
    [
        (signal.SIGTERM, 143, "Terminated (SIGTERM)"),
        (signal.SIGINT, 130, "Interrupted (Ctrl-C)"),
    ],
)
def test_a_stopped_run_writes_its_partial_summary(tmp_path, label, signum, code, how):
    dockerfile = (
        "FROM python:3.12-slim\nWORKDIR /app\n"
        "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts\n"
        f'LABEL target="{label}"\n'
    )
    _task(tmp_path, "quick", dockerfile=dockerfile)
    _task(tmp_path, "slow", dockerfile=dockerfile, solve="sleep 900\n")
    job = tmp_path / "jobs" / "stop"
    out = tmp_path / "ci" / "run.json"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "benchflow.cli.main",
            "run",
            "tasks",
            "-j",
            "2",
            "--jobs-dir",
            "jobs",
            "--job",
            "stop",
            "--summary-out",
            str(out),
        ],
        cwd=tmp_path,
        env=_env(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 900
        while not list(job.glob("quick__*/result.json")):
            assert proc.poll() is None, proc.stdout.read() if proc.stdout else ""
            assert time.monotonic() < deadline, "quick never finished"
            time.sleep(1)
        while not list(job.glob("slow__*/config.json")):
            assert time.monotonic() < deadline, "slow never started"
            time.sleep(1)
        # While it runs: running, not interrupted, and nothing to resume.
        live = _bench(tmp_path, "results", "jobs/stop")
        assert f"Running (process {proc.pid}):" in live.stdout, live.stdout
        assert "1 running" in live.stdout
        assert "bench eval resume" not in live.stdout
        proc.send_signal(signum)
        output, _ = proc.communicate(timeout=300)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == code, output
    assert f"{how}: stopped 1 running trial" in output, output
    assert "Partial summary (what finished):" in output, output
    summary = json.loads((job / "summary.json").read_text())
    assert summary["stopped"] is True
    assert (summary["total"], summary["passed"]) == (1, 1)
    assert (summary["planned"], summary["not_run"]) == (2, 1)
    assert (summary["interrupted"], summary["not_started"]) == (1, 0)
    assert summary["interrupted_tasks"] == ["slow"]
    doc = json.loads(out.read_text())
    assert doc["stopped"] is True and doc["exit_code"] == code
    assert (doc["total"], doc["passed"], doc["not_run"], doc["interrupted"]) == (
        1,
        1,
        1,
        1,
    )
    # The one trial this invocation ran is counted as ran, not reused
    # (FIX-49 r3 L2: 6c7426dcb said ran 0, reused 1).
    assert (doc["ran"], doc["reused"]) == (1, 0), doc
    # No temporary files of the atomic writes are left beside them.
    assert sorted(p.name for p in out.parent.iterdir()) == ["run.json"]
    after = _bench(tmp_path, "results", "jobs/stop")
    assert "1 interrupted" in after.stdout and "bench eval resume" in after.stdout
    listed = _bench(tmp_path, "eval", "list", "jobs")
    assert "stopped: 1 interrupted, 0 not started" in listed.stdout, listed.stdout
    # A trainer reads the stop as a stop, not as the job's end (FIX-49 r3
    # M1: 6c7426dcb exited 0, "1 rollouts streamed").
    stream = _bench(tmp_path, "train", "stream", "jobs/stop", "--timeout", "20")
    assert stream.returncode == 1, stream.stderr
    assert len(stream.stdout.splitlines()) == 1
    said = " ".join(stream.stderr.split())
    assert "the job was stopped before it finished" in said, said
    assert "next: `bench eval resume" in said, said
