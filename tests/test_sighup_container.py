"""A closed terminal stops a run cleanly, on real Docker (SIGHUP-1).

The terminal window closes: its controlling process (the session leader, a
shell in real life) dies and the terminal is hung up, so the kernel sends
SIGHUP to the foreground process group, where ``bench run`` is. On
5653de15 bench died at once and its sandboxes stayed up. Now it stops as on
SIGTERM: it stops the trials, removes the run's sandboxes (the warm pool's
prepared ones too) and exits 129. Under ``nohup`` the run goes on.

The terminal is closed for real: an interactive bash runs a recorder as its
foreground job, the recorder starts bench and writes down its exit code (it
catches SIGHUP, so it outlives the hangup), and the pty's master is closed.
The job then gets SIGHUP twice, the kernel's and the shell's. The
single-trial case also runs the other way a session ends: its leader
SIGKILLed, then the master closed (one SIGHUP). Each test labels its image with its own
``target=SIGHUP1-…``, so the leftover check and the cleanup touch only its own
containers and images. Skipped when no Docker daemon is available.
"""

from __future__ import annotations

import json
import os
import pty
import select
import shlex
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

# The session leader: start the recorder, then wait to be killed.
_LEADER = """
import subprocess, sys, time
subprocess.Popen([sys.executable, "-c", sys.argv[1], *sys.argv[2:]])
time.sleep(3600)
"""

# The recorder: outlive the hangup, run bench, write its exit code.
_RECORDER = """
import json, signal, subprocess, sys, time
signal.signal(signal.SIGHUP, lambda *a: None)
rc_path, argv = sys.argv[1], sys.argv[2:]
started = time.monotonic()
code = subprocess.call(argv)
with open(rc_path, "w") as f:
    json.dump({"rc": code, "seconds": time.monotonic() - started}, f)
"""


@pytest.fixture
def label():
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = "SIGHUP1-" + uuid.uuid4().hex[:8]
    yield name
    # Only what this test made: containers and images with its own label.
    ids = _ids("ps", "-aq", name)
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
    images = _ids("image", "ls", "-q", name)
    if images:
        subprocess.run(["docker", "rmi", "-f", *images], capture_output=True)


def _ids(*args: str) -> list[str]:
    *cmd, name = args
    done = subprocess.run(
        ["docker", *cmd, "--filter", f"label=target={name}"],
        capture_output=True,
        text=True,
    )
    return sorted(set(done.stdout.split()))


def _running(name: str) -> int:
    return len(_ids("ps", "-q", name))


def _dockerfile(name: str) -> str:
    return (
        "FROM python:3.12-slim\nWORKDIR /app\n"
        "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts\n"
        f'LABEL target="{name}"\n'
    )


def _hang_up_after(
    root: Path,
    bench_args: list[str],
    name: str,
    *,
    sandboxes: int,
    nohup: bool = False,
    how: str = "close",
) -> dict:
    """Run bench in a pty session; once ``sandboxes`` are up (and 5 s more),
    hang the terminal up. Returns the recorder's record, with the seconds
    from the hangup to bench's exit.

    ``how="close"``: an interactive bash runs the recorder (and bench) as its
    foreground job, and the terminal's master is closed, as sshd, tmux or a
    terminal window does. The job gets SIGHUP twice: the kernel's, and the
    shell's resend to its jobs. ``how="leader-killed"``: the session's leader
    is SIGKILLed, then the master closed; the job gets one SIGHUP.
    """
    env = _env(root)
    env.pop("BENCHFLOW_NO_PROGRESS")  # the live display, as in a terminal
    rc_path = root / "rc.json"
    bench = [sys.executable, "-m", "benchflow.cli.main", *bench_args]
    if nohup:
        bench = ["nohup", *bench]
    if how == "close":
        recorder = root / "recorder.py"
        recorder.write_text(_RECORDER)
        argv = ["/bin/bash", "--norc", "--noprofile", "-i"]
        line = shlex.join([sys.executable, str(recorder), str(rc_path), *bench])
    else:
        argv = [sys.executable, "-c", _LEADER, _RECORDER, str(rc_path), *bench]
        line = ""
    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover - the child, the session's leader
        for sig in (signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_DFL)
        os.chdir(root)
        os.execve(argv[0], argv, env)
    if line:
        os.write(fd, line.encode() + b"\n")
    out = bytearray()
    started = time.monotonic()
    up_at = None
    try:
        while True:
            if select.select([fd], [], [], 0.5)[0]:
                try:
                    out += os.read(fd, 65536)
                except OSError:
                    break
            if up_at is None and _running(name) >= sandboxes:
                up_at = time.monotonic()
            if up_at is not None and time.monotonic() - up_at >= 5:
                break
            assert time.monotonic() - started < 240, out.decode(errors="replace")
    finally:
        if how != "close":
            os.kill(pid, signal.SIGKILL)  # the controlling process dies
        os.close(fd)  # the terminal goes away
        os.waitpid(pid, 0)  # the shell dies of the hangup
    hung_up = time.monotonic()
    while not rc_path.exists():
        assert time.monotonic() - hung_up < 240, "bench never exited"
        time.sleep(0.5)
    time.sleep(0.2)  # the recorder's write
    record = json.loads(rc_path.read_text())
    record["after_hangup"] = time.monotonic() - hung_up
    return record


@pytest.mark.parametrize("how", ["close", "leader-killed"])
def test_a_closed_terminal_stops_the_trial_and_removes_its_sandbox(
    tmp_path, label, how
):
    _task(
        tmp_path,
        "slow",
        dockerfile=_dockerfile(label),
        solve="sleep 120\necho hello > /app/out.txt\n",
    )
    record = _hang_up_after(
        tmp_path,
        [
            "run",
            "tasks/slow",
            "--jobs-dir",
            "jobs",
            "--job",
            "hup",
            "--concurrency",
            "1",
        ],
        label,
        sandboxes=1,
        how=how,
    )
    assert record["rc"] == 129, record
    assert record["after_hangup"] < 60, record  # not the 120 s solve
    assert _running(label) == 0
    assert _ids("ps", "-aq", label) == []
    receipts = list((tmp_path / "jobs" / "hup").glob("*/budget-cancelled.json"))
    assert receipts and json.loads(receipts[0].read_text())["reason"] == "interrupted"


def test_a_closed_terminal_drains_the_warm_pool_too(tmp_path, label):
    """Guards 9b0320f: two trials and one immutable-image warm sandbox drain."""
    built = subprocess.run(
        ["docker", "build", "-q", "-"],
        input=_dockerfile(label),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert built.returncode == 0, built.stderr
    image = built.stdout.strip()
    assert image.startswith("sha256:") and len(image) == 71
    for name in ("slow-a", "slow-b", "slow-c"):
        task = _task(
            tmp_path,
            name,
            dockerfile=_dockerfile(label),
            solve="sleep 120\necho hello > /app/out.txt\n",
        )
        definition = task / "task.md"
        definition.write_text(
            definition.read_text().replace(
                "verifier:\n", f'sandbox:\n  docker_image: "{image}"\nverifier:\n', 1
            )
        )
    record = _hang_up_after(
        tmp_path,
        [
            "run",
            "tasks",
            "--jobs-dir",
            "jobs",
            "--job",
            "hup-pool",
            "--concurrency",
            "2",
            "--warm-pool",
            "1",
        ],
        label,
        sandboxes=3,
    )
    assert record["rc"] == 129, record
    assert record["after_hangup"] < 90, record
    assert _ids("ps", "-aq", label) == []


def test_a_closed_terminal_stops_a_sharded_run_and_its_workers(tmp_path, label):
    """Two worker processes with two trials each. The workers are in the
    terminal's foreground job, so each gets the kernel's SIGHUP, the shell's
    resend and the parent's SIGTERM. On 06cd177e that third signal
    force-exited every worker without its cleanup: bench exited 129 with all
    their sandboxes running (the review's M1)."""
    for name in ("slow-a", "slow-b", "slow-c", "slow-d"):
        _task(
            tmp_path,
            name,
            dockerfile=_dockerfile(label),
            solve="sleep 120\necho hello > /app/out.txt\n",
        )
    record = _hang_up_after(
        tmp_path,
        [
            "run",
            "tasks",
            "--jobs-dir",
            "jobs",
            # --job-name is refused with --worker-concurrency.
            "--concurrency",
            "4",
            "--worker-concurrency",
            "2",
        ],
        label,
        sandboxes=4,
    )
    assert record["rc"] == 129, record
    assert record["after_hangup"] < 90, record
    assert _running(label) == 0
    assert _ids("ps", "-aq", label) == []


def test_under_nohup_the_run_survives_a_closed_terminal(tmp_path, label):
    _task(
        tmp_path,
        "short",
        dockerfile=_dockerfile(label),
        solve="sleep 25\necho hello > /app/out.txt\n",
    )
    record = _hang_up_after(
        tmp_path,
        [
            "run",
            "tasks/short",
            "--jobs-dir",
            "jobs",
            "--job",
            "nohup",
            "--concurrency",
            "1",
        ],
        label,
        sandboxes=1,
        nohup=True,
    )
    assert record["rc"] == 0, (record, (tmp_path / "nohup.out").read_text()[-2000:])
    results = list((tmp_path / "jobs" / "nohup").glob("*/result.json"))
    assert results, sorted(p.name for p in (tmp_path / "jobs").rglob("*"))
    assert json.loads(results[0].read_text())["rewards"] == {"reward": 1.0}
    assert _ids("ps", "-aq", label) == []
