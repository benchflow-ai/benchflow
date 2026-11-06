"""DX lane A, end to end: the dogfooders' repros through ``bench run`` (VM Docker).

Each run is a one-change copy of a hello task, as in
the dogfooders' repros, and each test reads
what a user sees at the default log level: the cause, whose problem, the
next step, no traceback, and no retry of a failure that would repeat.
Skipped without a Docker daemon; the images are built from
``python:3.12-slim`` and labelled DX-a.
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
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

_DOCKERFILE = (
    "FROM python:3.12-slim\nWORKDIR /app\n"
    "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts\n"
    'LABEL target="DX-a"\n'
)
_PASS = (
    "#!/bin/bash\n"
    "if grep -q hello /app/out.txt 2>/dev/null; then echo 1.0 > "
    "/logs/verifier/reward.txt; else echo 0.0 > /logs/verifier/reward.txt; fi\n"
)


def _docker_ready() -> bool:
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except Exception:
        return False
    return probe.returncode == 0


@pytest.fixture(autouse=True)
def _needs_docker():
    if not _docker_ready():
        pytest.skip("docker daemon not running")


def _task(
    root: Path,
    name: str,
    *,
    agent_timeout: int = 300,
    verifier_timeout: int = 120,
    dockerfile: str = _DOCKERFILE,
    solve: str = "echo hello > /app/out.txt\n",
    test: str = _PASS,
) -> Path:
    task = root / "tasks" / name
    for folder in ("environment", "oracle", "verifier"):
        (task / folder).mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(dockerfile)
    (task / "oracle" / "solve.sh").write_text("#!/bin/bash\n" + solve)
    (task / "verifier" / "test.sh").write_text(test)
    (task / "task.md").write_text(
        f'---\nschema_version: "1.3"\nagent:\n  timeout_sec: {agent_timeout}\n'
        f"verifier:\n  timeout_sec: {verifier_timeout}\n---\n# {name}\n\n"
        "## prompt\n\nWrite the word hello into the file /app/out.txt.\n"
    )
    return task


def _env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        NO_COLOR="1",
        COLUMNS="160",
        BENCHFLOW_NO_PROGRESS="1",
        BENCHFLOW_LOG_DIR=str(root / "logs"),
    )
    env.pop("BENCHFLOW_LOG_LEVEL", None)
    return env


def _bench(root: Path, *args: str, timeout: int = 600) -> tuple[int, str, float]:
    started = time.monotonic()
    done = subprocess.run(
        [sys.executable, "-m", "benchflow.cli.main", *args],
        cwd=root,
        env=_env(root),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return done.returncode, done.stdout + done.stderr, time.monotonic() - started


def _result(job: Path) -> dict:
    found = sorted(job.rglob("result.json"))
    assert found, f"no result.json under {job}"
    return json.loads(found[-1].read_text())


def _no_traceback(out: str) -> None:
    # Lines quoted from the task's own output start with "| ".
    own = [line for line in out.splitlines() if not line.strip().startswith("| ")]
    assert not [line for line in own if "Traceback" in line], out
    assert "This is a bug in BenchFlow" not in out


def test_a_broken_dockerfile_is_the_tasks_at_its_line_once(tmp_path):
    """DF-p2-task-author-5 / DF-p5-cli-craft-7, -8: on 66471e95 a 20-frame
    traceback and `other error: RuntimeError: Docker compose command failed
    ... (infrastructure problem)`."""
    _task(
        tmp_path,
        "e-build",
        dockerfile=_DOCKERFILE + "RUN apt-get install -y no-such-package-xyz\n",
    )
    job = tmp_path / "jobs" / "e-build"
    code, out, _ = _bench(
        tmp_path, "run", "tasks/e-build", "--jobs-dir", "jobs", "--job", "e-build"
    )
    assert code == 1, out
    _no_traceback(out)
    assert "Retrying" not in out
    assert (
        "1 errored: the task's image did not build: environment/Dockerfile:5 "
        "`RUN apt-get install -y no-such-package-xyz` exited with code 100 "
        "(E: Unable to locate package no-such-package-xyz) (task problem): e-build"
    ) in out
    assert "next: fix that line in the task's environment/Dockerfile; " in out
    hint = shlex.join(["docker", "build", str(tmp_path / "tasks/e-build/environment")])
    assert f"`{hint}` prints the whole build log" in out
    assert "`bench explain image_build` says more" in out
    assert "docker-compose-base.yaml" not in out
    assert _result(job)["error_category"] == "image_build"


def test_an_oracle_out_of_time_is_a_timeout_run_once(tmp_path):
    """DF-p5-cli-craft-1: on 66471e95 three tracebacks, three attempts (36 s
    for a 5 s budget) and `the sandbox or its transport failed
    (infrastructure problem)`."""
    _task(tmp_path, "agent-timeout", agent_timeout=3, solve="sleep 60\n")
    job = tmp_path / "jobs" / "agent-timeout"
    code, out, seconds = _bench(
        tmp_path,
        "run",
        "tasks/agent-timeout",
        "--jobs-dir",
        "jobs",
        "--job",
        "agent-timeout",
    )
    assert code == 1, out
    _no_traceback(out)
    assert "Retrying" not in out
    assert "1 errored: the agent ran out of time (agent problem): agent-timeout" in out
    assert "next: raise the task's agent timeout" in out
    assert "infrastructure problem" not in out
    result = _result(job)
    assert result["error_category"] == "timeout"
    assert (
        "oracle/solve.sh timed out after 3s, the task's agent timeout"
        in (result["error"])
    )
    assert seconds < 60


def test_a_verifier_timeout_runs_once(tmp_path):
    """DF-p5-cli-craft-17: on 66471e95 the trial ran three times."""
    _task(
        tmp_path,
        "verifier-timeout",
        verifier_timeout=3,
        test="#!/bin/bash\nsleep 60\n",
    )
    code, out, _ = _bench(
        tmp_path,
        "run",
        "tasks/verifier-timeout",
        "--jobs-dir",
        "jobs",
        "--job",
        "verifier-timeout",
    )
    assert code == 0, out
    _no_traceback(out)
    assert "Retrying" not in out
    assert out.count("Starting environment: verifier-timeout") == 1
    assert "Verifier scored 0: verifier timed out after 3.0s" in out
    result = _result(tmp_path / "jobs" / "verifier-timeout")
    assert result["rewards"] == {"reward": 0.0}
    assert result["verifier_error"] is None


def test_a_crashed_verifier_says_what_test_sh_did_and_quotes_it(tmp_path):
    """DF-p2-task-author-11: on 66471e95 `accepting reward`, then
    `RewardFileEmptyError: Reward file is empty at <reachable:...>`, and a
    next step of a check that passes."""
    _task(
        tmp_path,
        "b1-crash",
        test=(
            "#!/bin/bash\npython3 -c \"open('/logs/verifier/reward.txt', 'w')"
            '.write(str(score()))"\n'
        ),
    )
    job = tmp_path / "jobs" / "b1-crash"
    code, out, _ = _bench(
        tmp_path, "run", "tasks/b1-crash", "--jobs-dir", "jobs", "--job", "b1-crash"
    )
    assert code == 0, out
    _no_traceback(out)
    assert "accepting reward" not in out
    assert "<reachable:" not in out
    assert (
        "Verifier scored 0: test.sh exited 1 and wrote an empty verifier/reward.txt"
    ) in out
    assert "1 scored (0 passed, 1 failed), 0 unscored" in out
    stdout = next(job.glob("*/verifier/test-stdout.txt")).read_text()
    assert "NameError: name 'score' is not defined" in stdout
    assert "bench tasks check" not in out
    assert _result(job)["verifier_error_category"] is None


def _pty_run(argv: list[str], cwd: Path, env: dict, after: float) -> tuple[int, str]:
    """Run ``argv`` in a terminal, press Ctrl-C after ``after`` seconds.

    The child starts with SIGINT and SIGQUIT at their defaults, as a shell
    starts a foreground job. A non-interactive shell starts an ``&`` job with
    both ignored, an ignored signal stays ignored across exec, and Python then
    installs no Ctrl-C handler: a suite launched that way (``nohup … &``) saw
    the run ignore Ctrl-C and exit 0 (CTRLC-1).
    """
    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover - the child
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGQUIT, signal.SIG_DFL)
        os.chdir(cwd)
        os.execve(argv[0], argv, env)
    out = bytearray()
    started, sent = time.monotonic(), False
    while True:
        if not sent and time.monotonic() - started >= after:
            os.write(fd, b"\x03")
            sent = True
        ready, _, _ = select.select([fd], [], [], 0.2)
        if ready:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
        if time.monotonic() - started > 300:
            os.kill(pid, 9)
            break
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status), out.decode(errors="replace")


def test_ctrl_c_says_what_it_stopped_and_how_to_resume(tmp_path):
    """DF-p5-cli-craft-10: on 66471e95 nothing was printed, the last frame
    read `starting…`, and `bench results` called the job not a job."""
    _task(tmp_path, "slow", solve="sleep 120\necho hello > /app/out.txt\n")
    env = _env(tmp_path)
    env.pop("BENCHFLOW_NO_PROGRESS")  # the live display, as in a terminal
    code, out = _pty_run(
        [
            sys.executable,
            "-m",
            "benchflow.cli.main",
            "run",
            "tasks/slow",
            "--jobs-dir",
            "jobs",
            "--job",
            "int4",
        ],
        tmp_path,
        env,
        after=15,
    )
    assert code == 130, out
    _no_traceback(out)
    assert "Interrupted (Ctrl-C): stopped 1 running trial and removed its sandbox." in (
        out
    )
    assert "Resume with: bench eval resume ./jobs/int4" in out
    tail = out[out.rindex("benchflow") :]
    assert "starting…" not in tail
    job = tmp_path / "jobs" / "int4"
    receipts = list(job.glob("*/budget-cancelled.json"))
    assert receipts and json.loads(receipts[0].read_text())["reason"] == "interrupted"
    code, out, _ = _bench(tmp_path, "results", "jobs/int4")
    assert code == 1, out
    assert "is a job that was stopped before any trial finished" in out
    assert "next: `bench eval resume ./jobs/int4` runs them" in out


def _sudo_ok() -> bool:
    return (
        shutil.which("sudo") is not None
        and subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0
    )


def test_a_full_disk_is_said_as_such_and_leaves_no_empty_summary(tmp_path):
    """DF-p5-cli-craft-5: on 66471e95 `verifier crashed`, `This is a bug in
    BenchFlow`, and a 0-byte summary.json."""
    if not _sudo_ok():
        pytest.skip("needs sudo to mount a 1 MB tmpfs")
    _task(tmp_path, "hello")
    tiny = tmp_path / "tiny"
    tiny.mkdir()
    subprocess.run(
        ["sudo", "-n", "mount", "-t", "tmpfs", "-o", f"size=1m,uid={os.getuid()}",
         "dxa-tiny", str(tiny)],
        check=True,
    )  # fmt: skip
    try:
        (tiny / "fill").write_bytes(b"\0" * (1000 * 1024))  # as the dogfood's dd
        code, out, _ = _bench(
            tmp_path, "run", "tasks/hello", "--jobs-dir", str(tiny / "jobs")
        )
        assert code == 1, out
        _no_traceback(out)
        assert "the disk that holds the job folder is full" in out
        assert "next: free space on that disk, or pass another --jobs-dir" in out
        assert not [p for p in tiny.rglob("summary.json") if p.stat().st_size == 0]
    finally:
        subprocess.run(["sudo", "-n", "umount", str(tiny)], capture_output=True)
