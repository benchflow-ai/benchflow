"""Kept services on real Docker: shells with job control and unredirected output.

Deterministic tier (``tests/integration/README.md``): real sandboxes, no
model (the ``oracle`` agent runs each task's solution as the agent user,
``[agent] user = "agent"``). Runs where Docker answers and skips with a reason
otherwise. Each task starts the same small HTTP service on port 5000, names
the port in ``x-keep-services``, and its test reads the service; a kept
service scores 1.

- K2p-R1-A-1: the service is a job of a job-control shell (``set -m``, as a
  terminal tool's interactive shell runs commands), in a process group of its
  own, and the shell is still alive at the kill. If the kill freezes the
  service and then kills the shell, the kernel sends the orphaned group
  SIGHUP and SIGCONT: a plain ``&`` service dies (score 0), and a nohup'd one
  runs on unfrozen through the hardening (the release records it under
  ``kept_services_unfrozen``). The shell dies in the SIGTERM grace in one
  task and only in the SIGKILL rounds (it ignores SIGTERM, as an interactive
  bash does) in the others.
- K2p-R2-A-1: the service is a plain job typed into an interactive bash on a
  terminal (``term.py`` holds the pty, as a tmux server does). When the
  terminal ends in the SIGTERM grace, the hung-up shell passes SIGHUP on to
  its jobs, so the shell must be gone first.
- K2p-R2-B-1: the service keeps a pool of spawned workers that do not hold
  its port and replaces any that die. If it runs while the kill ends them,
  it starts new ones every round and the kill fails (the trial errors).
- K2p-R1-A-2: solve.sh starts the service with no redirect of its own, so it
  inherits the oracle's output file; with that file in /logs the release
  killed the service as a holder of /logs (score 0).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic
# Whole trials on real Docker: the exec channel and the side-by-side steps
# before the verifier stay on, as in a run (tests/conftest.py).
DOCKER_RUN_DEFAULTS = True

SANDBOX, SKIP_REASON = h.select_sandbox()

TASK_TOML = """version = "1.0"

[metadata]
author_name = "benchflow"
difficulty = "easy"
category = "fixture"
x-keep-services = [{ port = 5000, name = "fixture service" }]

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 120.0
user = "agent"

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
storage_mb = 4096
"""
INSTRUCTION = (
    "Start the HTTP service at /opt/svc/server.py on port 5000 and leave it running.\n"
)
SERVER_PY = '''"""GET / answers "kept"; ``python3 server.py <port>``."""

import http.server
import sys


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"kept\\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


http.server.ThreadingHTTPServer(("0.0.0.0", int(sys.argv[1])), Handler).serve_forever()
'''
# A service with a pool of workers started by the spawn method: the workers
# do not hold the port, and the pool starts new ones when one dies.
POOL_SERVER_PY = '''"""GET / answers "kept"; a pool of 2 spawned workers; ``python3 pool_server.py <port>``."""

import http.server
import multiprocessing
import sys


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"kept\\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    pool = multiprocessing.Pool(2)
    pool.apply(abs, (-1,))
    http.server.ThreadingHTTPServer(("0.0.0.0", int(sys.argv[1])), Handler).serve_forever()
'''
# A terminal as a tmux server keeps one: an interactive bash on a pty of its
# own (its session), the line typed into it; the pty closes when it exits.
TERM_PY = '''"""``python3 term.py <line>``: type <line> into an interactive bash on a pty."""

import os
import pty
import sys

pid, fd = pty.fork()
if pid == 0:
    os.execvp("bash", ["bash", "--norc", "--noprofile", "-i"])
os.write(fd, sys.argv[1].encode() + b"\\n")
while True:
    try:
        if not os.read(fd, 4096):
            break
    except OSError:
        break
'''
# A launcher that stays the service's parent and passes SIGTERM on to it, as
# npm's run-script, uv run and pm2 do.
WRAP_PY = '''"""``python3 wrap.py``: run server.py on port 5000, pass SIGTERM on to it."""

import signal
import subprocess

child = subprocess.Popen(["python3", "/opt/svc/server.py", "5000"])
signal.signal(signal.SIGTERM, lambda *_: child.send_signal(signal.SIGTERM))
child.wait()
'''
# K2-R4-B-1: a service that, once it runs during grading, reads every
# process's command line and environment in the sandbox for the verifier's
# key, as is or inside a base64 run (how an exec carries an env); GET /leak
# answers what it found. The key is in its source, not its own command line.
FAKE_KEY = "bf-fake-judge-key-not-a-secret-7f3a91"
SNOOP_PY = '''"""GET / answers "kept", GET /leak what the scan found; ``python3 snoop.py <port>``."""

import base64
import http.server
import os
import re
import sys
import threading
import time

KEY = b"__KEY__"
RUN = re.compile(rb"[A-Za-z0-9+/=]{16,}")
found = []


def holds(raw):
    if KEY in raw:
        return True
    for run in RUN.findall(raw):
        try:
            if KEY in base64.b64decode(run + b"=" * (-len(run) % 4)):
                return True
        except ValueError:
            pass
    return False


def scan():
    me = str(os.getpid())
    while True:
        for pid in os.listdir("/proc"):
            if not pid.isdigit() or pid == me:
                continue
            for name in ("cmdline", "environ"):
                try:
                    with open(f"/proc/{pid}/{name}", "rb") as f:
                        raw = f.read()
                except OSError:
                    continue
                if holds(raw) and not found:
                    found.append(f"{name} of pid {pid}")
        time.sleep(0.02)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/leak":
            body = (("leaked: " + found[0]) if found else "clean").encode() + b"\\n"
        else:
            body = b"kept\\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


threading.Thread(target=scan, daemon=True).start()
http.server.ThreadingHTTPServer(("0.0.0.0", int(sys.argv[1])), Handler).serve_forever()
'''.replace("__KEY__", FAKE_KEY)
# Its test.sh: the key reached test.sh, and nothing of the agent's read it.
SNOOP_TEST_SH = """#!/bin/bash
sleep 2
leak=$(python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:5000/leak', timeout=5).read().decode().strip())" 2>/dev/null)
echo "scan: $leak" > /logs/verifier/leak.txt
if [ "$leak" = clean ] && [ "$JUDGE_API_KEY" = "__KEY__" ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
""".replace("__KEY__", FAKE_KEY)
# The image K2's keep-service fixtures use (tests/fixtures/keep_services).
DOCKERFILE = (
    "FROM python:3.12-slim@sha256:"
    "a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1\n"
    "COPY server.py pool_server.py term.py wrap.py snoop.py /opt/svc/\n"
    "WORKDIR /app\n"
    "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts\n"
)
TEST_SH = """#!/bin/bash
out=$(python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:5000/', timeout=5).read().decode().strip())" 2>/dev/null)
if [ "$out" = kept ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""
WAIT_FOR_SERVICE = """for _ in $(seq 1 50); do
    python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/', timeout=1)" 2>/dev/null && exit 0
    sleep 0.1
done
exit 0
"""


def _jobshell(launch: str, *, ignore_term: bool) -> str:
    """A solve.sh whose service is a job of a ``set -m`` shell that stays
    alive (a session of its own, as a terminal's shell has)."""
    trap = 'trap "" TERM; ' if ignore_term else ""
    return (
        "#!/bin/bash\n"
        f"setsid bash -c 'set -m; {trap}{launch} >/tmp/service.out 2>&1 & "
        "while :; do sleep 0.2; done' </dev/null >/dev/null 2>&1 &\n" + WAIT_FOR_SERVICE
    )


def _terminal(line: str) -> str:
    """A solve.sh that types ``line`` into an interactive bash on a terminal
    that stays open (``term.py``, a session of its own, as a tmux server is)."""
    return (
        "#!/bin/bash\n"
        f"setsid python3 /opt/svc/term.py '{line}' </dev/null >/dev/null 2>&1 &\n"
        + WAIT_FOR_SERVICE
    )


SOLVES = {
    # The shell dies in the SIGKILL rounds; a plain job.
    "jobshell-kill": _jobshell("python3 /opt/svc/server.py 5000", ignore_term=True),
    # The shell dies of the SIGTERM grace; a plain job.
    "jobshell-term": _jobshell("python3 /opt/svc/server.py 5000", ignore_term=False),
    # The shell dies in the SIGKILL rounds; a nohup'd job.
    "jobshell-nohup": _jobshell(
        "nohup python3 /opt/svc/server.py 5000", ignore_term=True
    ),
    # No redirect: the service's stdout and stderr are the oracle's output.
    "no-redirect": "#!/bin/bash\npython3 /opt/svc/server.py 5000 &\n"
    + WAIT_FOR_SERVICE,
    # A plain job typed into a terminal's interactive bash; the terminal ends
    # in the grace and the shell would pass its hangup on to the job.
    "terminal-plain": _terminal(
        "python3 /opt/svc/server.py 5000 >/tmp/service.out 2>&1 &"
    ),
    # A pool of spawned workers that do not hold the port; the master
    # replaces any that die while it runs.
    "pool-spawn": "#!/bin/bash\nnohup python3 /opt/svc/pool_server.py 5000 "
    ">/tmp/service.out 2>&1 &\n" + WAIT_FOR_SERVICE,
    # Both: the pool's master is a plain job typed into a terminal.
    "terminal-pool-spawn": _terminal(
        "python3 /opt/svc/pool_server.py 5000 >/tmp/service.out 2>&1 &"
    ),
    # K2p-R3-A-1: a launcher that passes SIGTERM on stays the service's
    # parent in its group and session.
    "launcher-forwards-term": "#!/bin/bash\nsetsid nohup python3 /opt/svc/wrap.py "
    ">/tmp/service.out 2>&1 </dev/null &\n" + WAIT_FOR_SERVICE,
}
# A plain job typed into a terminal with no redirect: it outlives the kill,
# but its terminal is hung up once the terminal ends (K2p-R3-A-2).
ON_TERMINAL = _terminal("python3 /opt/svc/server.py 5000 &")


@pytest.fixture
def require_docker() -> None:
    if SANDBOX != "docker":
        reason = SKIP_REASON if SANDBOX is None else f"Docker-only scenario ({SANDBOX})"
        pytest.skip(reason)


def _task(root: Path, name: str, solve: str) -> Path:
    task = root / name
    for sub in ("environment", "solution", "tests"):
        (task / sub).mkdir(parents=True)
    (task / "task.toml").write_text(TASK_TOML)
    (task / "instruction.md").write_text(INSTRUCTION)
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (task / "environment" / "server.py").write_text(SERVER_PY)
    (task / "environment" / "pool_server.py").write_text(POOL_SERVER_PY)
    (task / "environment" / "term.py").write_text(TERM_PY)
    (task / "environment" / "wrap.py").write_text(WRAP_PY)
    (task / "environment" / "snoop.py").write_text(SNOOP_PY)
    for path, text in (
        (task / "solution" / "solve.sh", solve),
        (task / "tests" / "test.sh", TEST_SH),
    ):
        path.write_text(text)
        path.chmod(0o755)
    return task


async def _run(
    root: Path, name: str, task: Path | None = None
) -> tuple[dict | None, dict]:
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sdk import SDK

    # DOCKER_RUN_DEFAULTS: the trial runs the side-by-side path, as a run does.
    assert DockerSandbox.exec_channel is True
    assert DockerSandbox.concurrent_steps is True
    if task is None:
        solve = ON_TERMINAL if name == "on-terminal" else SOLVES[name]
        task = _task(root / "tasks", name, solve)
    jobs = root / "jobs" / name
    result = await SDK().run(
        task_path=task, agent="oracle", jobs_dir=jobs, environment="docker"
    )
    (path,) = jobs.glob("**/kill-before-grading.json")
    return result.rewards, json.loads(path.read_text())


@pytest.mark.usefixtures("require_docker")
@pytest.mark.parametrize("name", list(SOLVES))
async def test_a_kept_service_survives_its_shell_and_its_output(tmp_path, name):
    rewards, record = await _run(tmp_path, name)
    assert rewards == {"reward": 1.0}, record
    assert [row["port"] for row in record["kept_services"]] == [5000], record
    assert "kept_services_dropped" not in record, record
    assert "kept_services_lost" not in record, record
    assert "kept_services_unfrozen" not in record, record
    assert record["kept_services_released"] == 1, record


@pytest.mark.usefixtures("require_docker")
async def test_a_kept_service_writing_to_its_ended_terminal_is_recorded(tmp_path):
    """K2p-R3-A-2: a job typed into a terminal with no redirect is kept and
    outlives the kill, but its terminal is hung up when the terminal ends, so
    its writes fail; the record names it and its terminal."""
    _rewards, record = await _run(tmp_path, "on-terminal")
    assert [row["port"] for row in record["kept_services"]] == [5000], record
    (row,) = record["kept_services_on_terminal"]
    assert row["terminal"].startswith("/dev/pts/"), record


@pytest.mark.usefixtures("require_docker")
async def test_a_hostile_kept_service_reaches_nothing_during_grading(tmp_path):
    """The grading-integrity fixture
    (tests/fixtures/keep_services/keep-service-hostile): from the verifier's
    service check on, the kept service tries to read the tests, write under
    /logs (by path and through folder descriptors it held in flight on a
    socket, K2-R3-A-1), plant conftest.py files and a pytest plugin, and
    rewrite the answers through links (K2-R3-B-1); its judge scores 1 only if
    every attempt was made and none reached anything. Afterwards every agent
    process is killed before the log folders reopen."""
    import shutil

    fixture = Path(__file__).parent / "fixtures" / "keep_services"
    task = tmp_path / "keep-service-hostile"
    shutil.copytree(fixture / "keep-service-hostile", task)
    rewards, record = await _run(tmp_path, "hostile", task)
    (judge,) = (tmp_path / "jobs" / "hostile").glob("**/verifier/judge.txt")
    text = judge.read_text()
    assert rewards == {"reward": 1.0}, text
    assert text.rstrip().endswith("PASS"), text
    assert "write_logs_agent_via_descriptor_in_flight" in text, text
    assert record["kept_services_released"] == 1, record
    assert isinstance(record["kept_services_ended"], int), record


@pytest.mark.usefixtures("require_docker")
async def test_a_kept_service_cannot_read_the_verifier_env(tmp_path):
    """K2-R4-B-1: with a key in [verifier.env], a kept service that reads
    every command line and environment in the sandbox while test.sh runs
    finds no trace of it, and test.sh still gets the key."""
    solve = (
        "#!/bin/bash\nnohup python3 /opt/svc/snoop.py 5000 >/tmp/service.out 2>&1 "
        "</dev/null &\n" + WAIT_FOR_SERVICE
    )
    task = _task(tmp_path / "tasks", "snoop", solve)
    toml = (task / "task.toml").read_text()
    head = "[verifier]\ntimeout_sec = 120.0\n"
    assert head in toml
    (task / "task.toml").write_text(
        toml.replace(head, head + f'\n[verifier.env]\nJUDGE_API_KEY = "{FAKE_KEY}"\n')
    )
    (task / "tests" / "test.sh").write_text(SNOOP_TEST_SH)
    rewards, record = await _run(tmp_path, "snoop", task)
    (leak,) = (tmp_path / "jobs" / "snoop").glob("**/verifier/leak.txt")
    assert leak.read_text().strip() == "scan: clean", leak.read_text()
    assert rewards == {"reward": 1.0}, record
    assert record["kept_services_released"] == 1, record
