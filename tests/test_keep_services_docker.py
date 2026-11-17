"""Kept services on real Docker: shells with job control and unredirected output.

Deterministic tier (``tests/integration/README.md``): real sandboxes, no
model (the ``oracle`` agent runs each task's solution as the agent user,
``[agent] user = "agent"``). Runs where Docker answers and skips with a reason
otherwise. Each task starts the same small HTTP service on port 5000, names
the port in ``x-keep-services``, and its test reads the service; a service
that answers scores 1.

What these assert is the default mode's grading, ``audit`` ((g2), F2b
19:37Z). The faithful default (the lead's 09:58Z ruling) scores what the
task's own tests report, and strict is the boundary. S2 (12:36Z): audit runs
no kill before grading. test.sh runs as written beside the agent's processes,
the kept service among them, and the rollout ends them all after grading
(``planes.quiesce_agent``). So no ``kill-before-grading.json`` is written, and
no freeze or release happens. Strict would harden, but a task that keeps
services cannot have a separate verifier (DF-4), and strict forces one, so
strict refuses these tasks before the agent's turn. The kept-services
hardening (the freeze, ``_setup._release_kept_services`` and the folders it
closes) therefore runs in no mode today. decides, after the cut,
whether kept services get it in strict or have no hardened mode for good.

The launches below were the subjects of the strict kill and release:

- K2p-R1-A-1: the service is a job of a job-control shell (``set -m``, as a
  terminal tool's interactive shell runs commands), in a process group of its
  own, and the shell is still alive at the end of the turn; the shell ignores
  SIGTERM, as an interactive bash does, in some tasks.
- K2p-R2-A-1: the service is a plain job typed into an interactive bash on a
  terminal (``term.py`` holds the pty, as a tmux server does).
- K2p-R2-B-1: the service keeps a pool of spawned workers that do not hold
  its port and replaces any that die.
- K2p-R1-A-2: solve.sh starts the service with no redirect of its own, so it
  inherits the oracle's output file.

In audit each of them must answer test.sh and be gone once grading is over.
"""

from __future__ import annotations

import json
import os
import stat
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
    """The rewards and the kill record of one oracle trial in the default
    mode. Kept for tests/test_keep_services_receipt_docker.py alone, left as
    it was while F2b rules on what an agent process can do to the start
    receipt in audit (19:51Z). It fails in audit, which writes no kill
    record (S2, 12:36Z)."""
    from benchflow.sdk import SDK

    if task is None:
        solve = ON_TERMINAL if name == "on-terminal" else SOLVES[name]
        task = _task(root / "tasks", name, solve)
    jobs = root / "jobs" / name
    result = await SDK().run(
        task_path=task, agent="oracle", jobs_dir=jobs, environment="docker"
    )
    (path,) = jobs.glob("**/kill-before-grading.json")
    return result.rewards, json.loads(path.read_text())


AGENT_USER = "agent"
# Run in the sandbox as root: every process whose real, effective or saved uid
# is the agent user's, from /proc (the image has no ps), with its state and
# command line.
AGENT_PROCESSES = r"""
import json, os, pwd, sys

uid = pwd.getpwnam(sys.argv[1]).pw_uid
rows = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        with open(f"/proc/{pid}/status") as f:
            status = dict(line.rstrip("\n").split(":\t", 1) for line in f if ":\t" in line)
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        continue
    if uid not in [int(v) for v in status.get("Uid", "").split()[:3]]:
        continue
    rows.append({"pid": int(pid), "state": status.get("State", ""), "cmd": cmd})
print("AGENT-PROCESSES " + json.dumps(rows))
"""


async def _agent_processes(env) -> list[dict] | str:
    import shlex

    from benchflow.sandbox._exec_output import exec_raw

    try:
        result = await exec_raw(
            env.exec,
            f"/usr/local/bin/python3 -c {shlex.quote(AGENT_PROCESSES)} {AGENT_USER}",
            user="root",
            timeout_sec=60,
        )
        line = next(
            line
            for line in (result.stdout or "").splitlines()
            if line.startswith("AGENT-PROCESSES ")
        )
    except Exception as exc:
        return f"not listed: {exc!r}"
    return json.loads(line.removeprefix("AGENT-PROCESSES "))


def _watch_grading(monkeypatch) -> dict:
    """Wrap the rollout's grading, test-only. The wrappers record the agent
    user's processes as test.sh's command returns (``_await_verifier``) and
    again right after the rollout ends them after grading (the first
    ``planes.quiesce_agent`` call after that). They also count every step of
    the kept-services hardening: a kill that keeps ports (the freeze),
    ``_release_kept_services`` and ``_end_kept_services``."""
    from benchflow.rollout import _setup
    from benchflow.rollout_planes import DefaultRolloutPlanes

    seen: dict = {
        "graded": False,
        "at_exit": None,
        "after": None,
        "kept_ports": [],
        "released": 0,
        "ended": 0,
    }
    real_await = _setup._await_verifier
    real_quiesce = DefaultRolloutPlanes.quiesce_agent
    real_release = _setup._release_kept_services
    real_end = _setup._end_kept_services

    async def await_verifier(env, verifier, *args, **kwargs):
        try:
            return await real_await(env, verifier, *args, **kwargs)
        finally:
            seen["graded"] = True
            seen["at_exit"] = await _agent_processes(env)

    async def quiesce_agent(self, env, sandbox_user, *, keep_ports=()):
        if keep_ports:
            seen["kept_ports"].append(tuple(keep_ports))
        outcome = await real_quiesce(self, env, sandbox_user, keep_ports=keep_ports)
        if seen["graded"] and not keep_ports and seen["after"] is None:
            seen["after"] = await _agent_processes(env)
        return outcome

    async def release(*args, **kwargs):
        seen["released"] += 1
        return await real_release(*args, **kwargs)

    async def end(*args, **kwargs):
        seen["ended"] += 1
        return await real_end(*args, **kwargs)

    monkeypatch.setattr(_setup, "_await_verifier", await_verifier)
    monkeypatch.setattr(DefaultRolloutPlanes, "quiesce_agent", quiesce_agent)
    monkeypatch.setattr(_setup, "_release_kept_services", release)
    monkeypatch.setattr(_setup, "_end_kept_services", end)
    return seen


async def _graded_run(
    root: Path,
    name: str,
    monkeypatch,
    task: Path | None = None,
    *,
    integrity: str | None = None,
) -> tuple[dict | None, dict, Path]:
    """One oracle trial with its grading watched (:func:`_watch_grading`), in
    the default mode (audit) or in *integrity*. Returns the rewards, what the
    watch saw, and the trial folder. No mode writes a record of a kill before
    grading for a kept-services task: off and audit run no such kill (S2,
    12:36Z), and strict refuses the task first (DF-4)."""
    from benchflow.sandbox.docker import DockerSandbox

    # DOCKER_RUN_DEFAULTS: the trial runs the side-by-side path, as a run does.
    assert DockerSandbox.exec_channel is True
    assert DockerSandbox.concurrent_steps is True
    if task is None:
        solve = ON_TERMINAL if name == "on-terminal" else SOLVES[name]
        task = _task(root / "tasks", name, solve)
    jobs = root / "jobs" / name
    seen = _watch_grading(monkeypatch)
    if integrity is None:
        from benchflow.sdk import SDK

        result = await SDK().run(
            task_path=task, agent="oracle", jobs_dir=jobs, environment="docker"
        )
    else:
        import benchflow as bf

        result = await bf.run(
            bf.RolloutConfig(
                task_path=task,
                agent="oracle",
                jobs_dir=jobs,
                environment="docker",
                sandbox_user=AGENT_USER,
                integrity=integrity,
            )
        )
    assert not list(jobs.glob("**/kill-before-grading.json")), sorted(jobs.glob("**/*"))
    (path,) = jobs.glob("*/*/result.json")
    trial = path.parent
    if integrity is None:
        # The default mode is audit (the lead's 10:01Z ruling).
        config = json.loads((trial / "config.json").read_text())
        assert config["integrity"] == "audit", config
    return result.rewards, seen, trial


def _graded_beside_then_ended(seen: dict, service: str = "server.py 5000") -> None:
    """Audit's grading of a kept-services task (S2, 12:36Z; the faithful
    default, 09:58Z). Nothing froze or released the service. It was still
    running when test.sh's command returned, and no process of the agent user
    was left once the rollout ended them after grading."""
    # The freeze, the release and the end of the kept services are strict's
    # hardening, and strict refuses a task that keeps services (DF-4).
    assert seen["kept_ports"] == [], seen
    assert seen["released"] == 0 and seen["ended"] == 0, seen
    at_exit = seen["at_exit"]
    assert isinstance(at_exit, list), seen
    assert any(service in row["cmd"] for row in at_exit), at_exit
    assert seen["after"] == [], seen


def _ran_task(root: Path, name: str) -> Path:
    """The no-redirect task whose solve.sh first leaves /logs/agent/solve-ran."""
    solve = "#!/bin/bash\necho ran > /logs/agent/solve-ran\n" + SOLVES[
        "no-redirect"
    ].removeprefix("#!/bin/bash\n")
    return _task(root / "tasks", name, solve)


# DF-4's refusal, as the trial's error records it.
STRICT_REFUSAL = (
    "ValueError: integrity=strict runs the verifier in a separate verifier "
    "sandbox, which this task cannot use here:\n- metadata.x-keep-services: the "
    "tests talk to the services the agent leaves running on port 5000 (fixture "
    "service) (from the task), and a separate verifier sandbox has none of them"
)


@pytest.mark.usefixtures("require_docker")
@pytest.mark.parametrize("name", list(SOLVES))
async def test_a_kept_service_survives_its_shell_and_its_output(
    tmp_path, monkeypatch, name
):
    """In audit, the default, there is no kill before grading (S2, 12:36Z).
    However solve.sh launched it, the service is still running while test.sh
    reads it, and the trial scores what test.sh writes. After grading the
    rollout ends every process of the agent user, including a pool's
    respawned workers, a terminal and its shell, and a launcher."""
    rewards, seen, _trial_dir = await _graded_run(tmp_path, name, monkeypatch)
    assert rewards == {"reward": 1.0}, seen
    _graded_beside_then_ended(seen)
    # No kill before grading in audit (S2), so no kill record: its rows of the
    # services kept (kept_services), dropped, lost or left unfrozen, and the
    # release's count (kept_services_released), are strict's (DF-4).


@pytest.mark.usefixtures("require_docker")
async def test_a_kept_service_on_its_terminal_scores_and_ends_after_grading(
    tmp_path, monkeypatch
):
    """K2p-R3-A-2's job, typed into a terminal with no redirect. In audit
    nothing ends its terminal before grading (S2, 12:36Z), so the service
    answers test.sh and the trial scores. The terminal (term.py) is still
    open then, and it ends with the service after grading."""
    rewards, seen, _trial_dir = await _graded_run(tmp_path, "on-terminal", monkeypatch)
    assert rewards == {"reward": 1.0}, seen
    _graded_beside_then_ended(seen)
    assert any("term.py" in row["cmd"] for row in seen["at_exit"]), seen
    # No release in audit (S2): nothing records the service under
    # kept_services_on_terminal, because only strict's kill ends its terminal.


# The grading-integrity fixture's attempts (its tests/judge.py DENIED).
HOSTILE_ATTEMPTS = (
    "read_tests_list",
    "read_tests_test_sh",
    "read_tests_grading_check",
    "write_logs_verifier",
    "write_logs_verifier_via_descriptor_in_flight",
    "write_logs_verifier_test_stdout_via_descriptor_in_flight",
    "write_logs_agent_via_descriptor_in_flight",
    "write_logs_artifacts_via_descriptor_in_flight",
    "write_logs_verifier_reward",
    "plant_conftest_tests",
    "plant_conftest_app",
    "plant_conftest_root",
    "plant_conftest_logs_verifier",
    "plant_plugin_module_site_packages",
    "plant_plugin_entry_point_site_packages",
    "plant_plugin_module_app",
    "plant_plugin_entry_point_app",
)
_SITE = "/usr/local/lib/python3.12/site-packages"
# The files the fixture can plant (its judge's PLANTED).
HOSTILE_PLANTABLE = (
    "/tests/conftest.py",
    "/app/conftest.py",
    "/conftest.py",
    "/logs/verifier/conftest.py",
    "/logs/verifier/kept-marker.txt",
    "/logs/verifier/kept-fd-marker.txt",
    "/logs/agent/kept-fd-marker.txt",
    "/logs/artifacts/kept-fd-marker.txt",
    "/logs/verifier/reward-kept.txt",
    "/logs/verifier/planted-conftest-loaded",
    "/logs/verifier/planted-plugin-loaded",
    f"{_SITE}/kept_marker_plugin.py",
    f"{_SITE}/kept_marker_plugin-0.dist-info",
    "/app/kept_marker_plugin.py",
    "/app/kept_marker_plugin-0.dist-info",
)
# The residual: what the kept service reaches in audit, where it runs beside
# test.sh and test.sh runs as written, with no conftest sweep and no plugin
# guard (S2 12:36Z, CIR's L14; the faithful default, 09:58Z). It reads the
# uploaded tests. It writes in /logs/agent and /logs/artifacts through the
# folder descriptors it held. It plants in the workspace, /app, where
# test.sh's pytest loads its plugin. If audit ever starts hardening, these
# lists change and this test fails, so the residual has to be decided again
# .
HOSTILE_REACHED = (
    "read_tests_list",
    "read_tests_test_sh",
    "read_tests_grading_check",
    "write_logs_agent_via_descriptor_in_flight",
    "write_logs_artifacts_via_descriptor_in_flight",
    "plant_conftest_app",
    "plant_plugin_module_app",
    "plant_plugin_entry_point_app",
)
HOSTILE_PLANTED = (
    "/app/conftest.py",
    "/logs/agent/kept-fd-marker.txt",
    "/logs/artifacts/kept-fd-marker.txt",
    "/logs/verifier/planted-plugin-loaded",
    "/app/kept_marker_plugin.py",
    "/app/kept_marker_plugin-0.dist-info",
)


def _hostile_task(tmp_path) -> Path:
    """A copy of the grading-integrity fixture
    (tests/fixtures/keep_services/keep-service-hostile)."""
    import shutil

    fixture = Path(__file__).parent / "fixtures" / "keep_services"
    task = tmp_path / "keep-service-hostile"
    shutil.copytree(fixture / "keep-service-hostile", task)
    return task


async def _hostile(tmp_path, monkeypatch, name: str, task: Path):
    """Run the hostile task; returns the rewards, the judge's lines and what
    the watch saw."""
    rewards, seen, trial = await _graded_run(tmp_path, name, monkeypatch, task)
    lines = (trial / "verifier" / "judge.txt").read_text().splitlines()
    return rewards, lines, seen


def _assert_the_audit_residual(rewards, lines: list[str], seen: dict) -> None:
    text = "\n".join(lines)
    reached = sorted(
        n
        for n in HOSTILE_ATTEMPTS
        if any(line.startswith(f"FAIL {n}: ") for line in lines)
    )
    denied = sorted(
        n
        for n in HOSTILE_ATTEMPTS
        if any(line.startswith(f"ok   {n}: ") for line in lines)
    )
    # Each attempt was made and judged.
    assert sorted(reached + denied) == sorted(HOSTILE_ATTEMPTS), text
    assert reached == sorted(HOSTILE_REACHED), text
    # Writes under /logs/verifier fail in every mode: the folder is root's
    # 0755 while test.sh runs (the verifier's close), and
    # test-stdout.txt is root's there.
    assert [n for n in denied if "logs_verifier" in n] == sorted(
        n for n in HOSTILE_ATTEMPTS if "logs_verifier" in n
    ), text
    present = sorted(p for p in HOSTILE_PLANTABLE if f"FAIL {p} absent" in lines)
    assert present == sorted(HOSTILE_PLANTED), text
    # The rewrites through its links reach the files the verifier reads; the
    # answer it could not hard-link (/logs is another filesystem) still reads 42.
    assert "ok   /logs/artifacts/answer.txt reads '42'" in lines, text
    assert "FAIL /logs/artifacts/linked.txt reads '99'" in lines, text
    assert "FAIL /logs/agent/notes.txt reads '99'" in lines, text
    # The planted plugin made test.sh's own check fail, and test.sh scored
    # its judge's FAIL: the trial scores what test.sh wrote.
    assert "FAIL pytest exited 1" in lines, text
    assert lines[-1] == "FAIL", text
    assert rewards == {"reward": 0.0}, text
    # Every agent process, the hostile service included, ended after grading.
    _graded_beside_then_ended(seen, "/opt/svc/server.py 5000")


@pytest.mark.usefixtures("require_docker")
async def test_a_hostile_kept_service_reaches_only_the_audit_residual(
    tmp_path, monkeypatch
):
    """The grading-integrity fixture (tests/fixtures/keep_services/
    keep-service-hostile). From the verifier's service check on, the kept
    service tries to read the tests, write under /logs (by path, and through
    the folder descriptors it held in flight on a socket, K2-R3-A-1), plant
    conftest.py files and a pytest plugin, and rewrite the answers through
    links (K2-R3-B-1). In audit it runs beside test.sh with no kill before
    grading (S2 12:36Z, L14). Every write under /logs/verifier fails, it
    reaches exactly HOSTILE_REACHED, and every agent process has ended
    after grading."""
    rewards, lines, seen = await _hostile(
        tmp_path, monkeypatch, "hostile", _hostile_task(tmp_path)
    )
    _assert_the_audit_residual(rewards, lines, seen)
    # "Reaches nothing", with the judge's PASS and a reward of 1, was the
    # strict release's result. No release in audit (S2): no
    # kept_services_released or kept_services_ended count either.


@pytest.mark.usefixtures("require_docker")
async def test_a_kept_service_cannot_read_the_verifier_env(tmp_path, monkeypatch):
    """K2-R4-B-1: with a key in [verifier.env], a kept service that reads
    every command line and environment in the sandbox while test.sh runs
    finds no trace of it, and test.sh still gets the key. In audit the
    agent's processes are alive through grading, so the verifier passes its
    env through a file only root can read."""
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
    rewards, seen, trial = await _graded_run(tmp_path, "snoop", monkeypatch, task)
    leak = (trial / "verifier" / "leak.txt").read_text()
    assert leak.strip() == "scan: clean", leak
    assert rewards == {"reward": 1.0}, seen
    _graded_beside_then_ended(seen, "snoop.py 5000")
    # No release in audit (S2): no kept_services_released count.


@pytest.mark.usefixtures("require_docker")
async def test_strict_refuses_a_kept_services_task_before_the_agent_s_turn(
    tmp_path, monkeypatch
):
    """DF-4: strict runs the verifier in a separate verifier sandbox, which has
    none of the services the task keeps. So strict refuses a kept-services
    task before the agent's turn: no reward, the reason in the trial's
    error, and solve.sh never runs. Until decides, no mode runs the
    kept-services hardening."""
    task = _ran_task(tmp_path, "strict")
    rewards, seen, trial = await _graded_run(
        tmp_path, "strict", monkeypatch, task, integrity="strict"
    )
    assert rewards is None
    result = json.loads((trial / "result.json").read_text())
    assert (result.get("rewards") or None) is None, result
    error = result.get("error") or ""
    assert error.startswith(STRICT_REFUSAL), error
    assert "so every trial would score 0" in error, error
    assert not list(trial.glob("**/solve-ran")), sorted(trial.glob("**/*"))
    assert not list(trial.glob("verifier/*")), sorted(trial.glob("verifier/*"))
    # Never graded, nothing frozen, released or ended.
    assert seen["graded"] is False and seen["after"] is None, seen
    assert seen["kept_ports"] == [] and seen["released"] == 0, seen
    assert seen["ended"] == 0, seen


@pytest.mark.usefixtures("require_docker")
@pytest.mark.parametrize("integrity", ["off", "audit", "strict"])
async def test_the_kept_services_hardening_is_unreachable_in_every_mode(
    tmp_path, monkeypatch, integrity
):
    """The lead, 19:40Z: the kept-services hardening path is unreachable
    today. That path is the freeze in the kill before grading,
    ``_setup._release_kept_services`` before test.sh, and the tests closed
    to the service. It runs in none of the integrity modes the CLI accepts,
    so nobody should read it as a protection that is on. off and audit grade
    beside the agent's processes (S2, 12:36Z; the faithful default, 09:58Z)
    and score what test.sh writes. strict refuses the task before the
    agent's turn (DF-4). decides whether the path comes back."""
    from benchflow.integrity.constants import INTEGRITY_MODES

    # Every mode --integrity accepts (normalize_integrity_mode).
    assert set(INTEGRITY_MODES) == {"off", "audit", "strict"}
    task = _ran_task(tmp_path, integrity)
    rewards, seen, trial = await _graded_run(
        tmp_path, integrity, monkeypatch, task, integrity=integrity
    )
    assert seen["released"] == 0, seen
    assert seen["kept_ports"] == [] and seen["ended"] == 0, seen
    ran = list(trial.glob("agent/solve-ran"))
    if integrity == "strict":
        assert rewards is None
        error = json.loads((trial / "result.json").read_text()).get("error") or ""
        assert error.startswith(STRICT_REFUSAL), error
        assert not ran, sorted(trial.glob("**/*"))
        assert seen["graded"] is False, seen
    else:
        assert rewards == {"reward": 1.0}, seen
        assert ran, sorted(trial.glob("**/*"))
        _graded_beside_then_ended(seen)


@pytest.mark.usefixtures("require_docker")
async def test_the_host_still_writes_its_verifier_folder_while_services_are_kept(
    tmp_path, monkeypatch
):
    """on Docker /logs/verifier is the host's own verifier/ folder of
    the trial, and BenchFlow on the host writes test-stdout.txt and its
    records there. In audit the kept service runs beside test.sh (S2,
    12:36Z). The verifier's own close makes the folder root's 0755 while
    test.sh runs, and gives it back to the host user afterwards. So the
    verifier does not crash, test-stdout.txt is there, and the trial
    scores."""
    rewards, seen, trial = await _graded_run(tmp_path, "no-redirect", monkeypatch)
    result = json.loads((trial / "result.json").read_text())
    assert result.get("verifier_error") is None, result.get("verifier_error")
    assert rewards == {"reward": 1.0}, seen
    assert (trial / "verifier" / "test-stdout.txt").is_file()
    assert (trial / "verifier").stat().st_uid == os.getuid()
    _graded_beside_then_ended(seen)
    # No release and no end of kept services in audit (S2): the kill record's
    # kept_services_released and kept_services_ended are strict's.


@pytest.mark.usefixtures("require_docker")
async def test_an_agent_user_with_the_host_users_uid_is_closed_out_of_the_folder(
    tmp_path, monkeypatch
):
    """With K2-R1-B-1: the image's own agent user has the host user's
    uid, so the host's folder is the agent's too. Such a trial is refused
    before it starts; this runs it past the refusal. In audit there
    is no release (S2, 12:36Z). The verifier's own close makes the folder
    root's 0755 while test.sh runs, which closes the agent's uid out of it.
    After test.sh the folder goes back to the host user, and the trial
    scores."""
    uid = _host_uid_or_skip()
    _without_the_host_uid_refusal(monkeypatch)
    task = _task(tmp_path / "tasks", "same-uid", SOLVES["no-redirect"])
    _same_uid(task, uid)
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho \"folder: $(stat -c '%u %a' /logs/verifier)\"\n"
        + GRADED_TEST_SH.removeprefix("#!/bin/bash\n")
    )
    rewards, seen, trial = await _graded_run(tmp_path, "same-uid", monkeypatch, task)
    assert rewards == {"reward": 1.0}, seen
    text = (trial / "verifier" / "test-stdout.txt").read_text()
    assert text == "folder: 0 755\ngraded: kept\n", text
    assert (trial / "verifier").stat().st_uid == uid
    _graded_beside_then_ended(seen)
    # No release in audit (S2): no kept_services_closed_folders row (the
    # release's close of the folder) and no kept_services_ended count.


# --- The host's verifier/ folder at the close before test.sh (KS3, R174-M1) -----

# test.sh for the tests below: what it prints is what test-stdout.txt must hold.
GRADED_TEST_SH = """#!/bin/bash
out=$(python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:5000/', timeout=5).read().decode().strip())" 2>/dev/null)
echo "graded: $out"
if [ "$out" = kept ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""
# A test.sh that writes only reward.json, whose reward BenchFlow aggregates
# and writes back into verifier/ during grading.
AGGREGATE_TEST_SH = """#!/bin/bash
out=$(python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:5000/', timeout=5).read().decode().strip())" 2>/dev/null)
v=0; [ "$out" = kept ] && v=1
echo "graded: $out"
printf '{"service_up": %s, "answer_ok": %s, "aggregate": {"method": "mean"}}\\n' "$v" "$v" > /logs/verifier/reward.json
"""
# Run in the sandbox as root, right after the verifier closes its outputs
# folder for test.sh (in audit there is no release, S2). It opens
# /logs/verifier, then, as the agent user, tries through that descriptor (as
# a kept service holding one in flight would) to create, rename and delete an
# entry there and to open each file for writing. It also lists every
# descriptor the agent user's processes hold under /logs/verifier, with its
# open flags. The victim is root's, made for the rename and the delete and
# removed after.
AS_AGENT_PROBE = r"""
import json, os, pwd, sys

user = pwd.getpwnam(sys.argv[1])
folder = os.open("/logs/verifier", os.O_RDONLY | os.O_DIRECTORY)
victim = ".ks3-victim"
fd = os.open(victim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644, dir_fd=folder)
os.close(fd)
read_end, write_end = os.pipe()
child = os.fork()
if child == 0:
    os.close(read_end)
    os.setgroups([])
    os.setgid(user.pw_gid)
    os.setuid(user.pw_uid)
    out = {"uid": os.getuid(), "access_w": os.access(f"/proc/self/fd/{folder}", os.W_OK)}

    def tried(name, action):
        try:
            action()
        except OSError as exc:
            out[name] = f"denied: {exc.strerror}"
        else:
            out[name] = "allowed"

    tried("create", lambda: os.close(os.open(".ks3-new", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644, dir_fd=folder)))
    tried("rename", lambda: os.rename(victim, ".ks3-moved", src_dir_fd=folder, dst_dir_fd=folder))
    tried("delete", lambda: os.unlink(victim, dir_fd=folder))
    for name in sorted(os.listdir(folder)):
        if name != victim:
            tried(f"write {name}", lambda name=name: os.close(os.open(name, os.O_WRONLY | os.O_APPEND, dir_fd=folder)))
    writing, inspected, unreadable = [], 0, 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/status") as f:
                uid = next(int(line.split()[2]) for line in f if line.startswith("Uid:"))
        except (OSError, StopIteration):
            continue
        if uid != user.pw_uid:
            continue
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            unreadable += 1
            continue
        inspected += 1
        for n in fds:
            try:
                target = os.readlink(f"/proc/{pid}/fd/{n}")
                with open(f"/proc/{pid}/fdinfo/{n}") as f:
                    flags = next(int(line.split()[1], 8) for line in f if line.startswith("flags:"))
            except (OSError, StopIteration):
                continue
            if target.startswith("/logs/verifier") and flags & 3:
                writing.append({"pid": int(pid), "fd": int(n), "target": target, "flags": oct(flags)})
    out.update(processes_inspected=inspected, processes_unreadable=unreadable, writing=writing)
    os.write(write_end, json.dumps(out).encode())
    os._exit(0)
os.close(write_end)
data = b""
while chunk := os.read(read_end, 65536):
    data += chunk
os.waitpid(child, 0)
for name in (victim, ".ks3-moved", ".ks3-new"):
    try:
        os.unlink(name, dir_fd=folder)
    except FileNotFoundError:
        pass
print("KS3-PROBE " + data.decode())
"""


def _host_uid_or_skip() -> int:
    uid = os.getuid()
    if not 1000 <= uid < 60000:
        pytest.skip(f"the image's agent user cannot take host uid {uid}")
    return uid


def _same_uid(task: Path, uid: int) -> None:
    """The image's own agent user takes the host user's uid."""
    dockerfile = task / "environment" / "Dockerfile"
    lines = [
        line
        for line in dockerfile.read_text().splitlines(keepends=True)
        if f"-u {uid} " not in line
    ]
    dockerfile.write_text(
        "".join(lines) + f"RUN useradd -m -u {uid} -s /bin/bash agent\n"
    )


def _without_the_host_uid_refusal(monkeypatch) -> None:
    """Let a trial whose agent user has the host user's uid reach grading,
    as it will once lifts the refusal."""
    from benchflow.task import keep_services

    monkeypatch.setattr(
        keep_services, "refuse_host_uid", lambda *a, **k: None, raising=False
    )


def _entries(folder: Path) -> dict[str, os.stat_result]:
    out = {}
    for top, dirs, files in os.walk(folder):
        for name in dirs + files:
            path = Path(top) / name
            out[str(path.relative_to(folder))] = path.lstat()
    return out


def _watch_the_close(monkeypatch, seen: list[dict], *, hold: str | None = None):
    """Wrap each close of the outputs folder before a test command
    (``Verifier._close_output_dir``). In audit that close protects the folder
    from the agent's processes, which are alive through grading; strict's
    release, which used to do it, never runs (S2). The wrapper records the
    host's verifier/ folder before and after the close, and
    AS_AGENT_PROBE's answer. With ``hold``, it first opens that file in
    verifier/ for writing, creating it if needed (BenchFlow's own uid, the
    agent user's in the same-uid case), and keeps the descriptor in
    ``seen``."""
    import shlex

    from benchflow.sandbox._exec_output import exec_raw
    from benchflow.task.verifier_core import Verifier

    real = Verifier._close_output_dir

    async def close(self, service, *, mounted):
        folder = Path(self._rollout_paths.rollout_dir) / "verifier"
        assert folder.is_dir(), folder
        row: dict = {"before": _entries(folder)}
        if hold is not None:
            row["held"] = os.open(
                folder / hold, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644
            )
        await real(self, service, mounted=mounted)
        row["after"] = _entries(folder)
        row["folder"] = folder.stat()
        result = await exec_raw(
            self._sandbox.exec,
            f"/usr/local/bin/python3 -c {shlex.quote(AS_AGENT_PROBE)} "
            f"{shlex.quote(AGENT_USER)}",
            user="root",
            timeout_sec=60,
        )
        line = next(
            line
            for line in (result.stdout or "").splitlines()
            if line.startswith("KS3-PROBE ")
        )
        row["probe"] = json.loads(line.removeprefix("KS3-PROBE "))
        seen.append(row)

    monkeypatch.setattr(Verifier, "_close_output_dir", close)


@pytest.mark.usefixtures("require_docker")
@pytest.mark.parametrize("same_uid", [True, False], ids=["same-uid", "other-uid"])
async def test_the_close_before_test_sh_leaves_verifier_root_s_alone(
    tmp_path, monkeypatch, same_uid
):
    """R174-M1 in audit (S2, 12:36Z: no release, the service runs beside
    test.sh). What is in the host's verifier/ folder before the test command
    can be owned by the host user's uid, which can be the agent user's. After
    the verifier's close, every entry is root's and not writable by group or
    others. A file another uid owned is gone, and root's test-stdout.txt is
    new. The folder is root's 0755 whoever owned it. The agent user can
    create, rename, delete and open for writing nothing there, and none of
    its processes, the kept service among them, holds a file there open for
    writing. After test.sh the folder is the host user's again."""
    host = os.getuid()
    task = _task(tmp_path / "tasks", "close", SOLVES["no-redirect"])
    (task / "tests" / "test.sh").write_text(GRADED_TEST_SH)
    if same_uid:
        _same_uid(task, _host_uid_or_skip())
        _without_the_host_uid_refusal(monkeypatch)
    closes: list[dict] = []
    _watch_the_close(monkeypatch, closes)
    rewards, seen, trial = await _graded_run(tmp_path, "close", monkeypatch, task)
    (row,) = closes
    before, after, folder, probe = (
        row["before"],
        row["after"],
        row["folder"],
        row["probe"],
    )
    # Root makes test-stdout.txt for test.sh in the closed folder (the
    # release's root copy of the host's file is strict's, S2).
    assert "test-stdout.txt" in after, after
    for name, st in after.items():
        assert st.st_uid == 0, (name, st.st_uid)
        assert not st.st_mode & 0o022, (name, oct(st.st_mode))
    for name, st in before.items():
        if stat.S_ISREG(st.st_mode) and st.st_uid != 0 and name in after:
            assert after[name].st_ino != st.st_ino, name
    assert stat.S_IMODE(folder.st_mode) == 0o755, oct(folder.st_mode)
    # Root's for test.sh in both cases. The release left it the host user's
    # when that uid was not the agent's; no release in audit (S2).
    assert folder.st_uid == 0, folder.st_uid
    assert (probe["uid"] == host) is same_uid, probe
    assert probe["access_w"] is False, probe
    denied = {k: v for k, v in probe.items() if k in ("create", "rename", "delete")}
    denied.update({k: v for k, v in probe.items() if k.startswith("write ")})
    assert "write test-stdout.txt" in denied, probe
    assert all(v.startswith("denied") for v in denied.values()), probe
    assert probe["writing"] == [], probe
    # The kept service runs while the folder is closed (S2).
    assert probe["processes_inspected"] >= 1, probe
    assert rewards == {"reward": 1.0}, seen
    assert (trial / "verifier" / "test-stdout.txt").read_text() == "graded: kept\n"
    assert (trial / "verifier").stat().st_uid == host
    _graded_beside_then_ended(seen)


@pytest.mark.usefixtures("require_docker")
async def test_a_descriptor_opened_before_the_close_writes_no_verifier_output(
    tmp_path, monkeypatch
):
    """R174-M1 in audit (S2, 12:36Z: no release). Before the verifier's close,
    BenchFlow's own process opens test-stdout.txt for writing as the agent
    user's uid, which it has here. It writes through that descriptor once
    test.sh is done. The write reaches only the file the close removed (it
    was not root's), so the test-stdout.txt the verifier reads holds
    test.sh's output alone."""
    from benchflow.task.verifier_core import Verifier

    task = _task(tmp_path / "tasks", "held", SOLVES["no-redirect"])
    (task / "tests" / "test.sh").write_text(GRADED_TEST_SH)
    _same_uid(task, _host_uid_or_skip())
    _without_the_host_uid_refusal(monkeypatch)
    closes: list[dict] = []
    _watch_the_close(monkeypatch, closes, hold="test-stdout.txt")
    real_after = Verifier._after_test_command
    wrote: list[int] = []

    async def after_test(self, *args, **kwargs):
        wrote.append(os.write(closes[0]["held"], b"FORGED after test.sh\n"))
        return await real_after(self, *args, **kwargs)

    monkeypatch.setattr(Verifier, "_after_test_command", after_test)
    try:
        rewards, seen, trial = await _graded_run(tmp_path, "held", monkeypatch, task)
    finally:
        for row in closes:
            os.close(row["held"])
    assert wrote and wrote[0] > 0, wrote
    text = (trial / "verifier" / "test-stdout.txt").read_text()
    assert text == "graded: kept\n", text
    assert rewards == {"reward": 1.0}, seen
    _graded_beside_then_ended(seen)


@pytest.mark.usefixtures("require_docker")
async def test_a_hostile_kept_service_with_the_host_users_uid_reaches_only_the_audit_residual(
    tmp_path, monkeypatch
):
    """The grading-integrity fixture with the agent user given the host
    user's uid (K2-R1-B-1), past the refusal, in audit (S2 12:36Z, L14). Its
    /logs/verifier descriptor in flight cannot open test-stdout.txt for
    writing, because root made that file for test.sh. Every other write
    under /logs/verifier fails too, and it reaches exactly what it reaches
    on a uid of its own (HOSTILE_REACHED)."""
    uid = _host_uid_or_skip()
    task = _hostile_task(tmp_path)
    _same_uid(task, uid)
    _without_the_host_uid_refusal(monkeypatch)
    rewards, lines, seen = await _hostile(
        tmp_path, monkeypatch, "hostile-same-uid", task
    )
    assert any(f"agent uid {uid}" in line for line in lines), lines
    assert any(
        line.startswith(
            "ok   write_logs_verifier_test_stdout_via_descriptor_in_flight: "
        )
        for line in lines
    ), lines
    _assert_the_audit_residual(rewards, lines, seen)
    # No release in audit (S2): no kept_services_closed_folders row.


@pytest.mark.usefixtures("require_docker")
async def test_an_agent_user_with_the_host_users_uid_is_refused_before_it_starts(
    tmp_path,
):
    """with the image's agent user on the host user's uid, the
    release would make verifier/ root's for grading and BenchFlow could
    write nothing there until the kept services end (a reward.json it
    aggregates would be lost). The trial is refused right after the agent
    user's uid is measured, before solve.sh runs: no score, no kill, and
    ``bench explain`` names the uid and the remedies."""
    from benchflow import failures
    from benchflow.sdk import SDK

    _host_uid_or_skip()
    solve = "#!/bin/bash\necho ran > /logs/agent/solve-ran\n" + SOLVES[
        "no-redirect"
    ].removeprefix("#!/bin/bash\n")
    task = _task(tmp_path / "tasks", "refused", solve)
    (task / "tests" / "test.sh").write_text(AGGREGATE_TEST_SH)
    _same_uid(task, os.getuid())
    jobs = tmp_path / "jobs" / "refused"
    result = await SDK().run(
        task_path=task, agent="oracle", jobs_dir=jobs, environment="docker"
    )
    assert result.rewards is None
    (path,) = jobs.glob("*/*/result.json")
    data = json.loads(path.read_text())
    error = data.get("error") or ""
    assert error.startswith("KeptServiceFreezerUnavailable: "), error
    assert f"uid {os.getuid()}" in error, error
    assert not list(jobs.glob("**/solve-ran")), list(jobs.glob("**/*"))
    assert not list(jobs.glob("**/kill-before-grading.json"))
    cause = failures.cause_of(data)
    assert cause is not None and cause.key == "kept_service_freezer:host_uid", cause


@pytest.mark.usefixtures("require_docker")
async def test_a_reward_json_aggregate_scores_while_services_are_kept(
    tmp_path, monkeypatch
):
    """The default case, with the agent user on a uid of its own. In audit
    BenchFlow writes the aggregated reward.json back into the host's
    verifier/ folder, which the host user has again once test.sh has
    exited."""
    task = _task(tmp_path / "tasks", "aggregate", SOLVES["no-redirect"])
    (task / "tests" / "test.sh").write_text(AGGREGATE_TEST_SH)
    rewards, seen, trial = await _graded_run(tmp_path, "aggregate", monkeypatch, task)
    assert rewards is not None and rewards.get("reward") == 1.0, seen
    # The kill record named the user (record["user"]). No kill in audit (S2),
    # so the trial's config names it.
    assert json.loads((trial / "config.json").read_text())["sandbox_user"] == "agent"
    written = json.loads((trial / "verifier" / "reward.json").read_text())
    assert written.get("reward") == 1.0, written
    assert (trial / "verifier" / "test-stdout.txt").read_text() == "graded: kept\n"
    _graded_beside_then_ended(seen)


# The first attempt spoils its start receipt and waits, as a test command
# whose start BenchFlow cannot confirm; the retry runs the test. (A removed
# receipt is never read as missing on Docker: the probe's empty output comes
# back as None, and it waits out the budget instead.)
RETRY_TEST_SH = """#!/bin/bash
if [ ! -e /tmp/ks3-first-attempt ]; then
    touch /tmp/ks3-first-attempt
    # Where the receipt is: a kept-service trial's is in its attempt's
    # folder.
    for r in /run/runtime/"$BENCHFLOW_VERIFIER_ATTEMPT" /run/runtime/*/"$BENCHFLOW_VERIFIER_ATTEMPT"; do
        [ -e "$r" ] && echo spoiled > "$r"
    done
    echo "first attempt"
    sleep 300
fi
echo "second attempt"
""" + GRADED_TEST_SH.removeprefix("#!/bin/bash\n")


async def _retried(tmp_path, monkeypatch, name: str, *, same_uid: bool):
    from benchflow.rollout import _setup

    monkeypatch.setattr(_setup, "_VERIFIER_START_GRACE_SEC", 5.0)
    task = _task(tmp_path / "tasks", name, SOLVES["no-redirect"])
    (task / "tests" / "test.sh").write_text(RETRY_TEST_SH)
    if same_uid:
        _same_uid(task, _host_uid_or_skip())
        _without_the_host_uid_refusal(monkeypatch)
    closes: list[dict] = []
    _watch_the_close(monkeypatch, closes)
    rewards, seen, trial = await _graded_run(tmp_path, name, monkeypatch, task)
    return rewards, seen, closes, trial


@pytest.mark.usefixtures("require_docker")
async def test_a_verifier_retried_in_place_rewrites_the_host_s_folder(
    tmp_path, monkeypatch
):
    """A verifier whose start receipt is gone is retried in place. In audit
    (S2, 12:36Z: no release) each attempt's close leaves only root's entries,
    which the agent user cannot write. BenchFlow writes test-stdout.txt
    again for the retry, and the trial scores the second attempt."""
    rewards, seen, closes, trial = await _retried(
        tmp_path, monkeypatch, "retry", same_uid=False
    )
    assert len(closes) == 2, closes
    for row in closes:
        assert all(st.st_uid == 0 for st in row["after"].values()), row["after"]
        assert row["probe"]["access_w"] is False, row["probe"]
        assert row["probe"]["writing"] == [], row["probe"]
    text = (trial / "verifier" / "test-stdout.txt").read_text()
    assert text == "second attempt\ngraded: kept\n", text
    assert rewards == {"reward": 1.0}, seen
    _graded_beside_then_ended(seen)


@pytest.mark.usefixtures("require_docker")
async def test_a_same_uid_verifier_retried_in_place_rewrites_the_folder(
    tmp_path, monkeypatch
):
    """The same retry with the agent user on the host user's uid, past the
    refusal. This was xfail(strict=True): the first release made
    verifier/ root's, and BenchFlow could not write test-stdout.txt there
    again for the retry. Audit has no release (S2, 12:36Z), so the retry
    rewrites the folder and scores. The reason applies again only if
    strict ever runs kept services."""
    rewards, seen, closes, trial = await _retried(
        tmp_path, monkeypatch, "retry-same-uid", same_uid=True
    )
    assert len(closes) == 2, closes
    text = (trial / "verifier" / "test-stdout.txt").read_text()
    assert text == "second attempt\ngraded: kept\n", text
    assert rewards == {"reward": 1.0}, seen
    _graded_beside_then_ended(seen)
