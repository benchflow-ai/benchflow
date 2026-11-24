"""A verifier that inherits the image's non-root user writes its own reward, on real Docker.

The image's USER is uid 2000 and the task's ``[verifier] user`` is empty, so test.sh runs as the container's own
user. /logs/verifier is closed to every other user from sandbox setup and again right before the turn (root's or
the host user's, 0755); the close before test.sh gives it to the verifier's user, with or without services the agent
keeps for grading. test.sh scores 1 only when it runs as uid 2000 (and, with kept services, reaches the agent's
service) and its reward lands.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()

IMAGE = """FROM python:3.12-slim
RUN useradd -m -s /bin/bash agent && useradd -u 2000 -m grader && mkdir -p /app && chmod 777 /app
WORKDIR /app
USER 2000:2000
"""

TASK_TOML = """version = "1.0"
{metadata}
[verifier]
user = ""
timeout_sec = 120.0

[agent]
timeout_sec = 60.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 512
"""

# The agent's turn: a service on 8080 that answers 42, kept for grading where the task keeps it.
SERVER = (
    "import socket; s=socket.socket(); s.bind(('127.0.0.1',8080)); s.listen(); "
    "open('/app/ready','w').close(); "
    "exec(\"while True:\\n c,a=s.accept(); c.sendall(b'42'); c.close()\")"
)
AGENT = f"""#!/bin/bash
setsid /usr/local/bin/python -I -c {shlex.quote(SERVER)} </dev/null >/dev/null 2>&1 &
while [ ! -e /app/ready ]; do sleep .05; done
"""

TEST_SH = """#!/bin/bash
uid=$(id -u)
probe="import socket; s=socket.create_connection(('127.0.0.1',8080),1); assert s.recv(2)==b'42'"
if [ "$uid" = 2000 ] && {{ [ {keep} = 0 ] || /usr/local/bin/python -I -c "$probe"; }}; then r=1; else r=0; fi
echo "uid=$uid"
echo $r > /logs/verifier/reward.txt
"""


def _task(root: Path, name: str, *, keep: bool) -> None:
    task = root / name
    for folder in ("environment", "solution", "tests"):
        (task / folder).mkdir(parents=True)
    metadata = "\n[metadata]\nx-keep-services = [{port = 8080}]\n" if keep else ""
    (task / "task.toml").write_text(TASK_TOML.format(metadata=metadata))
    (task / "instruction.md").write_text("Serve 42 on port 8080.\n")
    (task / "environment" / "Dockerfile").write_text(IMAGE)
    for path, text in (
        (task / "solution" / "solve.sh", "#!/bin/bash\n"),
        (task / "tests" / "test.sh", TEST_SH.format(keep=int(keep))),
    ):
        path.write_text(text)
        path.chmod(0o755)


@pytest.mark.parametrize("keep", [False, True], ids=["plain", "kept-services"])
def test_a_verifier_inheriting_the_image_user_writes_its_reward(tmp_path, keep):
    """Scored 1: test.sh ran as uid 2000 and wrote reward.txt in /logs/verifier."""
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else "Docker only")
    name = "inherit-kept" if keep else "inherit-plain"
    _task(tmp_path / "tasks", name, keep=keep)
    agent = tmp_path / "agent.sh"
    agent.write_text(AGENT)
    agent.chmod(0o755)
    jobs = tmp_path / "jobs"
    done = subprocess.run(
        [
            h.bench_executable(),
            "eval",
            "run",
            "--tasks-dir",
            str(tmp_path / "tasks"),
            "--agent",
            f"script:{agent}",
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
    stdout = results[0].parent / "verifier" / "test-stdout.txt"
    shown = stdout.read_text() if stdout.exists() else ""
    assert trial.get("error") is None and trial.get("verifier_error") is None, (
        trial.get("error") or trial.get("verifier_error"),
        shown,
    )
    assert (trial.get("rewards") or {}).get("reward") == 1.0, (shown, results[0].parent)
