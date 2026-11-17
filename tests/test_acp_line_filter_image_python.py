"""The in-sandbox ACP line filter runs on the official Python images (FP patch 47).

``python:3.12-slim`` and the images built on it have ``python3`` in
``/usr/local/bin`` only. The filter probed ``/usr/bin/python3`` alone, so on
them the agent's stdout reached Daytona's PTY unfiltered, and a Read of large
images (several MB of base64 on one line) could stall the relay before the
host-side rewrite saw it. It now also runs ``/usr/local/bin/python3`` when
that file, every link on the way to the file it names, and every folder on
the way are root's, the folders and the file closed to group and others, and
it runs the file it checked, not the name.

Real Docker, the filter run as a non-root user as the agent's is. Skipped
without a Docker daemon.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import uuid

import pytest

from benchflow.sandbox.process._acp_lines import LIMIT, filter_command

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="docker required"
)

IMAGE = "python:3.12-slim"
LABEL = os.environ.get("BENCHFLOW_TEST_CONTAINER_LABEL", "PT=1")
PNG = base64.b64encode(os.urandom(3 * 1024 * 1024)).decode()
LINE = (
    json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "s1",
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "toolu_1",
                    "status": "completed",
                    "content": [
                        {
                            "type": "content",
                            "content": {
                                "type": "image",
                                "data": PNG,
                                "mimeType": "image/png",
                            },
                        }
                    ],
                },
            },
        }
    ).encode()
    + b"\n"
)


@pytest.fixture
def box():
    try:
        ready = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except Exception:
        ready = None
    if ready is None or ready.returncode != 0:
        pytest.skip("docker daemon not running")
    name = "bf-acp-filter-" + uuid.uuid4().hex[:8]
    run = subprocess.run(
        [
            "docker", "run", "-d", "--name", name, "--label", LABEL,
            "--network", "none", IMAGE, "sleep", "600",
        ],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start {IMAGE}: {run.stderr.strip()}")
    try:
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _agent_line(name: str) -> bytes:
    """What the agent's stdout ``cat`` of the line gives through the filter,
    run as uid 1000 in *name*."""
    done = subprocess.run(
        [
            "docker", "exec", "-i", "-u", "1000", name, "/bin/bash", "-c",
            filter_command("cat; exit 3"),
        ],
        input=LINE,
        capture_output=True,
        timeout=120,
    )  # fmt: skip
    assert done.returncode == 3, done.stderr
    return done.stdout


def test_a_python_slim_image_filters_the_agents_lines(box):
    out = _agent_line(box)
    assert len(out) <= LIMIT + 1
    update = json.loads(out)["params"]["update"]
    assert (update["toolCallId"], update["status"]) == ("toolu_1", "completed")


def test_a_python_others_may_replace_is_not_run(box):
    """With /usr/local/bin group-writable, the agent's line passes as it was
    (the host-side rewrite still applies)."""
    made = subprocess.run(
        ["docker", "exec", "-u", "0", box, "chmod", "g+w", "/usr/local/bin"],
        capture_output=True,
        text=True,
    )
    assert made.returncode == 0, made.stderr
    assert _agent_line(box) == LINE


def _dx(name: str, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", "-u", "0", name, "/bin/sh", "-c", command],
        capture_output=True,
        text=True,
    )


def _picked(name: str) -> subprocess.CompletedProcess:
    """The interpreter the filter's runner execs in *name* (``$bf_acp_py``,
    empty when the filter stays off) on stdout, and the agent's stderr, run
    as uid 1000."""
    return subprocess.run(
        [
            "docker", "exec", "-u", "1000", name, "/bin/bash", "-c",
            filter_command("printf '%s\\n' \"$bf_acp_py\""),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )  # fmt: skip


def _python(name: str) -> str:
    """The file ``/usr/local/bin/python3`` names in *name*."""
    return _dx(name, "readlink -f /usr/local/bin/python3").stdout.strip()


def test_the_filter_runs_the_file_it_checked(box):
    """a review L3: the runner execs the file python3 resolves to, which the
    check walked, not the name, so a link that changes after the check
    cannot pick what runs. A chain of root's links in root's folders, with a
    relative link and ``..``, still runs."""
    target = _python(box)
    assert target != "/usr/local/bin/python3"
    assert _picked(box).stdout == f"{target}\n"
    made = _dx(
        box,
        f"ln -s ../bin/{target.rsplit('/', 1)[1]} /usr/local/lib/py && "
        "ln -s /usr/local/lib/py /opt/py && ln -sf /opt/py /usr/local/bin/python3",
    )
    assert made.returncode == 0, made.stderr
    assert _picked(box).stdout == f"{target}\n"
    assert len(_agent_line(box)) <= LIMIT + 1


@pytest.mark.parametrize(
    "chain",
    [
        "mkdir /home/u && chown 1000 /home/u && ln -s {t} /home/u/py && "
        "chown -h 1000 /home/u/py && ln -sf /home/u/py /usr/local/bin/python3",
        "mkdir /home/u && chown 1000 /home/u && ln -s {t} /home/u/py && "
        "ln -sf /home/u/py /usr/local/bin/python3",
        "ln -s {t} /usr/local/bin/py-agent && chown -h 1000 /usr/local/bin/py-agent "
        "&& ln -sf py-agent /usr/local/bin/python3",
        "mkdir /home/u && chown 1000 /home/u && ln -s /home/u /opt/u && "
        "ln -sf /opt/u/../../{t} /usr/local/bin/python3",
    ],
    ids=[
        "agent-link-in-agent-folder",
        "root-link-in-agent-folder",
        "agent-link-in-root-folder",
        "dotdot-through-agent-folder",
    ],
)
def test_a_link_the_agent_could_change_refuses_the_interpreter(box, chain):
    """a review L3, case 4: python3 links to root's interpreter through a
    link the agent owns, or one in a folder the agent may write. The check
    read only the name and the file it resolves to, so it ran the name, and
    the agent's link chose the interpreter. Every link and folder on the way
    is checked now, and the filter stays off (the host-side rewrite still
    applies)."""
    made = _dx(box, chain.format(t=_python(box)))
    assert made.returncode == 0, made.stderr
    assert _picked(box).stdout == "\n"
    assert _agent_line(box) == LINE


def test_the_check_writes_nothing_to_the_agents_stderr(box):
    """a review N1: python:3.12-slim has no /usr/bin/python3, and the
    check's stat said so on the agent's stderr, which joins the ACP stream
    on Daytona's PTY. Without any python3, it said so twice."""
    done = _picked(box)
    assert (done.returncode, done.stderr) == (0, "")
    assert _dx(box, "rm /usr/local/bin/python3").returncode == 0
    done = _picked(box)
    assert (done.returncode, done.stdout, done.stderr) == (0, "\n", "")
