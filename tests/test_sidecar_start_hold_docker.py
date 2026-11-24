"""A task's compose services are held until their cloud metadata rule is in,
under an open network too.

On 0.9.3 through 0.10.0 the services were held in the inert bootstrap only under
a restricted policy: under ``public`` a sidecar's own start-up code (its
entrypoint, an early ``command:``) ran before BenchFlow's helper added its
rule, and reached the VM's metadata server. Now every service waits until its
rule is in and proven, then starts; a service that can't wait (no shell in its
image) refuses the trial with the reason; the task's granted opt-out holds
nothing, as before. The probe asks for the instance id only; the rows run only
where the host gets that answer (a GCE VM, meant to have no service account).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from tests.test_metadata_optin_docker import (
    LABEL,
    _docker_ready,
    _host_reaches_metadata,
    _reached,
    _refusal_text,
    _sdk_run,
    _task,
    build_probe_image,
)

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
    pytest.mark.skipif(not _docker_ready(), reason="no Docker daemon answers"),
    pytest.mark.skipif(
        not _host_reaches_metadata(),
        reason="the host gets no answer from a GCE metadata server",
    ),
]


@pytest.fixture(scope="module")
def probe_image(tmp_path_factory):
    yield from build_probe_image(tmp_path_factory)


# The sidecar's very first action: the probe, then a mark that it was
# released and ran, then sleep.
START_PY = r"""
import os, subprocess, time
probe = subprocess.run(["python3", "/opt/mdprobe/probe.py", "side-start"], capture_output=True, text=True)
with open("/shared/start.txt", "a") as out:
    out.write(probe.stdout)
    out.write(f"SIDE-RAN uid={os.getuid()}\n")
time.sleep(3600)
"""
TEST_SH = (
    "#!/bin/bash\n"
    "for _ in $(seq 1 60); do grep -q SIDE-RAN /shared/start.txt 2>/dev/null && break; sleep 1; done\n"
    "cp /shared/start.txt /logs/verifier/start.txt 2>/dev/null\n"
    "echo 1 > /logs/verifier/reward.txt\n"
)


def _start_task(root: Path, image: str, side: dict, *, key: str | None = None) -> Path:
    task = _task(root, image, key=key)
    env = task / "environment"
    (env / "start.py").write_text(START_PY)
    service = {
        "image": image,
        "volumes": ["shared:/shared", "./start.py:/start.py:ro"],
        **side,
    }
    (env / "docker-compose.yaml").write_text(
        json.dumps(
            {
                "services": {"main": {"volumes": ["shared:/shared"]}, "side": service},
                "volumes": {"shared": {}},
            }
        )
    )
    (task / "tests" / "test.sh").write_text(TEST_SH)
    (task / "tests" / "test.sh").chmod(0o755)
    return task


def _start_rows(trial: Path) -> tuple[list[dict], list[str]]:
    path = trial / "verifier" / "start.txt"
    lines = path.read_text(errors="replace").splitlines() if path.is_file() else []
    rows = [
        json.loads(x.split(" ", 1)[1]) for x in lines if x.startswith("BF-MDPROBE ")
    ]
    return rows, [x for x in lines if x.startswith("SIDE-RAN")]


RUN = ["python3", "/start.py"]
STARTS = {
    "entrypoint": {"entrypoint": RUN},
    "command": {"command": RUN},
}
USERS = {"root": {}, "service-user": {"user": "65534:65534"}}


@pytest.mark.parametrize("user", sorted(USERS))
@pytest.mark.parametrize("how", sorted(STARTS))
def test_a_sidecar_s_start_up_code_never_reaches_the_server(
    tmp_path, probe_image, how, user
):
    """Under ``public``, the sidecar's first action (its entrypoint, or its
    command) as root or as its own user gets no answer; it is released and
    runs once its rule is in, and service-network.json says it was covered."""
    task = _start_task(tmp_path, probe_image, {**STARTS[how], **USERS[user]})
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    rows, ran = _start_rows(trial)
    assert rows and ran, (_refusal_text(outcome), rows, ran)
    assert not _reached(rows), (
        f"the sidecar's start reached the server: {_reached(rows)}"
    )
    record = json.loads((trial / "service-network.json").read_text())
    side = [s for s in record.get("services") or () if s.get("service") == "side"]
    assert (
        side and side[0].get("metadata_rule") is True and side[0].get("proven") is True
    ), side


def test_a_sidecar_that_cannot_be_held_refuses_the_trial(tmp_path, probe_image):
    """A sidecar whose image has no shell can't wait for its rule: the trial
    fails at start under ``public`` too, and the sidecar's code never runs.
    (Where the sidecar is also held for its binds, the reason is Compose's or
    main's start error, not the named refusal: a known nit.)"""
    context = tmp_path / "bare-image"
    context.mkdir()
    (context / "start.py").write_text(START_PY)
    (context / "Dockerfile").write_text("FROM scratch\nCOPY start.py /start.py\n")
    tag = f"bf-test-mdhold-bare:{uuid.uuid4().hex[:10]}"
    built = subprocess.run(
        ["docker", "build", "--quiet", "--label", f"{LABEL}=image", "-t", tag, "."],
        cwd=context,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert built.returncode == 0, built.stderr[-300:]
    try:
        task = _start_task(
            tmp_path, probe_image, {"image": tag, "command": ["/start.py"]}
        )
        outcome, _trial = _sdk_run(
            tmp_path, task, "oracle", sandbox_user="agent", flag=False
        )
        said = _refusal_text(outcome)
        assert said, "the trial started with a sidecar that could not be held"
        ran = [] if _trial is None else _start_rows(_trial)[1]
        assert not ran, ran
    finally:
        subprocess.run(["docker", "rmi", "--force", tag], capture_output=True)


def test_the_granted_opt_out_holds_nothing(tmp_path, probe_image):
    """Key and flag: the block is lifted, so nothing is held for it and the
    sidecar's start-up code reaches the server, as before."""
    task = _start_task(tmp_path, probe_image, STARTS["command"], key="environment")
    outcome, trial = _sdk_run(tmp_path, task, "oracle", sandbox_user="agent", flag=True)
    assert trial is not None, _refusal_text(outcome)
    rows, ran = _start_rows(trial)
    assert ran and _reached(rows), (rows, ran)


SERVER_PY = r"""
import http.server, socketserver
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("0.0.0.0", 8080), http.server.SimpleHTTPRequestHandler) as httpd:
    httpd.serve_forever()
"""
HEALTH = [
    "CMD",
    "python3",
    "-c",
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/', timeout=1)",
]


def test_a_healthy_sidecar_main_depends_on_still_starts_and_scores(
    tmp_path, probe_image
):
    """Start-up timing under the hold: a sidecar serving HTTP with a short
    healthcheck (1 s interval, 2 s timeout, 3 retries), and main depending on
    it with ``condition: service_healthy``. The trial starts, test.sh reaches
    the sidecar, and it scores 1."""
    task = _task(tmp_path, probe_image)
    env = task / "environment"
    (env / "server.py").write_text(SERVER_PY)
    (env / "docker-compose.yaml").write_text(
        json.dumps(
            {
                "services": {
                    "main": {"depends_on": {"web": {"condition": "service_healthy"}}},
                    "web": {
                        "image": probe_image,
                        "volumes": ["./server.py:/server.py:ro"],
                        "command": ["python3", "/server.py"],
                        "healthcheck": {
                            "test": HEALTH,
                            "interval": "1s",
                            "timeout": "2s",
                            "retries": 3,
                        },
                    },
                }
            }
        )
    )
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\n"
        "python3 -c \"import urllib.request; urllib.request.urlopen('http://web:8080/', timeout=5)\" "
        "&& echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt\n"
    )
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    result = json.loads((trial / "result.json").read_text())
    assert not result.get("error"), result.get("error")
    assert (result.get("rewards") or {}).get("reward") == 1.0, result.get("rewards")
