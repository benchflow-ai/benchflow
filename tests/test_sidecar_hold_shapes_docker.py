"""Compose shapes that start under Docker and must start under the hold too
(measured on the hold's first version), and a sidecar that Docker
restarts.

- A strict healthcheck (1 s interval, 1 retry, no start period) on a sidecar
  main depends on with ``condition: service_healthy``: Docker's first probe
  comes one interval after the start, so it passes there; the hold's check
  counts the same way from the sidecar's release.
- A sidecar with ``restart: on-failure`` that exits 1 on its first start:
  BenchFlow starts it again (Docker never does), covers it and releases it,
  and it comes up.
- A sidecar whose image has no shell (distroless) can't wait in the hold:
  refused before start with that reason, never a misleading one about main.
- A sidecar with a restart policy that exits later in the trial stays down:
  the trial records it, and is scored as test.sh says and flagged.

The probes ask for the instance id only; the rows run only where the host gets
that answer (a GCE VM, meant to have no service account).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_metadata_optin_docker import (
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

POLICIES = ["public", "no-network"]


@pytest.fixture(scope="module")
def probe_image(tmp_path_factory):
    yield from build_probe_image(tmp_path_factory)


def _compose(task: Path, services: dict, volumes: dict | None = None) -> None:
    document: dict = {"services": services}
    if volumes:
        document["volumes"] = volumes
    (task / "environment" / "docker-compose.yaml").write_text(json.dumps(document))


def _script(task: Path, name: str, body: str) -> None:
    path = task / ("tests" if name == "test.sh" else "solution") / name
    path.write_text(body)
    path.chmod(0o755)


def _result(trial: Path) -> dict:
    return json.loads((trial / "result.json").read_text())


def _rows(lines: list[str]) -> list[dict]:
    return [
        json.loads(x.split(" ", 1)[1]) for x in lines if x.startswith("BF-MDPROBE ")
    ]


def _lines(trial: Path, name: str) -> list[str]:
    path = trial / "verifier" / name
    return path.read_text(errors="replace").splitlines() if path.is_file() else []


# -------------------------------------------------- a fast healthcheck

# No bind and no volume shared with main: on 0.10.0 the sidecar
# started natively under ``public``. It reports over HTTP.
SERVER_PY = r"""
import http.server, socketserver, time
time.sleep(0.3)  # a server that takes a moment to listen
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
PASS = "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
REACH_WEB = (
    "#!/bin/bash\n"
    "python3 -c \"import urllib.request; urllib.request.urlopen('http://web:8080/', timeout=5)\" "
    "&& echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt\n"
)


@pytest.mark.parametrize("bound", [False, True], ids=["plain", "with-a-bind"])
@pytest.mark.parametrize("policy", POLICIES)
def test_a_strict_healthcheck_that_passes_under_docker_passes_under_the_hold(
    tmp_path, probe_image, policy, bound
):
    """A fast healthcheck: interval 1 s, 1 retry, no start period; main depends on the
    sidecar with ``condition: service_healthy``. Starts and scores 1. (With
    a bind the sidecar was held on 0.10.0 too, and failed there.)"""
    task = _task(tmp_path, probe_image, policy=policy)
    web = {
        "image": probe_image,
        "command": ["python3", "-c", SERVER_PY],
        "healthcheck": {"test": HEALTH, "interval": "1s", "retries": 1},
    }
    if bound:
        (task / "environment" / "server.py").write_text(SERVER_PY)
        web.update(
            volumes=["./server.py:/server.py:ro"], command=["python3", "/server.py"]
        )
    _compose(
        task,
        {"main": {"depends_on": {"web": {"condition": "service_healthy"}}}, "web": web},
    )
    # Under no-network the verifier's own network keeps test.sh from the
    # sidecar; the start is what this row is about.
    _script(task, "test.sh", REACH_WEB if policy == "public" else PASS)
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    result = _result(trial)
    assert not result.get("error"), result.get("error")
    assert (result.get("rewards") or {}).get("reward") == 1.0, result.get("rewards")


# -------------------------------------------------------- a restart at start-up

# Every start probes first and keeps the probe's row in its own file system
# (kept across Docker's restarts of the container). ONCE: the first start
# exits 1; the next serves the rows. CRASH: the first start serves them and
# exits 1 when asked (GET /crash); the next serves them again.
_SIDE_HEAD = r"""
import http.server, os, socketserver, subprocess, sys
again = os.path.exists("/tmp/started")
probe = subprocess.run(["python3", "/opt/mdprobe/probe.py", "side-restart" if again else "side-start"], capture_output=True, text=True)
with open("/tmp/rows.txt", "a") as out:
    out.write(probe.stdout)
    out.write("SIDE-RAN-AGAIN\n" if again else "SIDE-RAN\n")
open("/tmp/started", "a").close()
"""
_SIDE_SERVE = r"""
class Rows(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/crash":
            os._exit(1)
        body = open("/tmp/rows.txt", "rb").read()
        self.send_response(200); self.end_headers(); self.wfile.write(body)
socketserver.TCPServer.allow_reuse_address = True
socketserver.TCPServer(("0.0.0.0", 8080), Rows).serve_forever()
"""
ONCE_PY = _SIDE_HEAD + "if not again:\n    sys.exit(1)\n" + _SIDE_SERVE
CRASH_PY = _SIDE_HEAD + _SIDE_SERVE
SIDE_PROBE_PY = _SIDE_HEAD + _SIDE_SERVE
FETCH = (
    "#!/bin/bash\n"
    "python3 - <<'EOF'\n"
    "import time, urllib.request\n"
    "rows = b''\n"
    "for _ in range(__TRIES__):\n"
    "    try:\n"
    "        rows = urllib.request.urlopen('http://side:8080/', timeout=3).read(); break\n"
    "    except Exception:\n"
    "        time.sleep(1)\n"
    "open('/logs/verifier/start.txt', 'wb').write(rows)\n"
    "ok = b'__MARK__' in rows or __ALWAYS__\n"
    "open('/logs/verifier/reward.txt', 'w').write('1' if ok else '0')\n"
    "EOF\n"
)


def _fetch(mark: str, *, always: bool = False, tries: int = 20) -> str:
    return (
        FETCH.replace("__MARK__", mark)
        .replace("__ALWAYS__", str(always))
        .replace("__TRIES__", str(tries))
    )


def _restart_task(
    root: Path,
    image: str,
    policy: str,
    script: str,
    *,
    restart: str | None = "on-failure",
    shared: bool = False,
) -> Path:
    """With *shared*, main and the sidecar both mount the volume ``turn`` at
    /shared (the image's, mode 1777)."""
    task = _task(root, image, policy=policy)
    main: dict = {}
    side = {"image": image, "command": ["python3", "-c", script]}
    if restart is not None:
        side["restart"] = restart
    if shared:
        main["volumes"] = side["volumes"] = ["turn:/shared"]
    _compose(task, {"main": main, "side": side}, {"turn": {}} if shared else None)
    return task


@pytest.mark.parametrize("policy", POLICIES)
def test_a_sidecar_that_exits_once_at_start_is_covered_again_and_comes_up(
    tmp_path, probe_image, policy
):
    """BenchFlow starts the sidecar again; it is covered before its code
    runs again, and the trial scores 1. Neither start's probe gets an
    answer. (Under no-network the verifier's network keeps
    test.sh from the sidecar: the start is the row there.)"""
    task = _restart_task(tmp_path, probe_image, policy, ONCE_PY)
    public = policy == "public"
    _script(task, "test.sh", _fetch("SIDE-RAN-AGAIN") if public else PASS)
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    result = _result(trial)
    lines = _lines(trial, "start.txt")
    assert not result.get("error"), (result.get("error"), lines)
    assert (result.get("rewards") or {}).get("reward") == 1.0, (result, lines)
    if public:
        rows = _rows(lines)
        assert {"side-start", "side-restart"} <= {r.get("phase") for r in rows}, lines
        assert not _reached(rows), _reached(rows)


# ------------------------------------------------- a restart later in the trial

CRASH_SOLVE = (
    "#!/bin/bash\n"
    "sleep 12\n"
    "python3 -c \"import urllib.request; urllib.request.urlopen('http://side:8080/crash', timeout=5)\" "
    "2>/dev/null\n"
    "sleep 8\n"
)
# test.sh asks the sidecar to crash, gives Docker a moment to restart it,
# then reads its rows; it passes only on the restarted start's mark, unless
# __ALWAYS__.
CRASH_IN_TEST = (
    "#!/bin/bash\n"
    "python3 - <<'EOF'\n"
    "import time, urllib.request\n"
    "try:\n"
    "    urllib.request.urlopen('http://side:8080/crash', timeout=3)\n"
    "except Exception:\n"
    "    pass\n"
    "time.sleep(4)\n"
    "rows = b''\n"
    "for _ in range(6):\n"
    "    try:\n"
    "        rows = urllib.request.urlopen('http://side:8080/', timeout=3).read(); break\n"
    "    except Exception:\n"
    "        time.sleep(1)\n"
    "open('/logs/verifier/start.txt', 'wb').write(rows)\n"
    "ok = b'SIDE-RAN-AGAIN' in rows or __ALWAYS__\n"
    "open('/logs/verifier/reward.txt', 'w').write('1' if ok else '0')\n"
    "EOF\n"
)


# The sidecar kills itself mid-turn, a cause inside its own container: its
# own thread exits once the turn has left a file in a volume both share
# (never on a timer, which a slow start-up can outrun).
SELF_CRASH_PY = (
    _SIDE_HEAD
    + r"""
import threading, time
def exit_in_the_turn():
    while not os.path.exists("/shared/turn-began"):
        time.sleep(0.2)
    os._exit(1)
if not again:
    threading.Thread(target=exit_in_the_turn, daemon=True).start()
"""
    + _SIDE_SERVE
)
SELF_CRASH_SOLVE = "#!/bin/bash\ntouch /shared/turn-began\nsleep 20\n"
CRASHES = {
    "by-a-request-from-main": (CRASH_PY, CRASH_SOLVE, False),
    "by-itself": (SELF_CRASH_PY, SELF_CRASH_SOLVE, True),
}


@pytest.mark.parametrize("how", sorted(CRASHES))
def test_a_sidecar_with_a_restart_policy_that_exits_in_the_agent_s_turn_stays_down(
    tmp_path, probe_image, how
):
    """The sidecar (``restart: on-failure``) exits during the oracle's run
    (crashed by a request from main under the run's policy, or by itself).
    After start-up an exit is final: nothing starts it again. A cause inside
    the task's containers: the trial is scored as test.sh says (it needs the
    sidecar back, so 0), flagged for a read, and result.json names the
    service, its phase and its exit status."""
    script, solve, shared = CRASHES[how]
    task = _restart_task(tmp_path, probe_image, "public", script, shared=shared)
    _script(task, "solve.sh", solve)
    _script(task, "test.sh", _fetch("SIDE-RAN-AGAIN", tries=8))
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    result = _result(trial)
    _exited(result, "the agent's turn", 1)
    assert (result.get("rewards") or {}).get("reward") == 0.0, (
        result.get("rewards"),
        result.get("verifier_error"),
    )


@pytest.mark.parametrize(
    "needs_it", [True, False], ids=["test-needs-it", "test-passes"]
)
def test_a_sidecar_with_a_restart_policy_that_exits_in_test_sh_stays_down(
    tmp_path, probe_image, needs_it
):
    """The sidecar exits while test.sh runs and stays down: nothing of it
    reaches the server. The trial is scored as test.sh says (0 when it needs
    the sidecar back, 1 when it passes anyway), flagged for a read, the
    service named."""
    task = _restart_task(tmp_path, probe_image, "public", CRASH_PY)
    _script(task, "test.sh", CRASH_IN_TEST.replace("__ALWAYS__", str(not needs_it)))
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    rows = _rows(_lines(trial, "start.txt"))
    assert not _reached(rows), _reached(rows)
    result = _result(trial)
    _exited(result, "test.sh", 1)
    want = 0.0 if needs_it else 1.0
    assert (result.get("rewards") or {}).get("reward") == want, (
        result.get("rewards"),
        result.get("verifier_error"),
    )


def test_an_exit_status_is_kept_when_docker_s_backlog_has_lost_it(
    tmp_path, probe_image, monkeypatch
):
    """Docker keeps a short backlog of events: on a busy daemon, asking for a
    held container's last exit after the fact can come back empty. The
    record still has the exit status."""
    from unittest.mock import AsyncMock

    from benchflow.sandbox import _inert_bootstrap

    monkeypatch.setattr(_inert_bootstrap, "_last_exit", AsyncMock(return_value=None))
    task = _restart_task(tmp_path, probe_image, "public", CRASH_PY)
    _script(task, "test.sh", CRASH_IN_TEST.replace("__ALWAYS__", "True"))
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    _exited(_result(trial), "test.sh", 1)


def test_an_outside_cause_leaves_the_trial_unscored_with_its_reason(
    tmp_path, probe_image, monkeypatch
):
    """BenchFlow's own process failing (the controller can't cover a
    restarted sidecar's new namespace) is an outside cause: the trial is not
    scored, and its error says why."""
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.service_cover import ServiceCoverError

    async def fails(self, name, cid):
        raise ServiceCoverError(
            f"sandbox startup refused: could not cover {name!r} again"
        )

    monkeypatch.setattr(DockerSandbox, "_cover_restarted_service", fails)
    task = _restart_task(tmp_path, probe_image, "public", ONCE_PY)
    _script(task, "test.sh", PASS)
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    said = _refusal_text(outcome)
    result = _result(trial) if trial is not None else {}
    assert not (result.get("rewards") or {}), result.get("rewards")
    assert "could not cover 'side' again" in (said + str(result.get("error"))), said


# --------------------------------- nothing of the image runs before the rules

# The image's own /bin/sleep (or /bin/sh) replaced by a
# wrapper that asks for the instance id first. The old hold looped on the
# image's sleep under its shell, before the service's rule.
WRAPPERS = {
    "sleep": (
        "RUN mv /bin/sleep /bin/sleep.real && printf '%s\\n' '#!/bin/sh.real' "
        "'python3 /opt/mdprobe/probe.py hold-sleep >> /tmp/rows.txt 2>/dev/null' "
        "'exec /bin/sleep.real \"$@\"' > /bin/sleep && chmod 755 /bin/sleep "
        "&& cp /bin/sh /bin/sh.real\n"
    ),
    "sh": (
        "RUN cp /bin/sh /bin/sh.real && rm /bin/sh && printf '%s\\n' '#!/bin/sh.real' "
        "'python3 /opt/mdprobe/probe.py hold-sh >> /tmp/rows.txt 2>/dev/null' "
        "'exec /bin/sh.real \"$@\"' > /bin/sh && chmod 755 /bin/sh\n"
    ),
}


@pytest.fixture(scope="module")
def wrapped_images(probe_image, tmp_path_factory):
    import uuid

    from tests.test_metadata_optin_docker import LABEL

    made = {}
    for name, step in WRAPPERS.items():
        context = tmp_path_factory.mktemp(f"wrap-{name}")
        (context / "Dockerfile").write_text(f"FROM {probe_image}\n{step}")
        tag = f"bf-test-mdhold-wrap-{name}:{uuid.uuid4().hex[:10]}"
        built = subprocess.run(
            ["docker", "build", "--quiet", "--label", f"{LABEL}=image", "-t", tag, "."],
            cwd=context,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert built.returncode == 0, built.stderr[-400:]
        made[name] = tag
    yield made
    for tag in made.values():
        subprocess.run(["docker", "rmi", "--force", tag], capture_output=True)


@pytest.mark.parametrize("where", ["sidecar", "main"])
@pytest.mark.parametrize("wrapped", sorted(WRAPPERS))
def test_nothing_of_the_image_runs_in_a_held_service_before_its_rule(
    tmp_path, probe_image, wrapped_images, wrapped, where
):
    """A held service's image with its own /bin/sleep or
    /bin/sh replaced by a wrapper that asks for the instance id. Under
    ``public``, nothing of it may run before the service's rule (BenchFlow's
    waiter holds it), so no row of the wrapper's gets an answer; the service
    then runs and the trial scores."""
    image = wrapped_images[wrapped]
    task = _task(tmp_path, image if where == "main" else probe_image)
    side_image = image if where == "sidecar" else probe_image
    _compose(
        task,
        {
            "main": {},
            "side": {"image": side_image, "command": ["python3", "-c", SIDE_PROBE_PY]},
        },
    )
    carry_main = (
        "cp /tmp/rows.txt /logs/verifier/main-rows.txt 2>/dev/null\n"
        if where == "main"
        else ""
    )
    _script(
        task, "test.sh", _fetch("SIDE-RAN").replace("EOF\n", "EOF\n" + carry_main, 1)
    )
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    rows = _rows(_lines(trial, "start.txt")) + _rows(_lines(trial, "main-rows.txt"))
    held = [r for r in rows if str(r.get("phase", "")).startswith("hold-")]
    assert not _reached(held), _reached(held)
    result = _result(trial)
    assert (result.get("rewards") or {}).get("reward") == 1.0, (
        result.get("rewards"),
        result.get("error"),
        result.get("verifier_error"),
    )


# ---------------------------------------------------------- an image with no shell

DISTROLESS = "gcr.io/distroless/python3-debian12"


@pytest.fixture(scope="module")
def distroless():
    pulled = subprocess.run(
        ["docker", "pull", "--quiet", DISTROLESS],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if pulled.returncode:
        pytest.skip(f"could not pull {DISTROLESS}: {pulled.stderr[-200:]}")
    return DISTROLESS


@pytest.mark.parametrize("bound", [False, True], ids=["plain", "with-a-bind"])
def test_a_shell_less_sidecar_is_held_by_benchflow_s_waiter_and_released(
    tmp_path, probe_image, distroless, bound
):
    """A distroless sidecar serving HTTP has no
    shell to wait in, and needs none: BenchFlow's waiter holds it until its
    rule is in, then starts its own process. The trial starts and scores."""
    task = _task(tmp_path, probe_image)
    web = {"image": distroless, "command": ["-m", "http.server", "8080"]}
    if bound:
        (task / "environment" / "index.html").write_text("ok\n")
        web["volumes"] = ["./index.html:/index.html:ro"]
    _compose(task, {"main": {}, "web": web})
    _script(task, "test.sh", REACH_WEB)
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    result = _result(trial)
    assert (result.get("rewards") or {}).get("reward") == 1.0, (
        result.get("rewards"),
        result.get("error"),
    )


# The values a distroless sidecar's own program gets for two names a held
# container clears (PYTHONPATH, one of the loader's start-up names, and PS4):
# a space, an "=" and a ";" in them, so a byte lost or added shows.
SETENV_VALUES = {"PYTHONPATH": "/opt/b6-setenv a=b;c", "PS4": "+b6 setenv= "}
ENV_SERVE_PY = r"""
import http.server, os, socketserver
class Env(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = repr([os.environ.get("PYTHONPATH"), os.environ.get("PS4")]).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(body)
socketserver.TCPServer.allow_reuse_address = True
socketserver.TCPServer(("0.0.0.0", 8080), Env).serve_forever()
"""
ENV_CHECK = (
    "#!/bin/bash\n"
    "python3 - <<'EOF'\n"
    "import time, urllib.request\n"
    "body = b''\n"
    "for _ in range(20):\n"
    "    try:\n"
    "        body = urllib.request.urlopen('http://web:8080/', timeout=3).read(); break\n"
    "    except Exception:\n"
    "        time.sleep(1)\n"
    "want = repr([__PYTHONPATH__, __PS4__]).encode()\n"
    "open('/logs/verifier/reward.txt', 'w').write('1' if body == want else '0')\n"
    "EOF\n"
)


def test_a_distroless_sidecar_gets_its_cleared_variables_back_byte_for_byte(
    tmp_path, probe_image, distroless
):
    """A distroless sidecar (no env, no shell) whose task sets PYTHONPATH and
    PS4: the held container's environment, every root exec's, has them
    cleared, and bfx gives them back to the service's own program at its
    release (--setenv), byte for byte. Through the image's env it did not
    start at all. Nothing of the values reaches the trial's records."""
    task = _task(tmp_path, probe_image)
    web = {
        "image": distroless,
        "command": ["-c", ENV_SERVE_PY],
        "environment": dict(SETENV_VALUES),
    }
    _compose(task, {"main": {}, "web": web})
    check = ENV_CHECK.replace(
        "__PYTHONPATH__", repr(SETENV_VALUES["PYTHONPATH"])
    ).replace("__PS4__", repr(SETENV_VALUES["PS4"]))
    _script(task, "test.sh", check)
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    result = _result(trial)
    assert (result.get("rewards") or {}).get("reward") == 1.0, (
        result.get("rewards"),
        result.get("error"),
        result.get("verifier_error"),
    )
    # The waiter's argv carries the values; no record of the job keeps them
    # (the trial's files, and the job's run card, summary and config).
    marker = SETENV_VALUES["PYTHONPATH"].encode()
    job = trial.parent
    recorded = [
        str(path.relative_to(job))
        for path in job.rglob("*")
        if path.is_file() and marker in path.read_bytes()
    ]
    assert not recorded, recorded


# ------------------------------------------------- the run's flag alone


def test_with_the_run_s_flag_alone_a_sidecar_keeps_its_rule(tmp_path, probe_image):
    """``--allow-cloud-metadata`` without the task's key lifts the agent's
    own rule; under the hold the task's other services keep theirs (0.10.0
    removed it with the flag alone): the sidecar's first probe gets no
    answer. Only the task's granted opt-out lifts it
    (``test_sidecar_start_hold_docker.py``)."""
    task = _task(tmp_path, probe_image)
    _compose(
        task,
        {
            "main": {},
            "side": {"image": probe_image, "command": ["python3", "-c", SIDE_PROBE_PY]},
        },
    )
    _script(task, "test.sh", _fetch("SIDE-RAN"))
    outcome, trial = _sdk_run(tmp_path, task, "oracle", sandbox_user="agent", flag=True)
    assert trial is not None, _refusal_text(outcome)
    rows = _rows(_lines(trial, "start.txt"))
    assert rows, _lines(trial, "start.txt")
    assert not _reached(rows), _reached(rows)


# ------------------------------------------------------- both architectures

ARM_IMAGE = "python:3.12-slim"


def _binfmt_aarch64() -> bool:
    return Path("/proc/sys/fs/binfmt_misc/qemu-aarch64").exists()


@pytest.mark.skipif(not _binfmt_aarch64(), reason="no QEMU binfmt for aarch64")
def test_an_arm64_sidecar_is_held_by_the_host_s_waiter_and_released(
    tmp_path, probe_image
):
    """A sidecar of the other architecture (``platform: linux/arm64`` on an
    x86_64 host, under QEMU): the host's bfx holds it natively, then starts
    the arm64 process, and the trial scores."""
    pulled = subprocess.run(
        ["docker", "pull", "--quiet", "--platform", "linux/arm64", ARM_IMAGE],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if pulled.returncode:
        pytest.skip(f"could not pull the arm64 {ARM_IMAGE}: {pulled.stderr[-200:]}")
    task = _task(tmp_path, probe_image)
    server = (
        "import http.server, socketserver, platform\n"
        "body = ('SIDE-RAN ' + platform.machine()).encode()\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers(); self.wfile.write(body)\n"
        "socketserver.TCPServer(('0.0.0.0', 8080), H).serve_forever()\n"
    )
    _compose(
        task,
        {
            "main": {},
            "side": {
                "image": ARM_IMAGE,
                "platform": "linux/arm64",
                "command": ["python3", "-c", server],
            },
        },
    )
    _script(task, "test.sh", _fetch("SIDE-RAN aarch64", tries=90))
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    result = _result(trial)
    assert (result.get("rewards") or {}).get("reward") == 1.0, (
        result.get("rewards"),
        result.get("error"),
        _lines(trial, "start.txt"),
    )


@pytest.mark.skipif(not _binfmt_aarch64(), reason="no QEMU binfmt for aarch64")
def test_the_aarch64_waiter_holds_until_released_under_qemu(tmp_path):
    """The pinned aarch64 bfx, under QEMU in an arm64 container on this
    x86_64 VM: nothing runs before the release, and the release starts the
    program. Under binfmt, ``/proc`` shows QEMU's process, not the waiter's
    (its cmdline and signal mask are QEMU's), so ``--ready``/``--release``
    can't be shown here; the release is the USR1 itself, after the guest
    has had time to install its handler."""
    import secrets
    import time

    from benchflow._bfx.locate import BfxUnavailable, verified_binary

    try:
        binary = verified_binary("aarch64")
    except BfxUnavailable as exc:
        pytest.skip(f"no pinned aarch64 bfx here: {exc}")
    token = secrets.token_hex(16)
    started = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--platform",
            "linux/arm64",
            "-v",
            f"{binary}:/.bfx/bfx:ro",
            "--entrypoint",
            "/.bfx/bfx",
            ARM_IMAGE,
            "hold",
            "--token",
            token,
            "--",
            "python3",
            "-c",
            "import platform; print('RELEASED', platform.machine(), flush=True); import time; time.sleep(60)",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if started.returncode:
        pytest.skip(f"could not start an arm64 container: {started.stderr[-200:]}")
    cid = started.stdout.strip()

    def logs() -> str:
        return subprocess.run(
            ["docker", "logs", cid], capture_output=True, text=True
        ).stdout

    try:
        time.sleep(10)
        state = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", cid],
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert state == "true" and "RELEASED" not in logs(), (state, logs())
        subprocess.run(["docker", "kill", "--signal", "USR1", cid], capture_output=True)
        for _ in range(120):
            if "RELEASED" in logs():
                break
            time.sleep(0.5)
        assert "RELEASED aarch64" in logs(), logs()
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)


# ---------------------------------------------- the waiter, restarts and scoring


def test_a_service_with_an_init_is_held_and_released_through_bfx(tmp_path, probe_image):
    """Under ``init: true`` docker-init is PID 1 and forwards a host
    USR1 to the waiter, which would die if that came before its handlers.
    Every held service is released only through ``bfx hold --release`` once
    ``--ready`` reads 0, so it starts and the trial scores as without init."""
    task = _task(tmp_path, probe_image)
    _compose(
        task,
        {
            "main": {"init": True},
            "side": {
                "image": probe_image,
                "init": True,
                "command": ["python3", "-c", SIDE_PROBE_PY],
            },
        },
    )
    _script(task, "test.sh", _fetch("SIDE-RAN"))
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    result = _result(trial)
    assert (result.get("rewards") or {}).get("reward") == 1.0, (
        result.get("rewards"),
        result.get("error"),
    )
    rows = _rows(_lines(trial, "start.txt"))
    assert rows and not _reached(rows), rows


AGENT_CRASH = (
    "python3 -c \"import urllib.request; urllib.request.urlopen('http://side:8080/crash', timeout=5)\" "
    "2>/dev/null; sleep 6\n"
)


def test_the_agent_crashing_a_sidecar_is_a_scored_failure_flagged_for_a_read(
    tmp_path, probe_image
):
    """The agent user (a ``script:`` agent) makes one request to the
    sidecar's ``/crash`` under ``public``; the sidecar stays down. The trial
    is scored as test.sh says (0: the sidecar is gone), flagged for a read,
    the reason recorded."""
    task = _restart_task(tmp_path, probe_image, "public", CRASH_PY)
    script = tmp_path / "crash-agent.sh"
    script.write_text(AGENT_CRASH)
    _script(task, "test.sh", _fetch("SIDE-RAN-AGAIN", tries=8))
    outcome, trial = _sdk_run(
        tmp_path, task, f"script:{script}", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    result = _result(trial)
    _exited(result, "the agent's turn", 1)
    assert (result.get("rewards") or {}).get("reward") == 0.0, (
        result.get("rewards"),
        result.get("verifier_error"),
    )


def _exited(result: dict, phase: str, exit_code: int) -> None:
    """A held sidecar that exited and did not start again: recorded with its
    exit status and phase, never the trial's ending; flagged for a read
    unless it exited at start-up (the task's own one-shot shape)."""
    assert not result.get("verifier_error"), result.get("verifier_error")
    assert not result.get("held_after_restart"), result.get("held_after_restart")
    exited = result.get("held_exited") or []
    assert [e.get("service") for e in exited] == ["side"], exited
    assert (exited[0].get("phase"), exited[0].get("exit_code")) == (phase, exit_code)
    flags = [w for w in result.get("warnings") or [] if "did not start again" in w]
    if phase == "start-up":
        assert not flags, flags
    else:
        assert flags and "read it" in flags[0], result.get("warnings")


def test_the_agent_crashing_a_sidecar_with_no_restart_policy_is_a_scored_failure(
    tmp_path, probe_image
):
    """The agent user asks the sidecar's ``/crash``; with no restart policy
    it stays down, with no network namespace and nothing running. The
    re-proofs can't check it, and that never makes the trial unscored: it is
    scored as test.sh says (0: the sidecar is gone), flagged for a read."""
    task = _restart_task(tmp_path, probe_image, "public", CRASH_PY, restart=None)
    script = tmp_path / "crash-agent.sh"
    script.write_text(AGENT_CRASH)
    _script(task, "test.sh", _fetch("SIDE-RAN", tries=8))
    outcome, trial = _sdk_run(
        tmp_path, task, f"script:{script}", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    result = _result(trial)
    assert (result.get("rewards") or {}).get("reward") == 0.0, (
        result.get("rewards"),
        result.get("verifier_error"),
    )
    _exited(result, "the agent's turn", 1)


def test_the_agent_asking_a_sidecar_with_no_restart_policy_is_scored_as_usual(
    tmp_path, probe_image
):
    """The guard: the agent asks ``/`` instead; the sidecar stays up and the
    trial is scored, with nothing recorded."""
    task = _restart_task(tmp_path, probe_image, "public", CRASH_PY, restart=None)
    script = tmp_path / "ask-agent.sh"
    script.write_text(AGENT_CRASH.replace("/crash", "/"))
    _script(task, "test.sh", _fetch("SIDE-RAN", tries=8))
    outcome, trial = _sdk_run(
        tmp_path, task, f"script:{script}", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    result = _result(trial)
    assert (result.get("rewards") or {}).get("reward") == 1.0, (
        result.get("rewards"),
        result.get("verifier_error"),
    )
    assert not result.get("held_exited"), result.get("held_exited")


def test_code_test_sh_runs_crashing_a_sidecar_is_scored_as_test_sh_says(
    tmp_path, probe_image
):
    """test.sh runs code that crashes the sidecar (no restart policy) and
    then passes: the end-of-test re-proof can't check the sidecar, and the
    pass stands, flagged for a read."""
    task = _restart_task(tmp_path, probe_image, "public", CRASH_PY, restart=None)
    agent = tmp_path / "idle-agent.sh"
    agent.write_text("true\n")
    _script(
        task,
        "test.sh",
        "#!/bin/bash\n" + AGENT_CRASH + "echo 1 > /logs/verifier/reward.txt\n",
    )
    outcome, trial = _sdk_run(
        tmp_path, task, f"script:{agent}", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    result = _result(trial)
    assert (result.get("rewards") or {}).get("reward") == 1.0, (
        result.get("rewards"),
        result.get("verifier_error"),
    )
    _exited(result, "test.sh", 1)


@pytest.mark.parametrize(
    ("code", "agent"), [(0, "oracle"), (1, "oracle"), (1, "script")]
)
def test_a_one_shot_sidecar_done_at_start_up_is_scored_and_recorded(
    tmp_path, probe_image, code, agent
):
    """A sidecar with no restart policy that runs a second after its release
    and exits (0 or 1), before the agent's turn: the task's own shape. The
    trial is scored as test.sh says, the exit recorded, not flagged. (On
    0.10.0 such a sidecar ran before any rule.) The oracle and a ``script:``
    agent reach their turn by different steps; each reads the exits first."""
    one_shot = f"import sys, time\ntime.sleep(1)\nsys.exit({code})\n"
    task = _restart_task(tmp_path, probe_image, "public", one_shot, restart=None)
    _script(task, "test.sh", PASS)
    if agent == "script":
        idle = tmp_path / "idle-agent.sh"
        idle.write_text("true\n")
        agent = f"script:{idle}"
    outcome, trial = _sdk_run(tmp_path, task, agent, sandbox_user="agent", flag=False)
    assert trial is not None, _refusal_text(outcome)
    result = _result(trial)
    assert (result.get("rewards") or {}).get("reward") == 1.0, (
        result.get("rewards"),
        result.get("verifier_error"),
    )
    _exited(result, "start-up", code)


_PLANT_PY = (
    "import socket\n"
    "s = socket.socket(); s.settimeout(3)\n"
    "try:\n"
    "    s.connect(('1.1.1.1', 80)); print('connected')\n"
    "except Exception as e:\n"
    "    print('refused', type(e).__name__)\n"
)


def _refused_and_flagged(result: dict, *, passed: bool) -> None:
    """test.sh's network refused it: the record names the phase, the count
    and whether test.sh passed, and the trial's warnings flag it."""
    refusal = result.get("verifier_network_refused") or {}
    assert refusal.get("phase") == "test.sh", result
    assert refusal.get("mode") == "no-network", refusal
    assert refusal.get("refused", 0) > 0, refusal
    assert refusal.get("passed") is passed, refusal
    assert any(
        "refused test.sh" in w and "scored as test.sh says" in w
        for w in result.get("warnings") or []
    ), result.get("warnings")


@pytest.mark.parametrize("whose", ["agent-planted", "test-sh-own"])
def test_a_refused_connection_in_test_sh_is_a_scored_0_flagged_for_a_read(
    tmp_path, probe_image, whose
):
    """Under ``no-network`` test.sh runs code that tries an outside
    connection: code the agent planted, or test.sh's own (a test that needs
    the network). It is refused, and BenchFlow can't tell whose it was: the
    trial is scored 0 as test.sh says, never unscored, with the refusal
    recorded and the trial flagged for a read."""
    task = _task(tmp_path, probe_image, policy="no-network")
    _compose(task, {"main": {}})
    if whose == "agent-planted":
        agent = tmp_path / "plant-agent.sh"
        agent.write_text(f"cat > /tmp/plant.py <<'EOF'\n{_PLANT_PY}EOF\n")
        prelude = ""
    else:
        agent = tmp_path / "idle-agent.sh"
        agent.write_text("true\n")
        prelude = f"cat > /tmp/plant.py <<'EOF'\n{_PLANT_PY}EOF\n"
    run = (
        "out=$(python3 /tmp/plant.py 2>&1)\n"
        'echo "$out" > /logs/verifier/plant.txt\n'
        'case "$out" in connected*) echo 1;; *) echo 0;; esac > /logs/verifier/reward.txt\n'
    )
    _script(task, "test.sh", "#!/bin/bash\n" + prelude + run)
    outcome, trial = _sdk_run(
        tmp_path, task, f"script:{agent}", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    result = _result(trial)
    assert (result.get("rewards") or {}).get("reward") == 0.0, (
        result.get("rewards"),
        result.get("verifier_error"),
        _lines(trial, "plant.txt"),
    )
    assert not result.get("verifier_error"), result.get("verifier_error")
    _refused_and_flagged(result, passed=False)


def test_a_test_sh_that_passes_despite_a_refusal_stays_a_pass_flagged(
    tmp_path, probe_image
):
    """A refused connection that test.sh does not need: its pass stands, and
    the trial is flagged for a read with the refusal recorded."""
    task = _task(tmp_path, probe_image, policy="no-network")
    _compose(task, {"main": {}})
    agent = tmp_path / "idle-agent.sh"
    agent.write_text("true\n")
    _script(
        task,
        "test.sh",
        "#!/bin/bash\n"
        f"cat > /tmp/plant.py <<'EOF'\n{_PLANT_PY}EOF\n"
        "python3 /tmp/plant.py > /logs/verifier/plant.txt 2>&1\n"
        "echo 1 > /logs/verifier/reward.txt\n",
    )
    outcome, trial = _sdk_run(
        tmp_path, task, f"script:{agent}", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    result = _result(trial)
    assert (result.get("rewards") or {}).get("reward") == 1.0, (
        result.get("rewards"),
        result.get("verifier_error"),
        _lines(trial, "plant.txt"),
    )
    assert not result.get("verifier_error"), result.get("verifier_error")
    _refused_and_flagged(result, passed=True)
