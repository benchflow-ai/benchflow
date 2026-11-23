"""``kill-after-grading.json`` on real Docker (0.10.1).

Deterministic tier (``tests/integration/README.md``): real sandboxes, no
model. The oracle runs each task's solution as the agent user
(``[agent] user = "agent"``) in the default mode, audit, where the agent's
processes live through grading and the rollout ends them after it
(``planes.quiesce_agent``). 0.10.0 wrote no record of that kill.

One task's solve.sh leaves four processes, one per SIGTERM disposition the
record reads: the default action (``sleep``), ignored, blocked, and handled.
Another leaves nothing. A test-only wrapper around the quiesce lists the
agent user's processes with Python's own /proc reader just before it runs:
the record's pids and commands must be that listing's.

The listing is root's shell with the runtime kit's tools, so it must work on
an image without Python: Debian slim, and Alpine's BusyBox, with
bash added only because the oracle starts solve.sh as the agent user through
``su -s /bin/bash`` (as the repo's other Alpine trials add it). There a shell
solve.sh leaves processes of three dispositions, and the host's ``docker top``
is the independent listing.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic
DOCKER_RUN_DEFAULTS = True

SANDBOX, SKIP_REASON = h.select_sandbox()

TASK_TOML = """version = "1.0"

[metadata]
author_name = "benchflow"
difficulty = "easy"
category = "fixture"

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
DOCKERFILE = (
    "FROM python:3.12-slim@sha256:"
    "a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1\n"
    "WORKDIR /app\n"
    "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts\n"
)
TEST_SH = "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"


def _python(marker: str, setup: str) -> str:
    code = f"import signal, time; {setup}; time.sleep(600)"
    return f"setsid nohup python3 -c {shlex.quote(code)} {marker} </dev/null >/dev/null 2>&1 &\n"


# Each leftover carries a marker in its command line; the expected signal.
LEFTOVERS = {
    "sleep 600": "SIGTERM",
    "bf-ignored": "SIGKILL",
    "bf-blocked": "SIGKILL",
    "bf-caught": "SIGTERM or SIGKILL",
}
SOLVE_LEFTOVERS = (
    "#!/bin/bash\n"
    "setsid nohup sleep 600 </dev/null >/dev/null 2>&1 &\n"
    + _python("bf-ignored", "signal.signal(signal.SIGTERM, signal.SIG_IGN)")
    + _python(
        "bf-blocked", "signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})"
    )
    + _python("bf-caught", "signal.signal(signal.SIGTERM, lambda *a: None)")
    + "sleep 2\n"
)
SOLVE_NOTHING = "#!/bin/bash\ntrue\n"

# Root, in the sandbox: every process whose real, effective or saved uid is
# the agent user's, from /proc, with its command line.
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
    if status.get("State", "").startswith("Z"):
        continue
    rows.append({"pid": int(pid), "cmd": cmd})
print("AGENT-PROCESSES " + json.dumps(rows))
"""


@pytest.fixture
def require_docker() -> None:
    if SANDBOX != "docker":
        reason = SKIP_REASON if SANDBOX is None else f"Docker-only scenario ({SANDBOX})"
        pytest.skip(reason)


def _task(
    root: Path,
    name: str,
    solve: str,
    dockerfile: str = DOCKERFILE,
    test_sh: str = TEST_SH,
) -> Path:
    task = root / name
    for sub in ("environment", "solution", "tests"):
        (task / sub).mkdir(parents=True)
    (task / "task.toml").write_text(TASK_TOML)
    (task / "instruction.md").write_text("Leave whatever solve.sh leaves.\n")
    (task / "environment" / "Dockerfile").write_text(dockerfile)
    for path, text in (
        (task / "solution" / "solve.sh", solve),
        (task / "tests" / "test.sh", test_sh),
    ):
        path.write_text(text)
        path.chmod(0o755)
    return task


def _watch_quiesce(monkeypatch) -> list:
    """Test-only: list the agent user's processes, Python's way, as each
    ``planes.quiesce_agent`` call starts."""
    from benchflow.rollout_planes import DefaultRolloutPlanes
    from benchflow.sandbox._exec_output import exec_raw

    seen: list = []
    real = DefaultRolloutPlanes.quiesce_agent

    async def watched(self, env, sandbox_user, **kwargs):
        result = await exec_raw(
            env.exec,
            f"/usr/local/bin/python3 -c {shlex.quote(AGENT_PROCESSES)} {sandbox_user}",
            user="root",
            timeout_sec=60,
        )
        line = next(
            line
            for line in (result.stdout or "").splitlines()
            if line.startswith("AGENT-PROCESSES ")
        )
        seen.append(json.loads(line.removeprefix("AGENT-PROCESSES ")))
        return await real(self, env, sandbox_user, **kwargs)

    monkeypatch.setattr(DefaultRolloutPlanes, "quiesce_agent", watched)
    return seen


async def _run(
    root: Path, name: str, solve: str, **files: str
) -> tuple[dict | None, Path]:
    from benchflow.sdk import SDK

    task = _task(root / "tasks", name, solve, **files)
    jobs = root / "jobs" / name
    result = await SDK().run(
        task_path=task, agent="oracle", jobs_dir=jobs, environment="docker"
    )
    (trial,) = {p.parent for p in jobs.glob("**/verifier") if p.is_dir()}
    return result.rewards, trial


@pytest.mark.asyncio
async def test_an_audit_trial_records_the_processes_its_agent_left(
    tmp_path, monkeypatch, require_docker
):
    seen = _watch_quiesce(monkeypatch)
    rewards, trial = await _run(tmp_path, "leftovers", SOLVE_LEFTOVERS)
    assert rewards == {"reward": 1.0}
    path = trial / "kill-after-grading.json"
    assert path.exists(), sorted(p.name for p in trial.iterdir())
    assert not (trial / "kill-before-grading.json").exists()
    record = json.loads(path.read_text())
    assert record["user"] == "agent"
    assert record["signal_source"] == "sigterm_disposition"
    assert record["started_at"] <= record["ended_at"]
    assert record["left"] == []
    rows = record["processes"]
    assert all(row["user"] == "agent" and row["ended"] is True for row in rows), rows
    for marker, signal in LEFTOVERS.items():
        (row,) = [row for row in rows if marker in row["command"]]
        assert row["ended_by"] == signal, (marker, row)
    assert record["killed"] == len(rows)
    # The independent listing, made as the quiesce started, saw the same.
    assert seen, "the quiesce never ran"
    listed = {(row["pid"], row["cmd"]) for row in seen[-1]}
    assert {(row["pid"], row["command"]) for row in rows} == listed


@pytest.mark.asyncio
async def test_an_audit_trial_with_nothing_left_records_an_empty_list(
    tmp_path, monkeypatch, require_docker
):
    seen = _watch_quiesce(monkeypatch)
    rewards, trial = await _run(tmp_path, "nothing", SOLVE_NOTHING)
    assert rewards == {"reward": 1.0}
    path = trial / "kill-after-grading.json"
    assert path.exists(), sorted(p.name for p in trial.iterdir())
    record = json.loads(path.read_text())
    assert record["processes"] == []
    assert record["left"] == []
    assert record["killed"] == 0
    assert seen and seen[-1] == []


# Images without Python, pinned; the extra lines their Dockerfile needs.
NO_PYTHON_IMAGES = {
    "debian-slim": "debian:bookworm-slim@sha256:"
    "734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc",
    "alpine": "alpine:3.24@sha256:"
    "294b683cb724975bec92580e1e685676bd4b50bda910ddb8c51d4cabeaec77e6",
}
TEST_SH_POSIX = "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
# A plain sleep; one that ignores SIGTERM (an ignored signal stays ignored
# across exec); a shell that handles it, waiting on a sleep of its own.
SOLVE_SHELL = (
    "#!/bin/sh\n"
    "setsid nohup sleep 601 </dev/null >/dev/null 2>&1 &\n"
    "setsid nohup sh -c 'trap \"\" TERM; exec sleep 602' </dev/null >/dev/null 2>&1 &\n"
    "setsid nohup sh -c 'trap : TERM; sleep 603 & wait' bf-caught "
    "</dev/null >/dev/null 2>&1 &\n"
    "sleep 2\n"
)


def _watch_quiesce_from_the_host(monkeypatch) -> list:
    """Test-only: as each ``planes.quiesce_agent`` call starts, the commands
    of the agent user's processes, as the host's ``docker top`` lists the
    container's (no Python in the image to ask)."""
    import asyncio

    from benchflow.rollout_planes import DefaultRolloutPlanes
    from benchflow.sandbox._exec_output import exec_raw

    seen: list = []
    real = DefaultRolloutPlanes.quiesce_agent

    async def watched(self, env, sandbox_user, **kwargs):
        uid = await exec_raw(
            env.exec, f"id -u {sandbox_user}", user="root", timeout_sec=60
        )
        container = await env._main_container_id()
        top = await asyncio.create_subprocess_exec(
            "docker",
            "top",
            container,
            "-eo",
            "pid,uid,args",
            stdout=asyncio.subprocess.PIPE,
        )
        out, _ = await top.communicate()
        want = (uid.stdout or "").strip()
        rows = []
        for line in out.decode().splitlines()[1:]:
            fields = line.split(None, 2)
            if len(fields) == 3 and fields[1] == want:
                rows.append(fields[2].strip())
        seen.append(sorted(rows))
        return await real(self, env, sandbox_user, **kwargs)

    monkeypatch.setattr(DefaultRolloutPlanes, "quiesce_agent", watched)
    return seen


@pytest.mark.parametrize("image", sorted(NO_PYTHON_IMAGES))
@pytest.mark.asyncio
async def test_an_image_without_python_records_its_leftovers(
    tmp_path, monkeypatch, require_docker, image
):
    seen = _watch_quiesce_from_the_host(monkeypatch)
    dockerfile = (
        f"FROM {NO_PYTHON_IMAGES[image]}\n"
        + ("RUN apk add --no-cache bash\n" if image == "alpine" else "")
        + "WORKDIR /app\n"
        "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts\n"
    )
    rewards, trial = await _run(
        tmp_path,
        f"shell-{image}",
        SOLVE_SHELL,
        dockerfile=dockerfile,
        test_sh=TEST_SH_POSIX,
    )
    assert rewards == {"reward": 1.0}
    path = trial / "kill-after-grading.json"
    assert path.exists(), sorted(p.name for p in trial.iterdir())
    record = json.loads(path.read_text())
    assert "listing_error" not in record, record.get("listing_error")
    assert record["signal_source"] == "sigterm_disposition"
    assert record["left"] == []
    rows = record["processes"]
    assert all(row["user"] == "agent" and row["ended"] is True for row in rows), rows
    by_command = {row["command"]: row["ended_by"] for row in rows}
    assert by_command.pop("sleep 601") == "SIGTERM"
    assert by_command.pop("sleep 602") == "SIGKILL"
    assert by_command.pop("sleep 603") == "SIGTERM"
    ((caught, signal),) = by_command.items()
    assert "bf-caught" in caught and signal == "SIGTERM or SIGKILL"
    assert record["killed"] == len(rows)
    assert seen, "the quiesce never ran"
    assert sorted(row["command"] for row in rows) == seen[-1]


@pytest.mark.parametrize("image", sorted(NO_PYTHON_IMAGES))
def test_the_images_own_tools_list_a_stand_in_proc_as_the_host_does(
    tmp_path, require_docker, image
):
    """S-1 and S-4 on BusyBox and Debian's tools: the listing run in the
    image (its own sh, tr and cut; no kit here; Debian's sh is dash) over a
    stand-in /proc, command lines with line breaks and backslash escapes
    included, gives the rows the host's tools give."""
    import subprocess

    from benchflow.sandbox import _quiesce_listing
    from tests.test_kill_after_grading_record import (
        BREAKS,
        TERM,
        _list,
        _proc,
        add_escape_cases,
    )

    proc = tmp_path / "proc"
    _proc(proc, 101)
    _proc(proc, 102, ign=TERM, name="ignored")
    _proc(proc, 103, cgt=TERM, name="caught")
    for index, brk in enumerate(BREAKS):
        forged = f"x{brk}BF proc 4242 agent S 1 default forged{brk}BF clock 1 1"
        _proc(proc, 110 + index, cmdline=forged.encode() + b"\x00")
    _proc(proc, 130, cmdline=b"", name="n\u0085BF")
    escapes = add_escape_cases(proc)  # S-4: \n, \c (middle, last), \0012
    host = _list(proc)  # writes proc/stat
    command = _quiesce_listing.list_command([("agent", 1000)], proc="/p")
    done = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{proc}:/p:ro",
            NO_PYTHON_IMAGES[image],
            "sh",
            "-c",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert _quiesce_listing.parse_listing(done.stdout) == host
    assert sorted(row["pid"] for row in host) == sorted(
        [101, 102, 103, 130, *range(110, 110 + len(BREAKS)), *escapes]
    )


# A real audit trial: python:3.12-slim's sh is
# dash, and the agent leaves processes whose arguments hold \n and \c.
SOLVE_ESCAPES = (
    "#!/bin/bash\n"
    "setsid nohup sh -c 'sleep 600; :' 'x\\nBF proc 4242 agent S 1 default forged'"
    " </dev/null >/dev/null 2>&1 &\n"
    "setsid nohup sh -c 'sleep 601; :' 'y\\c' </dev/null >/dev/null 2>&1 &\n"
    "sleep 2\n"
)


@pytest.mark.asyncio
async def test_a_command_line_cannot_write_the_record_on_dash(
    tmp_path, monkeypatch, require_docker
):
    """S-4: no row the agent wrote, and every agent process root's Python
    listing saw is in the record, its command line as text."""
    seen = _watch_quiesce(monkeypatch)
    rewards, trial = await _run(tmp_path, "escapes", SOLVE_ESCAPES)
    assert rewards == {"reward": 1.0}
    record = json.loads((trial / "kill-after-grading.json").read_text())
    assert "listing_error" not in record, record
    pids = sorted(row["pid"] for row in record["processes"])
    assert 4242 not in pids, record
    assert seen, "the quiesce never ran"
    assert pids == sorted(row["pid"] for row in seen[-1]), (record, seen[-1])
    commands = {row["command"] for row in record["processes"]}
    assert "sh -c sleep 600; : x\\nBF proc 4242 agent S 1 default forged" in commands
    assert "sh -c sleep 601; : y\\c" in commands
    assert record["killed"] == len(pids) == 4


@pytest.mark.parametrize("case", ["fields", "fields-are-globs", "glob", "plain"])
@pytest.mark.parametrize("image", sorted(NO_PYTHON_IMAGES))
def test_a_newline_in_the_comm_cannot_choose_the_start_on_the_image(
    tmp_path, require_docker, image, case
):
    """N-6 with each image's own sh (Debian's is dash), its cwd a folder of
    files named by digits."""
    import subprocess

    from benchflow.sandbox import _quiesce_listing
    from tests.test_kill_after_grading_record import (
        BTIME,
        add_stat_case,
        digit_named_files,
    )

    proc = tmp_path / "proc"
    add_stat_case(proc, case)
    (proc / "stat").write_text(f"cpu  1 2 3\nbtime {BTIME}\nprocesses 9\n")
    work = digit_named_files(tmp_path / "work")
    command = _quiesce_listing.list_command([("agent", 1000)], proc="/p")
    done = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{proc}:/p:ro",
            "-v",
            f"{work}:/w:ro",
            "-w",
            "/w",
            NO_PYTHON_IMAGES[image],
            "sh",
            "-c",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    (row,) = _quiesce_listing.parse_listing(done.stdout)
    assert row["start_ticks"] == 12345, row


# A real audit trial: python:3.12-slim, WORKDIR
# /app, files named by digits there, and a process whose comm holds a newline
# and globs. Root's Python reads each real start after the last ")".
SOLVE_COMM = (
    "#!/bin/bash\n"
    "cd /app\n"
    'for n in 77 78 79 80; do : > "$PWD/$n"; done\n'
    "setsid nohup sh -c 'printf \") * * * * * *\\n\" > /proc/self/comm; sleep 600; :'"
    " </dev/null >/dev/null 2>&1 &\n"
    "sleep 2\n"
)
STARTS = r"""
import json, os
out = {}
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        data = open(f"/proc/{pid}/stat", "rb").read()
    except OSError:
        continue
    out[pid] = int(data[data.rindex(b")") + 2 :].split()[19])
print("STARTS " + json.dumps(out))
"""


@pytest.mark.asyncio
async def test_a_trial_process_cannot_choose_its_start(
    tmp_path, monkeypatch, require_docker
):
    """N-6 in a trial: each record row's start time is the kernel's, as root's
    Python reads it just before the quiesce."""
    from benchflow.rollout_planes import DefaultRolloutPlanes
    from benchflow.sandbox._exec_output import exec_raw

    seen: list = []
    real = DefaultRolloutPlanes.quiesce_agent

    async def watched(self, env, sandbox_user, **kwargs):
        result = await exec_raw(
            env.exec,
            f"/usr/local/bin/python3 -c {shlex.quote(STARTS)}",
            user="root",
            timeout_sec=60,
        )
        line = next(
            x for x in (result.stdout or "").splitlines() if x.startswith("STARTS ")
        )
        seen.append(json.loads(line.removeprefix("STARTS ")))
        return await real(self, env, sandbox_user, **kwargs)

    monkeypatch.setattr(DefaultRolloutPlanes, "quiesce_agent", watched)
    rewards, trial = await _run(tmp_path, "comm", SOLVE_COMM)
    assert rewards == {"reward": 1.0}
    record = json.loads((trial / "kill-after-grading.json").read_text())
    assert seen, "the quiesce never ran"
    rows = record["processes"]
    assert rows, record
    assert {r["pid"]: r["start_ticks"] for r in rows} == {
        r["pid"]: seen[-1][str(r["pid"])] for r in rows
    }
