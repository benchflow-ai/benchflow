"""A root oracle rewrites an agent-owned input in a workspace the grant made
sticky (benchflow-ai/benchflow#1249, reported by Benjamin-eecs).

With Daytona's daemon folder in a ``/root`` workspace, the grant makes
``/root`` root's and 1777 and gives every other entry to the sandbox user
(#1232), in oracle runs too. Daytona runs ``fs.protected_regular=2``: root
cannot ``O_CREAT``-open a file another user owns in a sticky folder others
may write, so a root solve.sh that rewrote an input in place failed with
EACCES and the oracle scored 0 on Daytona while it passed on Docker.

Real Docker (``python:3.12-slim``), the daemon's folder laid out as on a live
Daytona sandbox. ``fs.protected_regular`` is the kernel's, shared by every
container: the tests are skipped on a host where it is 0.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._runtime_tree import WORKSPACE_MODE_ATTR
from benchflow.sandbox import lockdown
from tests.test_c8_protected_runtime_container import _sandbox


def _protected_regular() -> int:
    try:
        return int(Path("/proc/sys/fs/protected_regular").read_text().strip())
    except (OSError, ValueError):
        return 0


pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
    pytest.mark.skipif(
        _protected_regular() < 1, reason="the kernel's fs.protected_regular is 0"
    ),
]

IMAGE = "python:3.12-slim"
LABEL = os.environ.get("BENCHFLOW_TEST_CONTAINER_LABEL", "PT=1")
SESSION = "/root/.daytona/sessions/s1/c1"
SOLVE = "#!/bin/bash\nset -e\nprintf 'new\\n' > /root/input.txt\n"


def _dx(name: str, command: str, user: str = "0") -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", "-u", user, name, "/bin/sh", "-c", command],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def box():
    probe = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    if probe.returncode != 0:
        pytest.skip("docker daemon not running")
    name = "bf-oracle-rewrite-" + uuid.uuid4().hex[:8]
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
        made = _dx(
            name,
            f"mkdir -p {SESSION} /solution /logs/agent /logs/verifier && "
            "chmod 700 /root && echo old > /root/input.txt",
        )
        assert made.returncode == 0, made.stderr
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _task(root: Path, name: str, extra_toml: str = "", solve: str = SOLVE) -> Path:
    (root / "solution").mkdir(parents=True)
    (root / "solution" / "solve.sh").write_text(solve)
    (root / "instruction.md").write_text("Replace /root/input.txt with new.\n")
    (root / "environment").mkdir()
    (root / "task.toml").write_text('version = "1.0"\n' + extra_toml)
    copied = subprocess.run(
        ["docker", "cp", str(root / "solution" / "solve.sh"), f"{name}:/solution/"],
        capture_output=True,
        text=True,
    )
    assert copied.returncode == 0, copied.stderr
    return root


def _oracle_rc(trajectory: list[dict]) -> int | None:
    return next(
        (s.get("return_code") for s in trajectory if s.get("type") == "oracle"), None
    )


@pytest.mark.asyncio
async def test_root_is_refused_an_agent_owned_file_in_the_sticky_grant(box):
    """The kernel rule itself: after the grant, root's ``>`` of the agent's
    file in /root is refused, and so is Python's ``open(path, "w")``."""
    await lockdown.setup_sandbox_user(
        _sandbox(box), "agent", "/root", guard_daemon=False
    )
    owners = _dx(box, "stat -c '%U %a' /root /root/input.txt").stdout.split("\n")
    assert owners[:2] == ["root 1777", "agent 644"]
    shell = _dx(box, "echo new > /root/input.txt")
    assert shell.returncode != 0 and "Permission denied" in shell.stderr
    python = _dx(box, "python3 -c \"open('/root/input.txt', 'w')\"")
    assert python.returncode != 0 and "PermissionError" in python.stderr


@pytest.mark.asyncio
async def test_a_root_oracle_rewrites_an_agent_owned_input_in_place(box, tmp_path):
    """#1249: solve.sh runs as root (the task names no [agent].user) and
    rewrites the input. The workspace loses group and other write first
    (1755) and gets the grant's 1777 back after; the daemon folder stays
    root's."""
    from benchflow.rollout._setup import _run_oracle

    sandbox = _sandbox(box)
    await lockdown.setup_sandbox_user(sandbox, "agent", "/root", guard_daemon=False)

    trajectory, _ = await _run_oracle(sandbox, _task(tmp_path, box), timeout=60)

    assert _oracle_rc(trajectory) == 0, trajectory
    assert _dx(box, "cat /root/input.txt").stdout == "new\n"
    owners = _dx(box, "stat -c '%U %a' /root /root/.daytona").stdout.split("\n")
    assert owners[:2] == ["root 1777", "root 755"]


@pytest.mark.asyncio
async def test_an_oracle_run_as_the_task_user_keeps_the_grant(box, tmp_path):
    """A task whose [agent].user is the sandbox user runs solve.sh as that
    user: the workspace stays the grant's, which that user needs, and the
    file it owns is its to rewrite."""
    from benchflow.rollout._setup import _run_oracle

    sandbox = _sandbox(box)
    await lockdown.setup_sandbox_user(sandbox, "agent", "/root", guard_daemon=False)
    task = _task(tmp_path, box, '[agent]\nuser = "agent"\n')

    trajectory, _ = await _run_oracle(sandbox, task, timeout=60)

    assert _oracle_rc(trajectory) == 0, trajectory
    assert _dx(box, "cat /root/input.txt").stdout == "new\n"
    assert _dx(box, "stat -c '%U %a' /root").stdout.split() == ["root", "1777"]


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace", ["/opt", "/usr/local"])
async def test_a_root_oracle_rewrites_an_agent_owned_input_in_an_opt_workspace(
    box, tmp_path, workspace
):
    """Row A3 (#1249 (c)): a workspace of /opt holds BenchFlow's reserved
    /opt/runtime, and /usr/local the loader's /usr/local/lib*, so the grant
    makes it root's and 1777 on every backend, plain Docker included, with no
    daemon folder. The grant records it as it does a daemon folder's
    workspace, and a root solve.sh meets it without group or other write.

    Mutant: record the mode only for a daemon folder (0.10.0's rule)."""
    from benchflow.rollout._setup import _run_oracle

    made = _dx(box, f"chmod 755 {workspace} && echo old > {workspace}/input.txt")
    assert made.returncode == 0, made.stderr
    sandbox = _sandbox(box)
    await lockdown.setup_sandbox_user(sandbox, "agent", workspace, guard_daemon=False)
    granted = _dx(box, f"stat -c '%U %a' {workspace} {workspace}/input.txt").stdout
    assert granted.split("\n")[:2] == ["root 1777", "agent 644"], granted
    solve = f"#!/bin/bash\nset -e\nprintf 'new\\n' > {workspace}/input.txt\n"
    task = _task(
        tmp_path, box, f'[environment]\nworkdir = "{workspace}"\n', solve=solve
    )

    trajectory, _ = await _run_oracle(sandbox, task, timeout=60)

    assert _oracle_rc(trajectory) == 0, trajectory
    assert _dx(box, f"cat {workspace}/input.txt").stdout == "new\n"
    assert _dx(box, f"stat -c '%U %a' {workspace}").stdout.split() == ["root", "1777"]


@pytest.mark.asyncio
async def test_a_lost_record_still_gets_the_step(box, tmp_path, monkeypatch):
    """Row N1 (review): the read before the grant is lost (its receipt
    fails), so no mode is known; the workspace is still recorded and the
    step before a root solve.sh acts on what the folder is, never skipped.

    Mutant: record nothing when the read is lost (0.10.0's rule)."""
    from benchflow.rollout._setup import _run_oracle
    from benchflow.sandbox._exec_receipt import RECEIPT_FAILED

    real = lockdown._read_exec

    async def lose_the_resolve(env, command, /, **kwargs):
        if str(command).startswith("cd /root && pwd -P"):
            return SimpleNamespace(return_code=RECEIPT_FAILED, stdout="", stderr="")
        return await real(env, command, **kwargs)

    monkeypatch.setattr(lockdown, "_read_exec", lose_the_resolve)
    sandbox = _sandbox(box)
    await lockdown.setup_sandbox_user(sandbox, "agent", "/root", guard_daemon=False)
    assert getattr(sandbox, WORKSPACE_MODE_ATTR) == {"/root": ""}

    trajectory, _ = await _run_oracle(sandbox, _task(tmp_path, box), timeout=60)

    assert _oracle_rc(trajectory) == 0, trajectory
    assert _dx(box, "cat /root/input.txt").stdout == "new\n"
