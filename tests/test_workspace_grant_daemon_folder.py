"""The workspace grant keeps a provider daemon's folder root's (#1232).

Daytona's daemon runs as root and keeps each exec's ``cmd.sh``, ``output.log``
and ``exit_code`` under ``.daytona`` in root's home. With a ``/root``
workspace, ``chown -R agent /root`` handed that folder to the agent, while the
review capture leaves it out as runtime state (``SANDBOX_RUNTIME_STATE_PATHS``):
a file the agent put there was never reviewed and stayed within test.sh's
reach. The grant now reserves it as it does the loader's folders.

Real Docker (``python:3.12-slim``), the daemon's folder laid out as on a live
Daytona sandbox; the grant runs as root, the checks as the agent user.
Skipped without a Docker daemon.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid

import pytest

from benchflow import _runtime_tree
from benchflow._runtime_tree import workspace_grant_command
from benchflow.sandbox import lockdown
from tests.test_c8_protected_runtime_container import _sandbox

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="docker required"
)

IMAGE = "python:3.12-slim"
LABEL = os.environ.get("BENCHFLOW_TEST_CONTAINER_LABEL", "PT=1")
SESSION = "/root/.daytona/sessions/s1/c1"


def _docker_ready() -> bool:
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except Exception:
        return False
    return probe.returncode == 0


def _dx(name: str, command: str, user: str = "0") -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", "-u", user, name, "/bin/sh", "-c", command],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def box():
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = "bf-grant-daemon-" + uuid.uuid4().hex[:8]
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


def test_the_grant_keeps_the_daemon_folder_roots(box):
    """Guards #1232: with a /root workspace holding ``.daytona``, the agent
    gets the rest of /root but cannot add, change or move anything in the
    daemon's folder, and root (the daemon) still writes there."""
    made = _dx(
        box,
        "useradd -m agent && "
        f"mkdir -p {SESSION} && echo 'echo hi' > {SESSION}/cmd.sh && "
        f"chmod 600 {SESSION}/cmd.sh && echo hi > {SESSION}/output.log && "
        "echo notes > /root/notes.md",
    )
    assert made.returncode == 0, made.stderr

    grant = _dx(box, workspace_grant_command("/root", "agent"))
    assert grant.returncode == 0, grant.stderr

    owner = _dx(box, "stat -c '%U %a' /root/.daytona /root/.daytona/sessions")
    assert owner.stdout.split("\n")[:2] == ["root 755", "root 755"], owner.stdout
    for attempt in (
        "echo 'HIDDEN = 1' > /root/.daytona/x.py",
        "mkdir /root/.daytona/sessions/fake",
        f"echo 'HIDDEN = 1' > {SESSION}/evil.py",
        "mv /root/.daytona /root/moved",
        "rm -rf /root/.daytona",
    ):
        done = _dx(box, attempt, user="agent")
        assert done.returncode != 0, attempt
    assert _dx(box, "ls -A /root/.daytona").stdout.split() == ["sessions"]
    # The daemon runs as root: its next exec's files still go in.
    daemon = _dx(
        box,
        "mkdir -p /root/.daytona/sessions/s2/c2 && "
        "echo 0 > /root/.daytona/sessions/s2/c2/exit_code",
    )
    assert daemon.returncode == 0, daemon.stderr
    # The rest of the workspace is the agent's.
    work = _dx(
        box,
        "echo more >> /root/notes.md && echo new > /root/new.txt && "
        "mkdir /root/results && mv /root/new.txt /root/results/",
        user="agent",
    )
    assert work.returncode == 0, work.stderr


# An agent trial's setup, and a scripted run's (the oracle, a regrade, a
# verifier recovery), which passes guard_daemon=False.
GUARDS = pytest.mark.parametrize(
    "guard_daemon", [True, False], ids=["agent", "scripted"]
)


@pytest.mark.asyncio
@GUARDS
@pytest.mark.parametrize(("image_mode", "restored"), [("700", "700"), ("777", "1777")])
async def test_test_sh_gets_the_workspace_in_its_image_mode(
    box, image_mode, restored, guard_daemon
):
    """The grant made /root root's and 1777 for the agent's turn; the
    hardening gives it back the image's mode before test.sh, sticky when
    others may write. The daemon folder stays root's and in place. A
    scripted run keeps the mode too, so its test.sh sees /root as an agent
    trial's does: it kept none, and its test.sh saw 1777 (a review L2)."""
    made = _dx(box, f"mkdir -p {SESSION} && chmod {image_mode} /root")
    assert made.returncode == 0, made.stderr
    sandbox = _sandbox(box)
    await lockdown.setup_sandbox_user(
        sandbox, "agent", "/root", guard_daemon=guard_daemon
    )
    assert _dx(box, "stat -c '%U %a' /root").stdout.split() == ["root", "1777"]
    # A second grant on the sandbox keeps the image's mode, not 1777.
    await lockdown.setup_sandbox_user(
        sandbox, "agent", "/root", guard_daemon=guard_daemon
    )
    assert getattr(sandbox, _runtime_tree.WORKSPACE_MODE_ATTR) == {"/root": image_mode}

    steps = lockdown._freeze_steps("/root", modes=lockdown._granted_modes(sandbox))
    assert steps[0].name == "restore_workspace_mode"
    await lockdown._run_prep_step(sandbox, steps[0])
    owner = _dx(box, "stat -c '%U %a' /root /root/.daytona")
    assert owner.stdout.split("\n")[:2] == [f"root {restored}", "root 755"]


@pytest.mark.asyncio
@GUARDS
async def test_a_linked_workspace_gets_its_folders_mode_back(box, guard_daemon):
    """The workspace is a link, ``/ws`` to ``/data`` (image mode 700). The
    mode kept is the folder's, under its own path, so test.sh sees /data at
    700. ``stat`` without ``-L`` read the link's own mode: ``{'/data':
    '777'}``, and test.sh saw 1777 (a review L1)."""
    made = _dx(
        box,
        "mkdir -p /data/.daytona/sessions/s1/c1 && chmod 700 /data && ln -s /data /ws",
    )
    assert made.returncode == 0, made.stderr
    sandbox = _sandbox(box)
    await lockdown.setup_sandbox_user(
        sandbox, "agent", "/ws", guard_daemon=guard_daemon
    )
    assert _dx(box, "stat -c '%U %a' /data").stdout.split() == ["root", "1777"]
    assert getattr(sandbox, _runtime_tree.WORKSPACE_MODE_ATTR) == {"/data": "700"}

    steps = lockdown._freeze_steps("/data", modes=lockdown._granted_modes(sandbox))
    assert steps[0].name == "restore_workspace_mode"
    await lockdown._run_prep_step(sandbox, steps[0])
    owner = _dx(box, "stat -c '%U %a' /data /data/.daytona")
    assert owner.stdout.split("\n")[:2] == ["root 700", "root 755"]


def test_without_a_grant_mode_the_freeze_is_unchanged():
    """No daemon folder (Docker, Modal): no step is added to the hardening."""
    names = [step.name for step in lockdown._freeze_steps("/app")]
    assert "restore_workspace_mode" not in names
    assert names == [step.name for step in lockdown._freeze_steps("/app", modes={})]
