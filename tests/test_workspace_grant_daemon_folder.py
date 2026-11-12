"""The workspace grant keeps a provider daemon's folder root's (#1232).

Daytona's daemon runs as root and keeps each exec's ``cmd.sh``, ``output.log``
and ``exit_code`` under ``.daytona`` in root's home. With a ``/root``
workspace, ``chown -R agent /root`` handed that folder to the agent, while the
review capture leaves it out as runtime state (``SANDBOX_RUNTIME_STATE_PATHS``):
a file the agent put there was never reviewed and stayed within test.sh's
reach. The grant now keeps the folder root's and bind-mounts it onto itself,
so the agent can neither write there nor move it away, and leaves the
workspace the agent's.

The first version of this fix (PR #1234, commit cfcdc841) made the workspace
root's and sticky instead. Daytona runs with ``fs.protected_regular=2``,
under which root may not open an existing file of another user's in a sticky
folder others can write with ``O_CREAT``: an oracle running as root could no
longer rewrite an input in place (weighted-gdp-calc's ``/root/gdp.xlsx``).
The root rewrites below guard against that layout; they fail on it only where
the host kernel's ``protected_regular`` is 1 or 2 (Ubuntu's default; Docker
Desktop's VM has 0).

Real Docker (``python:3.12-slim``), the daemon's folder laid out as on a live
Daytona sandbox; the grant runs as root, the checks as the agent user. Root
has ``CAP_SYS_ADMIN`` on Daytona: the pinned box adds it (and lifts AppArmor's
mount rule) so the mount works as there, the plain box keeps Docker's
defaults, which refuse it. Skipped without a Docker daemon.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid

import pytest

from benchflow._runtime_tree import workspace_grant_command

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="docker required"
)

IMAGE = "python:3.12-slim"
LABEL = os.environ.get("BENCHFLOW_TEST_CONTAINER_LABEL", "PT=1")
SESSION = "/root/.daytona/sessions/s1/c1"
MOUNT_CAPS = ["--cap-add", "SYS_ADMIN", "--security-opt", "apparmor=unconfined"]


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


def _box(options: list[str]):
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = "bf-grant-daemon-" + uuid.uuid4().hex[:8]
    run = subprocess.run(
        [
            "docker", "run", "-d", "--name", name, "--label", LABEL,
            "--network", "none", *options, IMAGE, "sleep", "600",
        ],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start {IMAGE}: {run.stderr.strip()}")
    try:
        made = _dx(
            name,
            # As on Daytona, whose daemon makes them 755 and 644.
            "umask 022 && useradd -m agent && "
            f"mkdir -p {SESSION} && echo 'echo hi' > {SESSION}/cmd.sh && "
            f"chmod 600 {SESSION}/cmd.sh && echo hi > {SESSION}/output.log && "
            "printf 'original\\n' > /root/input.txt",
        )
        assert made.returncode == 0, made.stderr
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture
def pinned_box():
    yield from _box(MOUNT_CAPS)


@pytest.fixture
def plain_box():
    yield from _box([])


def _can_mount(box: str) -> bool:
    return (
        _dx(box, "mkdir /tmp/m && mount -o bind /tmp/m /tmp/m && umount /tmp/m")
    ).returncode == 0


def _modes(box: str, *paths: str) -> list[str]:
    done = _dx(box, "stat -c '%U %a' " + " ".join(paths))
    assert done.returncode == 0, done.stderr
    return done.stdout.split("\n")[: len(paths)]


def _pinned(box: str) -> bool:
    return _dx(box, "grep -q ' /root/.daytona ' /proc/self/mountinfo").returncode == 0


def _agent_cannot_reach_the_daemon_folder(box: str) -> None:
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


def _agent_works_in_the_workspace(box: str) -> None:
    work = _dx(
        box,
        "echo more >> /root/input.txt && echo new > /root/new.txt && "
        "mkdir /root/results && mv /root/new.txt /root/results/",
        user="agent",
    )
    assert work.returncode == 0, work.stderr


def _root_rewrites_the_agents_file(box: str) -> None:
    # As a root oracle does (`printf > f`, Python's open(f, "w"), openpyxl's
    # save): O_CREAT on an existing file the grant gave the agent.
    assert _modes(box, "/root/input.txt") == ["agent 644"]
    shell = _dx(box, "printf 'edited\\n' > /root/input.txt")
    assert shell.returncode == 0, shell.stderr
    python = _dx(
        box,
        "python3 -c \"open('/root/input.txt', 'w').write('saved\\\\n')\"",
    )
    assert python.returncode == 0, python.stderr
    assert _dx(box, "cat /root/input.txt").stdout == "saved\n"


def test_the_grant_pins_the_daemon_folder_and_leaves_root_the_rewrite(pinned_box):
    """Guards #1232, and the fix of PR #1234's first version (cfcdc841): with
    a /root workspace holding ``.daytona``, the agent gets /root but cannot
    add, change or move anything in the daemon's folder, and root still
    rewrites a file of the agent's in /root, which a sticky /root refuses
    under ``protected_regular``."""
    if not _can_mount(pinned_box):
        pytest.skip("this Docker host refuses a bind mount even with SYS_ADMIN")
    grant = _dx(pinned_box, workspace_grant_command("/root", "agent"))
    assert grant.returncode == 0, grant.stderr

    assert _pinned(pinned_box)
    assert _modes(pinned_box, "/root", "/root/.daytona", "/root/.daytona/sessions") == [
        "agent 700",
        "root 755",
        "root 755",
    ]
    _agent_cannot_reach_the_daemon_folder(pinned_box)
    _agent_works_in_the_workspace(pinned_box)
    _root_rewrites_the_agents_file(pinned_box)


def test_without_the_mount_a_guarded_grant_makes_the_workspace_sticky(plain_box):
    """Guards #1232 where the sandbox refuses the mount and an untrusted
    turn runs as the agent: the workspace becomes root's and sticky, so the
    daemon's folder still cannot be moved."""
    if _can_mount(plain_box):
        pytest.skip("this Docker host allows a bind mount by default")
    grant = _dx(plain_box, workspace_grant_command("/root", "agent"))
    assert grant.returncode == 0, grant.stderr

    assert not _pinned(plain_box)
    assert _modes(plain_box, "/root", "/root/.daytona") == ["root 1777", "root 755"]
    _agent_cannot_reach_the_daemon_folder(plain_box)
    _agent_works_in_the_workspace(plain_box)


def test_without_the_mount_an_unguarded_grant_leaves_root_the_rewrite(plain_box):
    """Guards the fix of PR #1234's first version (cfcdc841): for a root-run
    oracle (``guarded=False``) the grant never makes the workspace sticky, so
    root rewrites the agent's file in /root where the mount is refused too;
    the daemon's folder stays root's."""
    if _can_mount(plain_box):
        pytest.skip("this Docker host allows a bind mount by default")
    grant = _dx(plain_box, workspace_grant_command("/root", "agent", guarded=False))
    assert grant.returncode == 0, grant.stderr

    assert not _pinned(plain_box)
    assert _modes(plain_box, "/root", "/root/.daytona") == ["agent 700", "root 755"]
    for attempt in (
        "echo 'HIDDEN = 1' > /root/.daytona/x.py",
        f"echo 'HIDDEN = 1' > {SESSION}/evil.py",
    ):
        assert _dx(plain_box, attempt, user="agent").returncode != 0, attempt
    _root_rewrites_the_agents_file(plain_box)


def test_without_the_daemon_folder_the_workspace_is_granted_whole(plain_box):
    """Docker sandboxes have no ``.daytona``: the grant is the plain
    ``chown -R`` it was before #1232."""
    assert _dx(plain_box, "rm -rf /root/.daytona").returncode == 0
    grant = _dx(plain_box, workspace_grant_command("/root", "agent"))
    assert grant.returncode == 0, grant.stderr
    assert _modes(plain_box, "/root", "/root/input.txt") == ["agent 700", "agent 644"]
    _root_rewrites_the_agents_file(plain_box)
