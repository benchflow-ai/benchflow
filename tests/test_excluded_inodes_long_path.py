"""Verifier hardening reads excluded identities past PATH_MAX (#1233).

``_excluded_inodes`` read each identity in the excluded trees (``/tmp`` among
them) with ``find ... -exec stat -c %d:%i {} +``, which hands ``stat`` the
full path. An agent that left a path longer than PATH_MAX (4096 bytes) in
``/tmp`` made ``stat`` fail with ENAMETOOLONG, the hardening raise, and the
trial end unscored instead of at its score. GNU find now prints each identity
itself; a find without ``-printf`` (BusyBox) still cannot reach such a path,
and the trial gets a warning instead of failing the hardening.

Real Docker, BenchFlow's own exec route and pre-agent inspector. Skipped
without a Docker daemon.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid

import pytest

from benchflow._utils.trial_warnings import recording_trial_warnings
from benchflow.sandbox import lockdown
from tests.test_c8_protected_runtime_container import _sandbox

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="docker required"
)

LABEL = os.environ.get("BENCHFLOW_TEST_CONTAINER_LABEL", "PT=1")
IMAGES = {"gnu-find": "python:3.12-slim", "busybox-find": "alpine:3.20"}
# 30 folders of 201-byte names: the file's path is about 6 KB. Made one
# relative step at a time, as an agent's program can; prints its identity.
DEEP = (
    "cd /tmp && mkdir deepp && cd -P deepp && i=0; "
    "while [ $i -lt 30 ]; do d=d$(printf %0200d $i); "
    "mkdir $d && cd -P $d || exit 1; i=$((i+1)); done; "
    "echo x > far.txt && stat -c %d:%i far.txt"
)


def _docker_ready() -> bool:
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except Exception:
        return False
    return probe.returncode == 0


def _dx(name: str, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", "-u", "0", name, "/bin/sh", "-c", command],
        capture_output=True,
        text=True,
    )


@pytest.fixture(params=sorted(IMAGES), ids=sorted(IMAGES))
def box(request):
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    image = IMAGES[request.param]
    name = "bf-long-path-" + uuid.uuid4().hex[:8]
    run = subprocess.run(
        [
            "docker", "run", "-d", "--name", name, "--label", LABEL,
            "--network", "none", image, "sleep", "600",
        ],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start {image}: {run.stderr.strip()}")
    try:
        yield request.param, name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.mark.asyncio
async def test_a_path_longer_than_path_max_does_not_stop_the_hardening(box):
    """Guards #1233: a file at a path over 4096 bytes in /tmp no longer
    raises "capturing excluded inode identities". GNU find returns its
    identity with the rest; BusyBox's find cannot reach it, and the
    trial gets a warning instead."""
    kind, name = box
    sandbox = _sandbox(name)
    await lockdown._prepare_bash_env_inspector(sandbox, blocked=("/tmp",))
    made = _dx(
        name, DEEP + " && echo near > /tmp/near.txt && stat -c %d:%i /tmp/near.txt"
    )
    assert made.returncode == 0, made.stderr
    far, near = (tuple(map(int, line.split(":"))) for line in made.stdout.split())

    with recording_trial_warnings() as warnings:
        identities = await lockdown._excluded_inodes(sandbox, ("/tmp",))
    unread = [text for text in warnings if "inode identity" in text]

    assert near in identities
    if kind == "gnu-find":
        assert far in identities
        assert not unread
    else:
        assert unread
