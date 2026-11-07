"""The agent user gets a uid and gid that own nothing in the image (PLAN P5).

``useradd -m agent`` takes the first uid /etc/passwd does not name. An image
can hold files of a uid no user has: autocodebench's ``uv`` and ``uvx`` in
``/root/.local/bin`` are uid 1001's (left by a ``docker commit``), and with
its ``sandbox`` user at 1000 the agent user became 1001 and owned them, so it
could rewrite a program the verifier runs (Rust P1.2c's finding 2). Setup now
picks the first uid from 1000 that no passwd or group entry names and no
file's owner or group is, and uses it as the gid too.

The fixture images (built on the machine's Docker; skipped without one) hold
files of uids and gids 1000 and 1001 that no user or group names: a Debian
image (useradd) and an Alpine one (busybox adduser, busybox find).
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from types import SimpleNamespace

import pytest

from benchflow.sandbox.lockdown import setup_sandbox_user
from tests.test_c8_protected_runtime_container import _sandbox

DOCKERFILE = """\
FROM {base}
RUN mkdir -p /root/.local/bin /opt/data /app \\
 && printf '#!/bin/sh\\necho uv\\n' > /root/.local/bin/uv \\
 && chmod 755 /root/.local/bin/uv && chown 1001:1001 /root/.local/bin/uv \\
 && touch /opt/data/a /opt/data/b && chown 1000:1000 /opt/data/a \\
 && chown 0:1002 /opt/data/b
"""

BASES = {"debian-useradd": "python:3.12-slim", "alpine-adduser": "alpine:3.20"}


def _docker_ready() -> bool:
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except Exception:
        return False
    return probe.returncode == 0


def _dx(name: str, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", "-u", "root", name, "sh", "-c", command],
        capture_output=True,
        text=True,
        timeout=300,
    )


class ContainerEnv:
    # PT's PATH lock (setup_sandbox_user) runs only on a sandbox that starts
    # root's shell by absolute path; this one does (/bin/sh below).
    supports_system_shell = True

    def __init__(self, name: str) -> None:
        self.name = name

    async def exec(self, command: str, user: str | None = None, **kwargs):
        done = subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                user or "root",
                self.name,
                "/bin/sh",
                "-c",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return SimpleNamespace(
            return_code=done.returncode, stdout=done.stdout, stderr=done.stderr
        )


@pytest.fixture(params=sorted(BASES), ids=sorted(BASES))
def box(request):
    if shutil.which("docker") is None or not _docker_ready():
        pytest.skip("docker required")
    tag = f"bf-v3-uid-{uuid.uuid4().hex[:8]}"
    build = subprocess.run(
        ["docker", "build", "-q", "-t", tag, "-"],
        input=DOCKERFILE.format(base=BASES[request.param]),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if build.returncode != 0:
        pytest.skip(f"could not build the fixture image: {build.stderr[-300:]}")
    name = f"{tag}-c"
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--network",
            "none",
            "--name",
            name,
            tag,
            "sleep",
            "600",
        ],
        capture_output=True,
        text=True,
    )
    try:
        if run.returncode != 0:
            pytest.skip(f"could not start the container: {run.stderr[:200]}")
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


@pytest.mark.asyncio
async def test_the_agent_user_owns_no_file_of_the_image(box):
    """Guards PLAN P5: uid 1001's uv stays out of the agent user's hands."""
    await setup_sandbox_user(_sandbox(box), "agent", "/app", guard_daemon=False)

    ids = _dx(box, "id -u agent; id -g agent")
    assert ids.returncode == 0, ids.stderr
    uid, gid = ids.stdout.split()
    assert uid not in ("1000", "1001", "1002")
    assert gid not in ("1000", "1001", "1002")
    # Outside its home and the workspace, nothing is the agent user's.
    owned = _dx(
        box,
        f"find / -xdev \\( -user {uid} -o -group {gid} \\) "
        "! -path '/home/agent' ! -path '/home/agent/*' "
        "! -path '/app' ! -path '/app/*' -print",
    )
    assert owned.stdout == ""
    assert _dx(box, "stat -c %u /root/.local/bin/uv").stdout.strip() == "1001"


@pytest.mark.asyncio
async def test_an_agent_user_the_image_already_has_is_kept(box):
    """A task that ships its own agent user keeps that user's ids."""
    made = _dx(
        box,
        "adduser -D -u 1001 agent 2>/dev/null || useradd -m -u 1001 agent",
    )
    assert made.returncode == 0, made.stderr

    await setup_sandbox_user(_sandbox(box), "agent", "/app", guard_daemon=False)

    assert _dx(box, "id -u agent").stdout.strip() == "1001"
