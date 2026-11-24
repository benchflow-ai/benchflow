"""the native harness's kill sees the CLI it stops.

Root in a Docker container has no CAP_SYS_PTRACE, so it cannot read the
environment of a process of another uid, and the kill that looked for the
run's marker there found nothing and reported the turn stopped (rc5's
timeout-pass trial: ``stopped: true``, then a model call 18 s later). The
marker is read as the sandbox user; root still sends the signals, and what
is left is recorded.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

import pytest

from benchflow.native_harness.client import NativeCLIClient, alive_script, kill_script
from benchflow.native_harness.harnesses import CLAUDE_CODE

RUN = uuid.uuid4().hex
#: Docker's default capability set, as root in a task container holds it.
DOCKER_DEFAULT_BOUND = (
    "-all,+chown,+dac_override,+fowner,+fsetid,+kill,+setgid,+setuid,+setpcap,"
    "+net_bind_service,+net_raw,+sys_chroot,+mknod,+audit_write,+setfcap"
)


def test_the_marker_is_read_as_the_sandbox_user():
    for script in (
        kill_script(RUN, grace_sec=2, user="agent"),
        alive_script(RUN, user="agent"),
    ):
        reuid = script.index("--reuid")
        assert script.index("setpriv") < reuid < script.index("grep -lF")
    # A root agent's processes are root's own: root reads them as before.
    assert "--reuid" not in kill_script(RUN, grace_sec=2)


class _Recorder:
    def __init__(self, out: str) -> None:
        self.out = out
        self.calls: list[tuple[str, object]] = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.calls.append((command, user))
        return SimpleNamespace(return_code=0, stdout=self.out, stderr="")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("out", "stopped", "left"),
    [
        ("gone\n", True, []),
        ("left 4440 4441\n", False, ["4440", "4441"]),
        ("blind no setpriv or su could read as the sandbox user\n", False, ["unseen"]),
    ],
)
async def test_the_client_kills_as_the_user_and_keeps_what_is_left(
    tmp_path, out, stopped, left
):
    sandbox = _Recorder(out)
    client = NativeCLIClient(
        env=sandbox,
        harness=CLAUDE_CODE,
        agent=CLAUDE_CODE.agent,
        launch_env={},
        sandbox_user="agent",
        cwd="/app",
        rollout_dir=tmp_path / "trial",
    )
    assert await client._kill(RUN, grace_sec=0) is stopped
    command, user = sandbox.calls[0]
    assert user == "root" and "--reuid" in command
    assert client._kill_left == left


def _sudo_ok() -> bool:
    if sys.platform != "linux" or shutil.which("setpriv") is None:
        return False
    probe = subprocess.run(["sudo", "-n", "true"], capture_output=True, check=False)
    return probe.returncode == 0


@pytest.mark.skipif(
    not _sudo_ok(), reason="needs Linux, util-linux setpriv and passwordless sudo"
)
def test_root_without_ptrace_kills_another_users_marked_process():
    run = uuid.uuid4().hex
    target = subprocess.Popen(
        [
            "sudo",
            "-n",
            "setpriv",
            "--reuid=65534",
            "--regid=65534",
            "--clear-groups",
            "env",
            f"RUNTIME_NATIVE_RUN={run}",
            "sleep",
            "120",
        ]
    )
    try:
        time.sleep(0.5)
        as_docker_root = [
            "sudo",
            "-n",
            "setpriv",
            f"--bounding-set={DOCKER_DEFAULT_BOUND}",
            "--",
            "sh",
            "-c",
        ]
        found = subprocess.run(
            [*as_docker_root, alive_script(run, user="nobody")],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert found.stdout.strip().endswith("alive"), found
        killed = subprocess.run(
            [*as_docker_root, kill_script(run, grace_sec=0, user="nobody")],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert killed.stdout.strip().endswith("gone"), killed
        assert target.wait(10) is not None
    finally:
        subprocess.run(
            ["sudo", "-n", "pkill", "-KILL", "-u", "65534", "-f", run],
            capture_output=True,
            check=False,
        )
