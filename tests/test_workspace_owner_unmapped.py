"""Guards #1168 against commit 61b4d227 with a real unmapped inode in a user namespace."""

import shlex
import shutil
import subprocess

import pytest

from benchflow.sandbox import lockdown


@pytest.mark.parametrize("python", [True, False], ids=["python", "shell"])
def test_freeze_copies_unmapped_regular_file(python):
    if shutil.which("docker") is None:
        pytest.skip("Docker required")
    command = lockdown._chown_workspace_cmd("/work")
    script = (
        "set -eu\nmkdir /work\nprintf original > /work/Announce.txt\n"
        "chown 197609:197121 /work/Announce.txt\nchmod 644 /work/Announce.txt\n"
        "unshare --user --map-root-user sh -c "
        + shlex.quote(
            "set -eu\n"
            + (
                "__bf_root_python=/usr/local/bin/python3\n"
                if python
                else "__bf_root_python=\n"
            )
            + command
            + '\n[ "$(stat -c %u /work/Announce.txt)" = 0 ]\n'
            + '[ "$(cat /work/Announce.txt)" = original ]\n'
            + '[ "$(stat -c %a /work/Announce.txt)" = 644 ]\n'
        )
    )
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cap-add",
            "SYS_ADMIN",
            "--security-opt",
            "seccomp=unconfined",
            "python:3.12-slim",
            "sh",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
