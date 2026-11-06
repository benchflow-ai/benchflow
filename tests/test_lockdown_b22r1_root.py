"""Guard CONFIRM-lockdown-4 at 8c11e9a4: freeze the physical workspace root."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult


@pytest.mark.parametrize("leaf", ["work", "work ' [1]\n"])
@pytest.mark.parametrize(
    "physical_leaf", ["project", "project\n", "project ' [*?]\\name\n\n"]
)
@pytest.mark.parametrize("operation", ["permissions", "ownership"])
async def test_symlink_workspace_root_is_frozen_without_following_nested_links(
    tmp_path, monkeypatch, leaf, physical_leaf, operation
):
    physical = tmp_path / physical_leaf
    physical.mkdir()
    answer = physical / "answer.txt"
    answer.write_text("correct\n")
    physical.chmod(0o777)
    answer.chmod(0o666)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "marker").write_text("dummy\n")
    outside.chmod(0o777)
    (outside / "marker").chmod(0o666)
    (physical / "escape").symlink_to(outside)
    workspace = tmp_path / leaf
    workspace.symlink_to(physical)
    # The steps run as root; here the walk's owner is the test user, so it
    # runs for real without root (_workspace_owner.py UID and GID).
    me = f"UID, GID = {os.getuid()}, {os.getgid()}"
    token = lockdown._root_python.set(sys.executable)
    try:
        if operation == "permissions":
            command = lockdown._prep_on_path(
                lockdown._close_workspace_cmd(str(workspace))
            )
        else:
            commands = []

            async def execute(command, **kwargs):
                if "-exec chown -h root:root" in command:
                    commands.append(command)
                return ExecResult(stdout="", stderr="", return_code=0)

            await lockdown._freeze_workspace(
                SimpleNamespace(exec=execute), str(workspace)
            )
            # Record the folder the walk starts in, the physical root.
            command = commands[0].replace(
                "    os.chdir(workspace)\n",
                "    os.chdir(workspace)\n    print(os.getcwd())\n",
            )
    finally:
        lockdown._root_python.reset(token)
    assert "UID = GID = 0" in command
    command = command.replace("UID = GID = 0", me)
    result = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    if operation == "permissions":
        assert not os.stat(answer).st_mode & 0o022
        assert not os.stat(physical).st_mode & 0o022
    else:
        assert result.stdout == str(physical) + "\n"
    assert (outside / "marker").read_text() == "dummy\n"
    assert os.stat(outside).st_mode & 0o777 == 0o777
    assert os.stat(outside / "marker").st_mode & 0o777 == 0o666
    assert (physical / "escape").is_symlink()
