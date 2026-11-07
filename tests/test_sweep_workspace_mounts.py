"""Guards the fix at 363d815b across mounted filesystems."""

import os
from pathlib import Path

from benchflow.sandbox.lockdown import _WORKSPACE_OWNER_SCRIPT, _close_workspace_cmd


def test_permission_closure_crosses_workspace_filesystems(tmp_path):
    """Model a task tmpfs with a separate filesystem already in the container."""
    # A Docker container has a separate /dev/shm without mounting anything
    # from its host. Run the close step's own walk with a recorder in place of
    # what it does to each entry, so this test never alters existing mounts.
    external = Path("/dev/shm")
    if (
        not external.is_dir()
        or os.stat(external).st_dev == os.stat(external.parent).st_dev
    ):
        # The walk's structural scope is also checked where no tmpfs exists:
        # it compares devices only to find a folder it moved back into.
        assert "-xdev" not in _close_workspace_cmd(str(tmp_path))
        assert _WORKSPACE_OWNER_SCRIPT.count("st_dev") == 1
        return
    answer = external / ("bf-sweep-answer-" + str(os.getpid()))
    answer.mkdir(mode=0o777)
    walk: dict = {}
    exec(compile(_WORKSPACE_OWNER_SCRIPT, "_workspace_owner.py", "exec"), walk)
    seen = []

    def record(name, entry):
        seen.append(os.path.abspath(name))
        return entry is not None and entry.is_dir(follow_symlinks=False)

    here = os.getcwd()
    try:
        (answer / "answer.txt").write_text("dummy answer")
        (answer / "answer.txt").chmod(0o666)
        # /dev spans the tmpfs in this namespace, just as /app would when its
        # task mounts a tmpfs below it.
        assert _close_workspace_cmd(str(external.parent)).endswith(
            " close /dev); else echo 'closing the workspace needs root Python' "
            ">&2; exit 1; fi"
        )
        walk["walk"](str(external.parent), record)
        assert str(answer / "answer.txt") in seen
    finally:
        os.chdir(here)
        (answer / "answer.txt").unlink()
        answer.rmdir()
