"""Guard P78-CONFIRM-2 at 78120e49 with raw tree-content comparisons."""

import os
import subprocess

import pytest

from benchflow._utils.dataset_registry import _task_matches_commit


@pytest.mark.parametrize(
    "change", [None, "bytes", "mode", "link", "missing", "extra", "fifo"]
)
def test_tree_integrity_preserves_honest_names_modes_and_links(tmp_path, change):
    """Guards 78120e49: compare root tasks, binary blobs and symlink bytes."""
    root = tmp_path / "task"
    root.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git("init", "-q")
    script = root / "tést \t\n.sh"
    script.write_bytes(b"#!/bin/sh\n\x00binary\xff\n")
    script.chmod(0o755)
    link = root / "link"
    link.symlink_to(script.name)
    git("add", ".")
    git(
        "-c",
        "user.name=BenchFlow Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-q",
        "-m",
        "fixture",
    )
    commit = git("rev-parse", "HEAD")
    assert _task_matches_commit(root, commit)
    if change == "bytes":
        script.write_bytes(b"changed")
    elif change == "mode":
        script.chmod(0o644)
    elif change == "link":
        link.unlink()
        link.symlink_to("different")
    elif change == "missing":
        script.unlink()
    elif change == "extra":
        (root / "extra").write_text("untracked")
    elif change == "fifo":
        os.mkfifo(root / "pipe")
    assert _task_matches_commit(root, commit) is (change is None)
