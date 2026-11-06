"""Legacy owners and orphan Git children."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchflow._utils import benchmark_repos as br


@pytest.mark.parametrize("contents", ["live", "", "not-a-pid"])
def test_an_existing_legacy_claim_is_not_overwritten(tmp_path, monkeypatch, contents):
    """Guards 12f002dc: even the creation-before-PID-write interval excludes."""
    monkeypatch.setattr(br, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(br, "_LOCK_TIMEOUT_SEC", 0.2)
    lock = tmp_path / "acme/.tasks.lock"
    lock.parent.mkdir()
    lock.write_text(str(os.getpid()) if contents == "live" else contents)
    before = lock.read_bytes()
    with pytest.raises(TimeoutError), br._repo_cache_lock("acme", "tasks"):
        pytest.fail("entered a live or indeterminate legacy critical section")
    assert lock.read_bytes() == before


_ORPHAN_PARENT = (
    r"""
import os, subprocess, sys, time
from pathlib import Path
from benchflow._utils import benchmark_repos as br
root = Path(sys.argv[1])
br._cache_dir = lambda: root / "cache"
def orphan(cmd, **kwargs):
    # Use the inheritance options of the actual mutating Git call. The child
    # exits on its own; no signal is sent to any process.
    kwargs.pop("check", None)
    child = subprocess.Popen(
        [sys.executable, "-c", """
    + repr("""
import sys, time
from pathlib import Path
root = Path(sys.argv[1])
(root / "ready").touch()
while not (root / "release").exists():
    time.sleep(0.01)
""")
    + r""", str(root)], **kwargs
    )
    while not (root / "ready").exists():
        time.sleep(0.01)
    os._exit(0)
with br._repo_cache_lock("acme", "tasks"):
    subprocess.run = orphan
    action = sys.argv[2]
    if action == "clone":
        br._git_can_sparse = lambda: False
        br._clone_repo_unlocked("acme", "tasks")
    elif action == "worktree":
        br._is_sparse_checkout = lambda _: False
        br._snapshot_repo_root(root, org="acme", repo_name="tasks", resolved_sha="a"*40)
    else:
        br._git_quiet(root, action)
"""
)


@pytest.mark.parametrize(
    "action", ["clone", "fetch", "checkout", "sparse-checkout", "worktree"]
)
def test_git_children_keep_the_lock_after_parent_exit(tmp_path, monkeypatch, action):
    """Guards 12f002dc: an orphan mutator still excludes the next parent."""
    monkeypatch.setattr(br, "_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(br, "_LOCK_TIMEOUT_SEC", 0.2)
    parent = subprocess.Popen(
        [sys.executable, "-c", _ORPHAN_PARENT, str(tmp_path), action],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert parent.wait(timeout=10) == 0
        assert (tmp_path / "ready").exists()
        with pytest.raises(TimeoutError), br._repo_cache_lock("acme", "tasks"):
            pytest.fail("entered while the orphan Git child still owns the cache")
    finally:
        (tmp_path / "release").touch()
        parent.wait(timeout=10)
    monkeypatch.setattr(br, "_LOCK_TIMEOUT_SEC", 5.0)
    with br._repo_cache_lock("acme", "tasks"):
        pass


_LEGACY_PARENT = r"""
import fcntl, os, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1])
cache = root / ".cache/datasets/acme"
cache.mkdir(parents=True)
fd = os.open(cache / ".tasks.lock", os.O_CREAT | os.O_RDWR, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX)
os.write(fd, str(os.getpid()).encode("ascii"))
subprocess.Popen([sys.executable, "-c", '''
import sys, time
from pathlib import Path
root = Path(sys.argv[1])
staging = root / ".cache/datasets/acme/_tasks_clone"
staging.mkdir()
(staging / "child-marker").write_text("still writing")
(root / "ready").touch()
while not (root / "release").exists():
    time.sleep(0.01)
(root / "done").touch()
''', str(root)], close_fds=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os._exit(0)
"""


@pytest.mark.parametrize("existing_checkout", [False, True])
def test_upgrade_does_not_reclaim_a_pre_fix_childs_cache(
    tmp_path, monkeypatch, existing_checkout
):
    """Guards P78-CONFIRM-1 at 78120e49: old children never held the flock."""
    monkeypatch.setattr(br, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(br, "_git_can_sparse", lambda: False)
    parent = subprocess.Popen([sys.executable, "-c", _LEGACY_PARENT, str(tmp_path)])
    assert parent.wait(timeout=10) == 0
    deadline = time.monotonic() + 10
    while not (tmp_path / "ready").exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    legacy = tmp_path / ".cache/datasets/acme"
    staging = legacy / "_tasks_clone/child-marker"
    # Existing checkouts and snapshots belong to that same old protocol.
    checkout = legacy / "tasks"
    if existing_checkout:
        (checkout / ".git").mkdir(parents=True)
        (checkout / "local-edit").write_text("keep checkout")
    snapshot = legacy / "tasks__snapshots" / ("a" * 40)
    (snapshot / ".git").mkdir(parents=True)
    (snapshot / "local-edit").write_text("keep snapshot")

    def fake_clone(cmd, **kwargs):
        assert cmd[:2] == ["git", "clone"]
        (Path(cmd[-1]) / ".git").mkdir(parents=True)

    monkeypatch.setattr(br.subprocess, "run", fake_clone)
    try:
        resolved = br._clone_repo("acme", "tasks")
        assert staging.read_text() == "still writing"
        assert resolved != checkout
        if existing_checkout:
            assert (checkout / "local-edit").read_text() == "keep checkout"
        assert (snapshot / "local-edit").read_text() == "keep snapshot"
        assert not br._cache_path("acme", "tasks__snapshots", "a" * 40).exists()
    finally:
        (tmp_path / "release").touch()
        deadline = time.monotonic() + 10
        while not (tmp_path / "done").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
