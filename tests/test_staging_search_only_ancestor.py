"""A task folder below a search-only ancestor is copied (benchflow#1237).

``copy_tree`` pins every component of the copied folder's real path from
``/`` with a no-follow open. A folder above the copied one is only passed
through, so it needs search permission alone, as a root-owned ``0711`` data
mount gives everyone else; the copied folder and every folder in it are
listed, so they still need read permission. Root passes every permission
check, so these tests need another user.
"""

from __future__ import annotations

import os

import pytest

from benchflow.task import staging

pytestmark = pytest.mark.skipif(
    os.geteuid() == 0, reason="root passes every permission check"
)

_REAL_OPEN = os.open


@pytest.fixture
def gated(tmp_path):
    """``gate/task`` with ``gate`` searchable but not readable (``0311``)."""
    gate = tmp_path / "gate"
    task = gate / "task"
    (task / "sub").mkdir(parents=True)
    (task / "a.txt").write_text("x")
    (task / "sub" / "b.txt").write_text("y")
    gate.chmod(0o311)
    yield task
    # So tmp_path can be removed, whatever a test did to them.
    for folder in (tmp_path / "gate.real", gate, task, task / "sub"):
        if folder.is_dir() and not folder.is_symlink():
            folder.chmod(0o755)


def test_a_task_below_a_search_only_ancestor_is_copied(gated, tmp_path):
    """The reported case: refused as 'changed while being copied' before."""
    staging.copy_tree(gated, tmp_path / "copy")
    assert (tmp_path / "copy" / "a.txt").read_text() == "x"
    assert (tmp_path / "copy" / "sub" / "b.txt").read_text() == "y"


def test_stage_environment_below_a_search_only_ancestor(tmp_path, monkeypatch):
    """What every backend's launch, ``Rollout.setup`` and the DinD preflight
    go through."""
    environment = tmp_path / "gate" / "task" / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM example.invalid/task\n")
    (tmp_path / "gate").chmod(0o311)
    stages = tmp_path / "stages"
    stages.mkdir()
    monkeypatch.setattr(staging.tempfile, "tempdir", str(stages))
    try:
        staged = staging.stage_environment(environment)
        try:
            dockerfile = staged.environment_dir / "Dockerfile"
            assert dockerfile.read_text() == "FROM example.invalid/task\n"
        finally:
            staged.remove()
    finally:
        (tmp_path / "gate").chmod(0o755)


@pytest.mark.skipif(not hasattr(os, "O_PATH"), reason="Linux's O_PATH")
def test_only_the_ancestors_are_opened_for_search(gated, tmp_path, monkeypatch):
    opened = []

    def record(path, flags, mode=0o777, *, dir_fd=None):
        if flags & os.O_DIRECTORY:
            opened.append((path, bool(flags & os.O_PATH)))
        return _REAL_OPEN(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(staging.os, "open", record)
    staging.copy_tree(gated, tmp_path / "copy")
    parts = os.path.realpath(gated).split(os.sep)[1:]
    # /, then each ancestor with O_PATH; the task and its folder to be read.
    assert opened == [
        (os.sep, True),
        *((part, True) for part in parts[:-1]),
        (parts[-1], False),
        ("sub", False),
    ]


@pytest.mark.parametrize("folder", ["task", "sub"])
def test_a_copied_folder_still_needs_read_permission(gated, tmp_path, folder):
    (gated if folder == "task" else gated / "sub").chmod(0o311)
    with pytest.raises(staging.StagingRefused, match="changed while being copied"):
        staging.copy_tree(gated, tmp_path / "copy")


@pytest.mark.parametrize("change", ["link", "gone"])
def test_an_ancestor_that_changes_when_it_is_pinned_is_refused(
    gated, tmp_path, monkeypatch, change
):
    """The race check is unchanged: the ancestor swapped for a link to
    itself, or moved away, between the path's resolution and its open."""
    gate, moved = tmp_path / "gate", tmp_path / "gate.real"

    def swap(path, flags, mode=0o777, *, dir_fd=None):
        if path == "gate" and dir_fd is not None and not moved.exists():
            gate.rename(moved)
            if change == "link":
                gate.symlink_to(moved)
        return _REAL_OPEN(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(staging.os, "open", swap)
    with pytest.raises(
        staging.StagingRefused, match="changed while being copied"
    ) as caught:
        staging.copy_tree(gated, tmp_path / "copy")
    assert caught.value.transient
    assert moved.is_dir()
    assert not (tmp_path / "copy" / "a.txt").exists()


def test_without_o_path_an_ancestor_is_opened_to_be_read(monkeypatch):
    """macOS has no O_PATH: an ancestor is opened as before."""
    flags = []
    monkeypatch.delattr(staging.os, "O_PATH", raising=False)
    monkeypatch.setattr(
        staging.os,
        "open",
        lambda path, flag, mode=0o777, *, dir_fd=None: flags.append(flag) or 7,
    )
    assert staging._open_ancestor("gate", 3) == 7
    assert flags == [os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC]
