"""The workspace walk skips an entry that vanishes mid-walk (benchflow#1238).

On Daytona with workspace ``/root`` the freeze crosses
``/root/.daytona/sessions``, where the daemon deletes finished session
folders on its own schedule: an entry listed and then removed stopped the
walk (``workspace freeze: [Errno 2] No such file or directory``) and the
trial ended unscored. These run the step's own script as lockdown sends it,
as its ``__main__``, with the owner it gives entries set to the test's user
(the step runs as root; test_lockdown_b22r1_root does the same), and stand
in for that other process by wrapping ``os.scandir``, ``os.open``,
``os.chown`` or ``os.rename`` in the script's ``os``.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
import types

import pytest

from benchflow.sandbox.lockdown import _WORKSPACE_OWNER_SCRIPT

_REAL_SCANDIR, _REAL_OPEN, _REAL_CHOWN = os.scandir, os.open, os.chown
_REAL_RENAME = os.rename


def _run(monkeypatch, workspace, mode="freeze", *, before=None, after=None, entry=None):
    """The script's exit message for *mode* on *workspace*, or None when it
    ran to its end. *before* and *after* run with a folder's entry names
    just before and just after it is listed, in that folder; *entry* maps
    each listed entry to the one the script is given."""
    me = f"UID, GID = {os.getuid()}, {os.getgid()}"
    source = _WORKSPACE_OWNER_SCRIPT.replace("UID = GID = 0", me, 1)
    assert source != _WORKSPACE_OWNER_SCRIPT

    def scandir(path="."):
        if before is not None:
            before(sorted(os.listdir(path)))
        with _REAL_SCANDIR(path) as it:
            entries = list(it)
        if after is not None:
            after(sorted(e.name for e in entries))
        if entry is not None:
            entries = [entry(e) for e in entries]
        return contextlib.nullcontext(entries)

    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(sys, "argv", ["_workspace_owner.py", mode, str(workspace)])
    here = os.getcwd()
    try:
        exec(compile(source, "_workspace_owner.py", "exec"), {"__name__": "__main__"})
    except SystemExit as exc:
        return str(exc.code)
    finally:
        os.chdir(here)
    return None


@pytest.fixture(autouse=True)
def pauses(monkeypatch):
    """The pauses between walks, taken at once (on this thread only)."""
    taken, here, sleep = [], threading.get_ident(), time.sleep

    def pause(seconds):
        if threading.get_ident() != here:
            return sleep(seconds)
        taken.append(seconds)

    monkeypatch.setattr(time, "sleep", pause)
    return taken


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "root"
    session = root / ".daytona" / "sessions" / "8f2c41"
    session.mkdir(parents=True)
    (session / "output.log").write_text("done\n")
    (session / "exit_code").write_text("0\n")
    (root / "answer.txt").write_text("42\n")
    (tmp_path / "moved").mkdir()
    return root


@pytest.mark.parametrize("mode", ["freeze", "close"])
def test_an_entry_unlinked_after_its_folder_is_listed_is_skipped(
    monkeypatch, workspace, mode
):
    """The reported repro: sessions/8f2c41/output.log goes after 8f2c41 is
    listed, as the Daytona daemon removes a finished session's files."""

    def unlink(names):
        if "output.log" in names:  # in 8f2c41
            os.unlink("output.log")

    assert _run(monkeypatch, workspace, mode, after=unlink) is None
    session = workspace / ".daytona" / "sessions" / "8f2c41"
    assert not (session / "output.log").exists()
    assert (session / "exit_code").read_text() == "0\n"
    assert (workspace / "answer.txt").read_text() == "42\n"


def test_a_folder_removed_before_it_is_entered_is_skipped(monkeypatch, workspace):
    def remove(path, flags, mode=0o777, *, dir_fd=None):
        if path == "8f2c41" and os.path.isdir(path):
            for name in os.listdir(path):
                os.unlink(os.path.join(path, name))
            os.rmdir(path)
        return _REAL_OPEN(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", remove)
    assert _run(monkeypatch, workspace) is None
    assert not (workspace / ".daytona" / "sessions" / "8f2c41").exists()
    assert (workspace / "answer.txt").read_text() == "42\n"


def test_an_entry_moved_into_a_walked_folder_is_settled_by_the_next_walk(
    monkeypatch, workspace, pauses
):
    """A name gone is also what a rename away looks like: the tree is walked
    again, so the entry is given to the owner where it went."""
    chowned = []

    def chown(path, uid, gid, *, dir_fd=None, follow_symlinks=True):
        chowned.append(os.path.realpath(path))
        return _REAL_CHOWN(path, uid, gid, follow_symlinks=follow_symlinks)

    def move(names):
        if "exit_code" in names and "output.log" in names:  # in 8f2c41
            os.rename("output.log", os.path.join(workspace, "output.log"))

    monkeypatch.setattr(os, "chown", chown)
    assert _run(monkeypatch, workspace, after=move) is None
    assert str(workspace / "output.log") in chowned
    assert pauses == [0.5]  # one more walk, after a pause


def test_entries_that_vanish_in_every_walk_fail_the_step(
    monkeypatch, workspace, pauses
):
    """Bounded: five walks, 7.5 s of pauses in all, inside the step's budget."""

    def make(names):
        if "answer.txt" in names:
            open("churn", "w").close()

    def unlink(names):
        if "churn" in names:
            os.unlink("churn")

    message = _run(monkeypatch, workspace, before=make, after=unlink)
    assert message == "workspace freeze: entries vanished during each of 5 walks"
    assert pauses == [0.5, 1.0, 2.0, 4.0]


def test_a_burst_of_removals_that_ends_passes_after_paced_walks(
    monkeypatch, workspace, pauses
):
    """The daemon removes a batch of session folders: the walks that overlap
    it lose entries, and the first walk after it ends passes the step."""
    walks = []

    def make(names):
        if "answer.txt" in names:  # the workspace root: one walk begins
            walks.append(names)
            if len(walks) <= 2:
                open("churn", "w").close()

    def unlink(names):
        if "churn" in names:
            os.unlink("churn")

    assert _run(monkeypatch, workspace, before=make, after=unlink) is None
    assert len(walks) == 3
    assert pauses == [0.5, 1.0]


def test_a_folder_moved_while_it_is_walked_still_fails(monkeypatch, workspace):
    def move(names):
        if "exit_code" in names:  # in 8f2c41: move the folder itself
            os.rename(
                os.path.join(workspace, ".daytona", "sessions", "8f2c41"),
                os.path.join(workspace.parent, "moved", "8f2c41"),
            )

    message = _run(monkeypatch, workspace, after=move)
    assert message == "workspace freeze: a folder moved during the step"


def test_a_folder_replaced_by_a_link_before_it_is_entered_still_fails(
    monkeypatch, workspace
):
    def swap(path, flags, mode=0o777, *, dir_fd=None):
        if path == "8f2c41" and not os.path.islink(path):
            os.rename(path, os.path.join(workspace.parent, "moved", "8f2c41"))
            os.symlink(os.path.join(workspace.parent, "moved", "8f2c41"), path)
        return _REAL_OPEN(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap)
    message = _run(monkeypatch, workspace)
    # O_NOFOLLOW with O_DIRECTORY on a link: ENOTDIR, never followed.
    assert message == "workspace freeze: [Errno 20] Not a directory: '8f2c41'"


def test_a_name_that_holds_another_entry_when_looked_at_again_still_fails(
    monkeypatch, workspace
):
    """Only an entry whose name is empty now counts as gone."""

    def replaced(path, uid, gid, *, dir_fd=None, follow_symlinks=True):
        if path == "output.log":
            os.unlink(path)
            os.symlink("/", path)
            raise FileNotFoundError(2, "No such file or directory", path)
        return _REAL_CHOWN(path, uid, gid, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "chown", replaced)
    message = _run(monkeypatch, workspace)
    assert message == (
        "workspace freeze: [Errno 2] No such file or directory: 'output.log'"
    )


@pytest.mark.parametrize("copy_left", [False, True], ids=["copy-gone", "copy-left"])
def test_a_copy_that_cannot_take_the_name_still_fails(
    monkeypatch, workspace, copy_left
):
    """Only an error that names the listed entry counts as it being gone.
    Closing replaces a file that is not root's with a copy: once the file's
    name is removed, a FileNotFoundError from renaming the copy into place
    names the copy, so the step fails instead of passing with the file
    lost."""

    def foreign(listed):
        if listed.name != "answer.txt":
            return listed
        st = listed.stat(follow_symlinks=False)
        seen = {k: getattr(st, k) for k in dir(st) if k.startswith("st_")}
        seen["st_uid"] = os.getuid() + 1  # another user's, as close sees it
        seen = types.SimpleNamespace(**seen)
        return types.SimpleNamespace(
            name=listed.name, stat=lambda follow_symlinks=True: seen
        )

    def rename(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        if dst == "answer.txt":
            if copy_left:
                raise FileNotFoundError(2, "No such file or directory", src, None, dst)
            os.unlink(src)  # the copy is gone
        return _REAL_RENAME(src, dst)

    monkeypatch.setattr(os, "rename", rename)
    message = _run(monkeypatch, workspace, "close", entry=foreign)
    assert message is not None
    assert message.startswith(
        "workspace close: [Errno 2] No such file or directory: '.bf"
    ), message
    assert not (workspace / "answer.txt").exists()
    assert not [name for name in os.listdir(workspace) if name.startswith(".bf")]


def test_the_script_keeps_one_walk_without_a_vanished_entry(
    monkeypatch, workspace, pauses
):
    walks = []

    def count(names):
        if "answer.txt" in names:
            walks.append(names)

    assert _run(monkeypatch, workspace, after=count) is None
    assert len(walks) == 1
    assert pauses == []
