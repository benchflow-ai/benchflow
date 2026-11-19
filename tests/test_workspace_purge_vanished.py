"""The workspace's symlink purge skips an entry that vanishes mid-walk
(benchflow#1238).

purge_external_symlinks is the freeze batch's first step, run as root on the
image's root python3 when there is no kit. On Daytona with workspace ``/root``
its walk crosses ``/root/.daytona/sessions``, where the daemon deletes
finished session folders on its own schedule. The walk was ``Path.rglob``,
which on CPython 3.9-3.11 enters each listed folder catching PermissionError
only: a session folder removed after its parent was listed ended the step with
FileNotFoundError and the trial unscored. A link removed between being judged
and being read or removed failed the step on every Python.

These run the step's command as lockdown renders it, through ``sh``, with its
own flags, on each CPython 3.9-3.13 on ``PATH`` and on the test's own. The
interpreter the command is handed is a wrapper that runs the step's script,
unchanged, after a few lines that stand in for the daemon: they remove an
entry at the moment the walk reaches it.

Only a vanished entry is skipped. A folder the walk cannot list, or an entry
it cannot check, for any other reason still fails the step, since the links
to outside in it would be left for the verifier: a folder chain deeper than
PATH_MAX, which any agent can make, is the real case.
"""

from __future__ import annotations

import errno
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow._root_python import FLAGS
from benchflow.sandbox import lockdown

_PYTHONS = [
    *(pytest.param(v, id=f"py{v}") for v in ("3.9", "3.10", "3.11", "3.12", "3.13")),
    pytest.param(None, id="this-python"),
]

# pathlib (and 3.13's glob, which it imports) first, so that each keeps the
# functions it would have had: 3.9 and 3.10 bind os.stat and others at import.
_HEAD = """\
import os as _os, sys as _sys, pathlib as _pathlib, glob as _glob
_sys.stderr.write('preamble %d.%d\\n' % _sys.version_info[:2])
"""

# Remove sessions/s1 at the moment the walk enters it (it was listed in
# sessions), through whichever scandir the walk lists with.
_FOLDER_GONE = """\
_real_scandir = _os.scandir
def _scandir(path='.'):
    if not isinstance(path, int):
        name = _os.fspath(path)
        if _os.path.basename(name) == 's1' and _os.path.isdir(name):
            for entry in _os.listdir(name):
                _os.unlink(_os.path.join(name, entry))
            _os.rmdir(name)
    return _real_scandir(path)
_os.scandir = _scandir
if hasattr(_pathlib, '_NormalAccessor'):  # 3.9, 3.10: rglob lists through it
    _pathlib._NormalAccessor.scandir = staticmethod(_scandir)
if hasattr(_glob, '_StringGlobber'):  # 3.13: rglob lists through it
    _glob._StringGlobber.scandir = staticmethod(_scandir)
"""

# Remove the link named 'doomed' once it is resolved, before it is removed.
_LINK_GONE_AFTER_REALPATH = """\
_real_realpath = _os.path.realpath
def _realpath(path, *args, **kwargs):
    resolved = _real_realpath(path, *args, **kwargs)
    if _os.path.basename(str(path)) == 'doomed' and _os.path.islink(path):
        _os.unlink(path)
    return resolved
_os.path.realpath = _realpath
"""

# Remove the looping link named 'doomed' once its stat fails, before it is
# removed.
_LINK_GONE_AFTER_LOOP = """\
_real_stat = _os.stat
def _stat(path, *args, **kwargs):
    try:
        return _real_stat(path, *args, **kwargs)
    except OSError:
        if _os.path.basename(str(path)) == 'doomed' and _os.path.islink(path):
            _os.unlink(path)
        raise
_os.stat = _stat
"""

# Remove the recorded link named 'recorded' in the workspace before its text
# is read.
_RECORDED_LINK_GONE = """\
_real_readlink = _os.readlink
def _readlink(path, *args, **kwargs):
    path_str = str(path)
    if path_str.startswith(WORKSPACE) and _os.path.basename(path_str) == 'recorded':
        _os.unlink(path)
    return _real_readlink(path, *args, **kwargs)
_os.readlink = _readlink
"""


# Fail the listing of the folder named 'locked' as an unreadable folder's
# would.
_FOLDER_UNLISTABLE = """\
_real_scandir = _os.scandir
def _scandir(path='.'):
    if not isinstance(path, int) and _os.path.basename(_os.fspath(path)) == 'locked':
        raise PermissionError(13, 'Permission denied', _os.fspath(path))
    return _real_scandir(path)
_os.scandir = _scandir
"""

# Fail the lstat of the entry named 'unreadable'.
_ENTRY_UNCHECKABLE = """\
_real_lstat = _os.lstat
def _lstat(path, *args, **kwargs):
    if _os.path.basename(_os.fspath(path)) == 'unreadable':
        raise PermissionError(13, 'Permission denied', _os.fspath(path))
    return _real_lstat(path, *args, **kwargs)
_os.lstat = _lstat
"""

# Remove the entry named 'fleeting' just before its lstat.
_ENTRY_GONE = """\
_real_lstat = _os.lstat
def _lstat(path, *args, **kwargs):
    if _os.path.basename(_os.fspath(path)) == 'fleeting':
        try:
            _os.unlink(path)
        except FileNotFoundError:
            pass
    return _real_lstat(path, *args, **kwargs)
_os.lstat = _lstat
"""


def _interpreter(version: str | None) -> str:
    if version is None:
        return sys.executable
    found = shutil.which(f"python{version}")
    if found is None:
        pytest.skip(f"no python{version} on PATH")
    done = subprocess.run(
        [found, "-I", "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        capture_output=True,
        text=True,
    )
    if done.stdout.strip() != version:
        pytest.skip(f"python{version} on PATH is {done.stdout.strip()!r}")
    return found


def _purge(tmp_path: Path, python: str, workspace: Path, preamble: str = ""):
    """Run the step's command on *workspace* with *preamble* ahead of its
    script; return the finished process."""
    baseline = tmp_path / "baseline"
    if not baseline.exists():
        baseline.mkdir(mode=0o700)
    first = tmp_path / "preamble.py"
    first.write_text(
        _HEAD + preamble.replace("WORKSPACE", repr(str(workspace) + os.sep))
    )
    flags = " ".join(FLAGS)
    wrapper = tmp_path / "python3"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'[ "$1 $2 $3 $4" = "{flags} -c" ] || exit 97\n'
        "script=$5\n"
        "shift 5\n"
        f'exec {shlex.quote(python)} {flags} -c "$(cat {shlex.quote(str(first))})\n'
        '$script" "$@"\n'
    )
    wrapper.chmod(0o755)
    command = lockdown._purge_external_symlinks_cmd(str(workspace))
    assert "/testbed_verify" in command
    command = command.replace("/testbed_verify", str(baseline))
    done = subprocess.run(
        ["sh", "-c", f"__bf_root_python={shlex.quote(str(wrapper))}; {command}"],
        capture_output=True,
        text=True,
    )
    assert done.stderr.startswith("preamble "), done.stderr
    return done


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace with Daytona session folders, an external link in a
    later session and one at the top."""
    root = tmp_path / "workspace"
    sessions = root / ".daytona" / "sessions"
    for name in ("s1", "s2"):
        (sessions / name).mkdir(parents=True)
        (sessions / name / "output.log").write_text("done\n")
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("outside\n")
    (sessions / "s2" / "ext").symlink_to(elsewhere)
    (root / "answer.txt").write_text("42\n")
    (root / "ext").symlink_to(elsewhere)
    return root


@pytest.mark.parametrize("version", _PYTHONS)
def test_a_session_folder_removed_before_it_is_entered_is_skipped(
    tmp_path: Path, workspace: Path, version
):
    """The reported case: s1 goes after sessions is listed, as the walk
    enters it. On 3.9-3.11 rglob raised FileNotFoundError: '.../sessions/s1';
    from 3.12 it skipped s1, as os.walk does on every Python."""
    python = _interpreter(version)
    done = _purge(tmp_path, python, workspace, _FOLDER_GONE)
    assert done.returncode == 0, done.stderr
    sessions = workspace / ".daytona" / "sessions"
    assert not (sessions / "s1").exists()
    # The walk went on past it: the later links outside were still removed.
    assert not (sessions / "s2" / "ext").is_symlink()
    assert not (workspace / "ext").is_symlink()
    assert (sessions / "s2" / "output.log").read_text() == "done\n"
    assert (workspace / "answer.txt").read_text() == "42\n"


@pytest.mark.parametrize("version", _PYTHONS)
@pytest.mark.parametrize(
    ("target", "preamble"),
    [
        pytest.param("../elsewhere.txt", _LINK_GONE_AFTER_REALPATH, id="outside"),
        pytest.param("doomed", _LINK_GONE_AFTER_LOOP, id="loop"),
    ],
)
def test_a_link_removed_before_the_purge_removes_it_is_skipped(
    tmp_path: Path, workspace: Path, version, target, preamble
):
    """A link judged for removal (it resolves outside, or loops) that is
    gone before it is removed has nothing left to purge."""
    python = _interpreter(version)
    (workspace / "doomed").symlink_to(target)
    done = _purge(tmp_path, python, workspace, preamble)
    assert done.returncode == 0, done.stderr
    assert not os.path.lexists(workspace / "doomed")
    assert not (workspace / "ext").is_symlink()


@pytest.mark.parametrize("version", _PYTHONS)
def test_a_recorded_link_removed_before_its_text_is_read_is_skipped(
    tmp_path: Path, workspace: Path, version
):
    python = _interpreter(version)
    (workspace / "recorded").symlink_to("answer.txt")
    baseline = tmp_path / "baseline"
    baseline.mkdir(mode=0o700)
    (baseline / "answer.txt").write_text("42\n")
    (baseline / "recorded").symlink_to("answer.txt")
    done = _purge(tmp_path, python, workspace, _RECORDED_LINK_GONE)
    assert done.returncode == 0, done.stderr
    assert not os.path.lexists(workspace / "recorded")
    assert not (workspace / "ext").is_symlink()


@pytest.mark.parametrize("version", _PYTHONS)
def test_links_are_judged_as_before_and_none_is_followed(
    tmp_path: Path, workspace: Path, version
):
    """Nothing vanishes: a link outside goes, a link inside stays, and a
    folder link is removed or kept without being entered."""
    python = _interpreter(version)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "inner").symlink_to(tmp_path / "elsewhere.txt")
    (workspace / "sub").mkdir()
    (workspace / "sub" / "ext").symlink_to(tmp_path / "elsewhere.txt")
    (workspace / "sub" / "in").symlink_to("../answer.txt")
    (workspace / "dir_out").symlink_to(outside)
    (workspace / "dir_in").symlink_to("sub")
    (workspace / "dangling_in").symlink_to("build/missing.json")
    (workspace / "dangling_out").symlink_to(tmp_path / "missing")
    done = _purge(tmp_path, python, workspace)
    assert done.returncode == 0, done.stderr
    assert not os.path.lexists(workspace / "ext")
    assert not os.path.lexists(workspace / "sub" / "ext")
    assert not os.path.lexists(workspace / "dir_out")
    assert not os.path.lexists(workspace / "dangling_out")
    assert not os.path.lexists(workspace / ".daytona" / "sessions" / "s2" / "ext")
    assert os.readlink(workspace / "sub" / "in") == "../answer.txt"
    assert os.readlink(workspace / "dir_in") == "sub"
    assert os.readlink(workspace / "dangling_in") == "build/missing.json"
    # dir_out was never entered: the link inside the folder it named stays.
    assert os.readlink(outside / "inner") == str(tmp_path / "elsewhere.txt")


@pytest.mark.parametrize("version", _PYTHONS)
def test_a_file_link_through_a_removed_folder_link_stays_dangling(
    tmp_path: Path, workspace: Path, version
):
    """A file link that reaches outside only through a folder link beside
    it: the walk takes a folder's folder links before its files, so the
    folder link goes first and the file link then resolves inside, through
    a name that is gone, and stays. Path.rglob judged them in listing order
    and removed the file link whenever it came first."""
    python = _interpreter(version)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x").write_text("outside\n")
    pairs = []
    for folder in (workspace, workspace / "sub"):
        folder.mkdir(exist_ok=True)
        for i in range(20):
            folder_link, file_link = folder / f"n{i:02d}", folder / f"m{i:02d}"
            made = [(folder_link, str(outside)), (file_link, f"n{i:02d}/x")]
            for link, target in made if i % 2 else reversed(made):
                link.symlink_to(target)
            pairs.append((folder_link, file_link))
    done = _purge(tmp_path, python, workspace)
    assert done.returncode == 0, done.stderr
    for folder_link, file_link in pairs:
        assert not os.path.lexists(folder_link)
        assert os.readlink(file_link) == f"{folder_link.name}/x"
        assert not file_link.exists()
    inside = str(workspace.resolve()) + os.sep
    for folder, dirs, files in os.walk(workspace):
        for name in dirs + files:
            entry = os.path.join(folder, name)
            if os.path.islink(entry):
                assert os.path.realpath(entry).startswith(inside), entry


@pytest.mark.parametrize("version", _PYTHONS)
def test_an_entry_removed_before_it_is_checked_is_skipped(
    tmp_path: Path, workspace: Path, version
):
    """An entry listed and then gone before the walk checks whether it is
    a link has nothing left to purge."""
    python = _interpreter(version)
    (workspace / "fleeting").symlink_to(tmp_path / "elsewhere.txt")
    done = _purge(tmp_path, python, workspace, _ENTRY_GONE)
    assert done.returncode == 0, done.stderr
    assert not os.path.lexists(workspace / "fleeting")
    assert not (workspace / "ext").is_symlink()
    assert not (workspace / ".daytona" / "sessions" / "s2" / "ext").is_symlink()


@pytest.mark.parametrize("version", _PYTHONS)
def test_a_folder_the_walk_cannot_list_fails_the_purge(
    tmp_path: Path, workspace: Path, version
):
    """os.walk without an onerror passed over a folder it could not list,
    and the step passed with the link to outside in it left."""
    python = _interpreter(version)
    (workspace / "locked").mkdir()
    (workspace / "locked" / "ext").symlink_to(tmp_path / "elsewhere.txt")
    done = _purge(tmp_path, python, workspace, _FOLDER_UNLISTABLE)
    assert (workspace / "locked" / "ext").is_symlink()
    assert done.returncode != 0, "purge passed with a link to outside left"
    assert "Permission denied" in done.stderr, done.stderr
    assert "locked" in done.stderr, done.stderr


@pytest.mark.parametrize("version", _PYTHONS)
def test_an_entry_the_walk_cannot_check_fails_the_purge(
    tmp_path: Path, workspace: Path, version
):
    """os.path.islink is False on any lstat error, so a link it could not
    check was kept and the step passed."""
    python = _interpreter(version)
    (workspace / "unreadable").symlink_to(tmp_path / "elsewhere.txt")
    done = _purge(tmp_path, python, workspace, _ENTRY_UNCHECKABLE)
    assert (workspace / "unreadable").is_symlink()
    assert done.returncode != 0, "purge passed with a link to outside left"
    assert "Permission denied" in done.stderr, done.stderr
    assert "unreadable" in done.stderr, done.stderr


def _path_max(path: Path) -> int:
    return os.pathconf(str(path), "PC_PATH_MAX")


def _deep_folder(top: Path, depth: int, size: int) -> int:
    """Make a chain of *depth* folders with *size*-byte names under *top*,
    each made relative to the last, as an agent could; return an open fd of
    the bottom one."""
    fd = os.open(str(top), os.O_RDONLY | os.O_DIRECTORY)
    for level in range(depth):
        name = f"{level:03d}".ljust(size, "x")
        os.mkdir(name, dir_fd=fd)
        child = os.open(name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
        os.close(fd)
        fd = child
    return fd


@pytest.mark.parametrize("version", _PYTHONS)
def test_a_folder_deeper_than_path_max_fails_the_purge(
    tmp_path: Path, workspace: Path, version
):
    """A folder chain deeper than PATH_MAX, with a link to outside at the
    bottom: the walk cannot reach it by path, so the step must fail rather
    than pass with the link left (a review's case)."""
    python = _interpreter(version)
    (workspace / "long").mkdir()
    depth = _path_max(tmp_path) // 101 + 2
    fd = _deep_folder(workspace / "long", depth, 100)
    try:
        os.symlink(str(tmp_path / "elsewhere.txt"), "deep_out", dir_fd=fd)
        done = _purge(tmp_path, python, workspace)
        assert os.readlink("deep_out", dir_fd=fd) == str(tmp_path / "elsewhere.txt")
    finally:
        os.close(fd)
        shutil.rmtree(workspace / "long")
    assert done.returncode != 0, "purge passed with a link to outside left"
    assert os.strerror(errno.ENAMETOOLONG) in done.stderr, done.stderr


@pytest.mark.parametrize("version", _PYTHONS)
def test_a_link_whose_path_is_longer_than_path_max_fails_the_purge(
    tmp_path: Path, workspace: Path, version
):
    """A folder the walk can list holds a link to outside whose name makes
    its path longer than PATH_MAX: it cannot be checked, so the step must
    fail rather than keep it."""
    python = _interpreter(version)
    path_max = _path_max(tmp_path)
    folder = workspace
    while len(str(folder)) < path_max - 200:
        folder = folder / ("d" * 150)
    folder.mkdir(parents=True)
    name = "o" * (path_max - len(str(folder)))
    assert len(str(folder / name)) > path_max
    fd = os.open(str(folder), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.symlink(str(tmp_path / "elsewhere.txt"), name, dir_fd=fd)
        done = _purge(tmp_path, python, workspace)
        assert os.readlink(name, dir_fd=fd) == str(tmp_path / "elsewhere.txt")
    finally:
        os.close(fd)
    assert done.returncode != 0, "purge passed with a link to outside left"
    assert os.strerror(errno.ENAMETOOLONG) in done.stderr, done.stderr
