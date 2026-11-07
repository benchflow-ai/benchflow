"""Guards R7-1 at a2052fa6 across all controller archive extractors."""

import gzip
import io
import os
import tarfile
from types import SimpleNamespace

import pytest

from benchflow.review.evidence import EvidenceError, _extract_archive
from benchflow.taskmd.verify import TaskMdVerifierError, unpack_kept

# Refused on every controller: none of these is one entry of the folder.
BAD_NAMES = [
    "/tmp/dummy",
    "../dummy",
    "x/../dummy",
    r"C:\dummy",
    r"\dummy",
    r"\\host\share\dummy",
    r"x\..\dummy",
    "dummy\x01",
    "dummy\nfile",
    "dummy\rfile",
]

# Refused on a Windows controller only, where Win32 makes each a device, a
# stream, a drive, another spelling or an alias.
WINDOWS_ONLY_NAMES = [
    "C:dummy",
    "NUL",
    "con.txt",
    "AUX.log",
    "COM1",
    "LPT9.txt",
    "COM¹",
    "CONIN$",
    "CONOUT$",
    "dummy.",
    "dummy ",
    "dummy:stream",
    "x/NUL/y",
    "x./y",
    "x /y",
    "x/C:dummy",
    "LONGNA~1",
    "a?b",
    "a*b",
    "a|b",
    "a<b",
]


def _tar(tmp_path, entries):
    archive = tmp_path / "input.tar"
    with tarfile.open(archive, "w") as out:
        for name, directory in entries:
            member = tarfile.TarInfo(name)
            member.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
            member.size = 0 if directory else 5
            out.addfile(member, None if directory else io.BytesIO(b"dummy"))
    return archive


def _extract(backend, archive, destination):
    if backend == "workspace":
        return _extract_archive(
            archive, destination, "/work", max_bytes=100000, max_entries=100
        )
    if backend == "taskmd":
        return unpack_kept(archive, "/keep", destination, {})
    from benchflow.sandbox._bounded_tar import extract_download

    return extract_download(
        gzip.compress(archive.read_bytes()),
        destination,
        tar_bytes=100000,
        content_bytes=100000,
        members=100,
    )


def _member(backend, name):
    # Absolute and drive paths must also be checked when nested below a kept root.
    return name if backend != "taskmd" or name.startswith("/") else "keep/" + name


@pytest.mark.parametrize("backend", ["workspace", "taskmd", "agentcore"])
@pytest.mark.parametrize("name", BAD_NAMES)
def test_all_members_validated_before_writes(tmp_path, backend, name):
    """Guards a2052fa6: an unsafe later member must not leave even the safe first file."""
    archive = _tar(tmp_path, [("keep/safe", False), (_member(backend, name), False)])
    destination = tmp_path / "output"
    destination.mkdir()
    with pytest.raises((EvidenceError, TaskMdVerifierError, RuntimeError, ValueError)):
        _extract(backend, archive, destination)
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize("backend", ["workspace", "taskmd", "agentcore"])
@pytest.mark.parametrize("name", WINDOWS_ONLY_NAMES)
def test_windows_names_validated_before_writes_on_windows(
    tmp_path, monkeypatch, backend, name
):
    """Guards a2052fa6 on a Windows controller, where these names are aliases."""
    from benchflow._utils import archive_paths

    monkeypatch.setattr(archive_paths, "os", SimpleNamespace(name="nt"))
    archive = _tar(tmp_path, [("keep/safe", False), (_member(backend, name), False)])
    destination = tmp_path / "output"
    destination.mkdir()
    with pytest.raises((EvidenceError, TaskMdVerifierError, RuntimeError, ValueError)):
        _extract(backend, archive, destination)
    assert list(destination.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="these names are aliases on Windows")
@pytest.mark.parametrize("backend", ["workspace", "taskmd", "agentcore"])
@pytest.mark.parametrize("name", WINDOWS_ONLY_NAMES)
def test_windows_only_names_are_ordinary_on_posix(tmp_path, backend, name):
    """Guards the fix against 800ecffd1, which applied the Win32 name rules on
    every controller: one ``aux.py`` or ``A:1022428818870.json`` in a Linux
    workspace failed the whole extraction, so the trial lost its review and
    its score on every retry. On a POSIX controller the name is kept as is."""
    member = _member(backend, name)
    archive = _tar(tmp_path, [("keep/safe", False), (member, False)])
    destination = tmp_path / "output"
    destination.mkdir()
    _extract(backend, archive, destination)
    assert (destination / "keep/safe").read_bytes() == b"dummy"
    assert (destination / member).read_bytes() == b"dummy"


@pytest.mark.parametrize("backend", ["workspace", "taskmd", "agentcore"])
@pytest.mark.parametrize(
    "entries",
    [
        [("keep/a", False), ("keep/./a", False)],
        [("keep/a", False), ("keep/a/b", False)],
        [("keep/a/b", False), ("keep/a", False)],
    ],
)
def test_duplicates_and_non_directory_parents_preflight(tmp_path, backend, entries):
    """Guards a2052fa6: normalized duplicates and either order of parent collisions."""
    destination = tmp_path / "output"
    destination.mkdir()
    with pytest.raises((EvidenceError, TaskMdVerifierError, RuntimeError, ValueError)):
        _extract(backend, _tar(tmp_path, entries), destination)
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize("backend", ["workspace", "taskmd", "agentcore"])
@pytest.mark.parametrize(
    "entries",
    [
        [("keep/A", False), ("keep/a", False)],
        [("keep/Dir/a", False), ("keep/dir/b", False)],
        [("keep/Dir/a", False), ("keep/dir", True)],
    ],
)
def test_windows_identity_includes_implicit_parents(
    tmp_path, monkeypatch, backend, entries
):
    """Guards 800ecffd's shared Windows identity without changing the host's pathlib class."""
    from benchflow._utils import archive_paths

    monkeypatch.setattr(archive_paths, "os", SimpleNamespace(name="nt"))
    destination = tmp_path / "output"
    destination.mkdir()
    with pytest.raises((EvidenceError, TaskMdVerifierError, RuntimeError, ValueError)):
        _extract(backend, _tar(tmp_path, entries), destination)
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize("backend", ["workspace", "taskmd", "agentcore"])
def test_safe_tree_preserved(tmp_path, backend):
    """Guards 800ecffd's compatibility: normal files and empty directories survive."""
    destination = tmp_path / "output"
    destination.mkdir()
    _extract(
        backend,
        _tar(tmp_path, [("keep", True), ("keep/file", False), ("keep/empty", True)]),
        destination,
    )
    assert (destination / "keep/file").read_bytes() == b"dummy"
    assert (destination / "keep/empty").is_dir()
