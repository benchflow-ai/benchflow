"""The taskmd judge's private staging can be visible through a Compose bind."""

import io
import tarfile

import pytest

from benchflow.taskmd import verify


def test_judge_record_never_promotes_swapped_staging_parent(tmp_path):
    """R12-2 sibling: a plain record path must not trust its current parent."""
    private = tmp_path / "private"
    (private / "records").mkdir(parents=True)
    marker = private / "records" / "criterion.json"
    marker.write_text("original")
    stage = tmp_path / "stage"
    stage.symlink_to(private, target_is_directory=True)
    with pytest.raises((OSError, verify.TaskMdVerifierError)):
        verify._write_json(stage / "records" / "criterion.json", {"changed": True})
    assert marker.read_text() == "original"


def test_kept_archive_never_writes_through_swapped_parent(tmp_path, monkeypatch):
    """R12-2 sibling: checked extraction ancestors are not pathname authority."""
    private = tmp_path / "private"
    private.mkdir()
    marker = private / "answer"
    marker.write_bytes(b"original")
    archive = tmp_path / "kept.tar"
    with tarfile.open(archive, "w") as tar:
        directory = tarfile.TarInfo("app")
        directory.type = tarfile.DIRTYPE
        tar.addfile(directory)
        member = tarfile.TarInfo("app/answer")
        member.size = 7
        tar.addfile(member, io.BytesIO(b"changed"))
    root = tmp_path / "fs"
    original = verify._real_folders
    calls = 0

    def folders(fs_root, parts, path):
        nonlocal calls
        result = original(fs_root, parts, path)
        if parts == ("app",):
            calls += 1
            if calls == 2:
                result.rename(root / "saved")
                result.symlink_to(private, target_is_directory=True)
        return result

    monkeypatch.setattr(verify, "_real_folders", folders)
    with pytest.raises((OSError, verify.TaskMdVerifierError)):
        verify.unpack_kept(archive, "/app", root, {})
    assert marker.read_bytes() == b"original"


def test_judge_archive_never_reopens_output_parent(tmp_path):
    """The outgoing runner stash is a staging writer too, not a trusted path."""
    custody = tmp_path / "custody"
    (custody / "app").mkdir(parents=True)
    (custody / "app" / "answer").write_text("answer")
    private = tmp_path / "private"
    (private / "records").mkdir(parents=True)
    marker = private / "records" / "stash.tar"
    marker.write_bytes(b"original")
    stage = tmp_path / "stage"
    stage.symlink_to(private, target_is_directory=True)
    with pytest.raises(OSError):
        verify.custody_archive(custody, ["/app"], stage / "records" / "stash.tar")
    assert marker.read_bytes() == b"original"


def test_judge_archive_never_follows_swapped_custody_parent(tmp_path):
    """A source ancestor cannot redirect the stash into controller-only data."""
    private = tmp_path / "private"
    (private / "app").mkdir(parents=True)
    (private / "app" / "secret").write_text("controller marker")
    custody = tmp_path / "custody"
    custody.symlink_to(private, target_is_directory=True)
    with pytest.raises(OSError):
        verify.custody_archive(custody, ["/app"], tmp_path / "stash.tar")
