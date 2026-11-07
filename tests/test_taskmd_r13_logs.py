"""Descriptor-only controller log writes: R12-2 regression from 7c24f4d88."""

import contextlib

from benchflow._utils import reachable_write as rw


def test_log_flush_never_reopens_a_planted_staging_name(tmp_path, monkeypatch):
    """Guards R12-2: matching-UID agents can plant files in mode-700 staging."""
    private = tmp_path / "private"
    private.mkdir()
    marker = private / "marker.json"
    marker.write_text("unchanged")
    shared = tmp_path / "jobs"
    shared.mkdir()
    target = rw.ReachablePath.directory(shared) / "output.log"

    @contextlib.contextmanager
    def exposed_staging(*args):
        folder = shared / ".benchflow-private-exposed"
        folder.mkdir(exist_ok=True)
        (folder / "published").symlink_to(marker)
        yield folder

    monkeypatch.setattr(rw, "private_folder", exposed_staging)
    with target.log() as log:
        log.write("first\n")
        log.flush()
        log.write("second\n")
        log.flush()
    assert marker.read_text() == "unchanged"
    assert (shared / "output.log").read_text() == "first\nsecond\n"


def test_controller_write_does_not_open_staging_content(tmp_path, monkeypatch):
    """Guards equivalent R12-2 writer: write() must not truncate planted content."""
    marker = tmp_path / "marker"
    marker.write_text("unchanged")
    target = rw.ReachablePath.directory(tmp_path) / "output"

    @contextlib.contextmanager
    def exposed_staging(*args):
        folder = tmp_path / "stage"
        folder.mkdir(exist_ok=True)
        (folder / "content").symlink_to(marker)
        yield folder

    monkeypatch.setattr(rw, "private_folder", exposed_staging)
    target.write("safe")
    assert marker.read_text() == "unchanged"
    assert (tmp_path / "output").read_text() == "safe"


def test_snapshot_staging_parent_cannot_redirect(tmp_path):
    """Guards the equivalent R12-2 snapshot writer at 7c24f4d88."""
    import pytest

    source = tmp_path / "source"
    source.mkdir()
    (source / "content").write_text("safe")
    private = tmp_path / "private"
    private.mkdir()
    (tmp_path / "staging").symlink_to(private, target_is_directory=True)
    with pytest.raises(rw.ReachableWriteError):
        rw.snapshot_tree(source, tmp_path / "staging" / "snapshot")
    assert list(private.iterdir()) == []


def test_publication_does_not_reopen_a_swapped_staging_parent(tmp_path):
    """Guards R12-2 source publication: an agent cannot redirect the source parent."""
    import pytest

    private = tmp_path / "private"
    private.mkdir()
    (private / "content").write_text("private")
    staging = tmp_path / "stage"
    staging.symlink_to(private, target_is_directory=True)
    target = rw.ReachablePath.directory(tmp_path) / "output"
    with pytest.raises(rw.ReachableWriteError):
        target.publish(staging / "content")
    assert (private / "content").read_text() == "private"
    assert not (tmp_path / "output").exists()


def test_staging_handle_survives_parent_link_swap(tmp_path):
    """Guards R12-2: the staging root itself is held until context exit."""
    import pytest

    root = tmp_path / "jobs"
    root.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    with rw.private_folder(root) as stage:
        moved = stage.with_name(stage.name + "-moved")
        stage.rename(moved)
        stage.symlink_to(private, target_is_directory=True)
        with pytest.raises(rw.ReachableWriteError):
            rw.destination(stage / "content").write("safe")
    assert list(private.iterdir()) == []
    assert list(moved.iterdir()) == []
    moved.rmdir()
