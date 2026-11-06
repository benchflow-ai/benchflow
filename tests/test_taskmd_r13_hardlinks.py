"""Docker hard-link compatibility regression from 7c24f4d88."""

import io
import tarfile

import pytest

from benchflow._utils import reachable_write as rw


def archive_with_link(target="output/a"):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        folder = tarfile.TarInfo("output")
        folder.type = tarfile.DIRTYPE
        folder.mode = 0o755
        archive.addfile(folder)
        regular = tarfile.TarInfo("output/a")
        regular.size = 6
        archive.addfile(regular, io.BytesIO(b"answer"))
        link = tarfile.TarInfo("output/b")
        link.type = tarfile.LNKTYPE
        link.linkname = target
        archive.addfile(link)
    data.seek(0)
    return data


def test_archive_hardlinks_become_independent_inodes(tmp_path):
    """Guards R12-4 introduced by 7c24f4d88: ordinary Docker links copy bytes."""
    rw.publish_tar_stream(
        archive_with_link(),
        rw.ReachablePath.directory(tmp_path) / "out",
        directory=True,
    )
    a, b = tmp_path / "out/a", tmp_path / "out/b"
    assert a.read_bytes() == b.read_bytes() == b"answer"
    assert a.stat().st_ino != b.stat().st_ino
    assert a.stat().st_nlink == b.stat().st_nlink == 1


@pytest.mark.parametrize(
    "target", ["../outside", "/outside", "output/later", "output", "other/a"]
)
def test_archive_hardlink_cannot_borrow_an_unvalidated_entry(tmp_path, target):
    """R13 control: links never resolve arbitrary source names on the host."""
    with pytest.raises((rw.ReachableWriteError, ValueError)):
        rw.publish_tar_stream(
            archive_with_link(target),
            rw.ReachablePath.directory(tmp_path) / "out",
            directory=True,
        )
    assert not (tmp_path / "out").exists()
