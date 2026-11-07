"""Caller-root regressions from 7c24f4d88 (R12-1 and the original R9-2)."""

import io
import tarfile

import pytest

from benchflow._utils import reachable_write as rw
from benchflow.sandbox._download import private_download


class Download:
    @private_download(directory=False)
    async def download_file(self, source_path, target_path):
        raise AssertionError("must stream")

    async def _stream_download(self, source, target, **kwargs):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as archive:
            entry = tarfile.TarInfo("marker.json")
            entry.size = 7
            archive.addfile(entry, io.BytesIO(b"changed"))
        data.seek(0)
        rw.publish_tar_stream(data, target, directory=False)


@pytest.mark.asyncio
async def test_plain_download_does_not_promote_agent_parent(tmp_path):
    """Guards R12-1 introduced by 7c24f4d88 through the actual download decorator."""
    shared, private = tmp_path / "shared", tmp_path / "private"
    shared.mkdir()
    private.mkdir()
    marker = private / "marker.json"
    marker.write_text("original")
    # The trusted root is selected before the agent plants a descendant link.
    root = rw.destination(shared / "unused", root=shared).parent
    (shared / "results").symlink_to(private, target_is_directory=True)
    with pytest.raises(rw.ReachableWriteError):
        await Download().download_file("/answer", shared / "results/marker.json")
    assert marker.read_text() == "original"
    assert root.is_dir()


@pytest.mark.asyncio
async def test_unregistered_plain_destination_never_resolves_parent(tmp_path):
    """Guards 7c24f4d88 without caller registration: default root stays pinned."""
    private = tmp_path / "private"
    private.mkdir()
    marker = private / "marker.json"
    marker.write_text("original")
    (tmp_path / "results").symlink_to(private, target_is_directory=True)
    with pytest.raises(rw.ReachableWriteError):
        await Download().download_file("/answer", tmp_path / "results/marker.json")
    assert marker.read_text() == "original"


@pytest.mark.asyncio
async def test_honest_plain_destination(tmp_path):
    """Plain outputs under the initial temp root do not need late root promotion."""
    target = tmp_path / "ordinary/subdir/marker.json"
    await Download().download_file("/answer", target)
    assert target.read_text() == "changed"
