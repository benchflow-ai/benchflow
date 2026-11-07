"""Guards R7-3: a2052fa6 copied diagnostic logs before looking for markers."""

import os

import pytest

from benchflow._utils import reachable_write
from benchflow.task.verifier_scan import (
    _has_dep_install_failure,
    _has_guard_load_failure,
)


@pytest.mark.parametrize("handle", [False, True])
@pytest.mark.parametrize("scanner", ["dependency", "guard"])
def test_sparse_log_scan_never_materializes_bytes(
    tmp_path, monkeypatch, handle, scanner
):
    """Guards a2052fa6 with a real sparse TiB inode and a bounded failing copy seam."""
    root = tmp_path / "logs"
    root.mkdir()
    path = root / "test-stdout.txt"
    marker = (
        "dependency install failed\n"
        if scanner == "dependency"
        else 'Error importing plugin "guard_dummy"\n'
    )
    with path.open("wb") as f:
        f.write(marker.encode())
        f.truncate(1024**4)
    assert path.stat().st_blocks * 512 < 1024**2

    def no_copy(*args, **kwargs):
        raise AssertionError("diagnostic scan tried to snapshot a sparse log")

    monkeypatch.setattr(reachable_write, "_snapshot_stream", no_copy)
    source = reachable_write.ReachablePath.directory(root) if handle else root
    if scanner == "dependency":
        assert _has_dep_install_failure(source / path.name)
    else:
        assert _has_guard_load_failure(source, "guard_dummy")
    assert path.stat().st_blocks * 512 < 1024**2


@pytest.mark.parametrize("handle", [False, True])
@pytest.mark.parametrize("kind", ["leaf", "parent", "fifo"])
def test_diagnostic_no_follow(tmp_path, handle, kind):
    """Guards cc921744's streaming change against reintroducing the R6-1 link reads."""
    root = tmp_path / "logs"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "log").write_text("dependency install failed")
    source = root / "log"
    if kind == "parent":
        (root / "alias").symlink_to(outside, target_is_directory=True)
        source = root / "alias/log"
    elif kind == "leaf":
        source.symlink_to(outside / "log")
    else:
        os.mkfifo(source)
    if handle:
        source = reachable_write.ReachablePath.directory(root) / source.relative_to(
            root
        )
    assert not _has_dep_install_failure(source)
