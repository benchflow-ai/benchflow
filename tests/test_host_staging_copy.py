"""Private host snapshots keep their custody with kernel copies."""

from __future__ import annotations

import errno
import os
import stat
import subprocess
import sys
import textwrap

import pytest

from benchflow._utils.task_authoring import task_digest
from benchflow.task import staging


def test_deep_snapshot_keeps_the_base_descriptor_budget(tmp_path):
    """Review 1 against the depth regression in 780fb9a9."""
    script = textwrap.dedent("""
        import os
        from pathlib import Path
        import resource
        import sys
        import tempfile

        from benchflow.task import staging

        work = Path(sys.argv[1])
        tempfile.tempdir = str(work)
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        limit = min(128, hard) if hard != resource.RLIM_INFINITY else 128
        resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
        descriptors = "/proc/self/fd" if Path("/proc/self/fd").is_dir() else "/dev/fd"
        before = set(os.listdir(descriptors))
        depth = (limit - len(before) - 8) * 3 // 4
        assert depth > 0
        environment = work / "task" / "environment"
        environment.mkdir(parents=True)
        leaf = environment
        for _ in range(depth):
            leaf /= "d"
            leaf.mkdir()
        (leaf / "payload").write_bytes(b"deep task payload")
        live = set(staging._live)
        snapshot = None
        try:
            snapshot = staging.stage_environment(environment)
            copied = snapshot.environment_dir.joinpath(*(["d"] * depth), "payload")
            assert copied.read_bytes() == b"deep task payload"
        finally:
            if snapshot is not None:
                snapshot.remove()
        assert (leaf / "payload").read_bytes() == b"deep task payload"
        assert set(os.listdir(descriptors)) == before
        assert set(staging._live) == live
        assert not list(work.glob(f"{staging.STAGE_PREFIX}*"))
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_kernel_copy_preserves_content_metadata_and_independence(tmp_path):
    """Commit 599d0f44: bytes, modes and times stay private."""
    source = tmp_path / "source"
    source.mkdir()
    payload = bytes(range(256)) * (12 * 1024 + 1)
    original = source / "script"
    original.write_bytes(payload)
    original.chmod(0o551)
    os.utime(original, ns=(1_700_000_000_123456789, 1_700_000_001_987654321))
    info = original.stat()
    (source / "empty").touch()
    digest = task_digest(source)

    destination = tmp_path / "destination"
    staging.copy_tree(source, destination)
    copied = destination / "script"
    assert copied.read_bytes() == payload
    assert (destination / "empty").read_bytes() == b""
    assert task_digest(destination) == digest
    assert stat.S_IMODE(copied.stat().st_mode) == 0o751
    assert copied.stat().st_mtime_ns == info.st_mtime_ns
    assert copied.stat().st_ino != original.stat().st_ino

    original.chmod(0o751)
    original.write_bytes(b"x" * len(payload))
    os.utime(original, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert task_digest(source) != digest
    assert copied.read_bytes() == payload
    copied.write_bytes(b"changed copy")
    assert original.read_bytes() == b"x" * len(payload)


def test_pinned_destination_keeps_nested_metadata_and_links(tmp_path):
    """Commit 780fb9a9: directory metadata and links are kept."""
    source = tmp_path / "source"
    nested = source / "deep" / "café"
    nested.mkdir(parents=True)
    (nested / "payload").write_bytes(b"nested bytes")
    (nested / "relative").symlink_to("payload")
    (source / "absolute").symlink_to(nested)
    nested.chmod(0o555)
    os.utime(nested, ns=(1_700_000_000_123456789, 1_700_000_001_987654321))
    before = nested.stat()
    destination = tmp_path / "destination"
    staging.copy_tree(source, destination)
    copied = destination / "deep" / "café"
    assert stat.S_IMODE(copied.stat().st_mode) == 0o755
    assert copied.stat().st_mtime_ns == before.st_mtime_ns
    assert copied.stat().st_atime_ns == before.st_atime_ns
    assert os.readlink(copied / "relative") == "payload"
    assert os.readlink(destination / "absolute") == "deep/café"
    assert (destination / "absolute" / "relative").read_bytes() == b"nested bytes"


@pytest.mark.parametrize("stop", [None, 0, errno.EXDEV, errno.EOPNOTSUPP])
def test_short_or_partial_kernel_copy_resumes_at_current_offsets(
    tmp_path, monkeypatch, stop
):
    """Commit 599d0f44: partial copies keep every byte once."""
    source = tmp_path / "source"
    source.mkdir()
    payload = b"first | middle | last"
    (source / "data").write_bytes(payload)
    calls = 0

    def copy_range(fd, out, count):
        nonlocal calls
        calls += 1
        if calls > 1 and stop is not None:
            if stop == 0:
                return 0  # A filesystem reporting zero before EOF.
            raise OSError(stop, "kernel copy unavailable")
        block = os.read(fd, min(count, 3))
        return os.write(out, block)

    monkeypatch.setattr(staging.os, "copy_file_range", copy_range, raising=False)
    destination = tmp_path / "destination"
    staging.copy_tree(source, destination)
    assert calls >= 2
    assert (destination / "data").read_bytes() == payload


@pytest.mark.parametrize(
    "unsupported",
    [None, errno.ENOSYS, errno.EXDEV, errno.EINVAL, errno.EOPNOTSUPP, errno.EPERM],
)
def test_unavailable_kernel_copy_uses_buffered_copy(tmp_path, monkeypatch, unsupported):
    """Commit 599d0f44: unsupported copies still work."""
    source = tmp_path / "source"
    source.mkdir()
    payload = b"fallback bytes" * 100_000
    (source / "data").write_bytes(payload)
    if unsupported is None:
        monkeypatch.delattr(staging.os, "copy_file_range", raising=False)
    else:

        def unavailable(*args):
            raise OSError(unsupported, "unsupported")

        monkeypatch.setattr(staging.os, "copy_file_range", unavailable, raising=False)
    destination = tmp_path / "destination"
    staging.copy_tree(source, destination)
    assert (destination / "data").read_bytes() == payload


@pytest.mark.parametrize("failure", [errno.ENOSPC, errno.EIO])
def test_kernel_copy_failure_refuses_and_removes_partial_snapshot(
    tmp_path, monkeypatch, failure
):
    """Commit 599d0f44: I/O errors still refuse and clean up."""
    environment = tmp_path / "task" / "environment"
    environment.mkdir(parents=True)
    (environment / "data").write_bytes(b"task data")
    monkeypatch.setattr(staging.tempfile, "tempdir", str(tmp_path))
    calls = 0

    def broken(fd, out, count):
        nonlocal calls
        calls += 1
        if calls == 1:
            return os.write(out, os.read(fd, 2))
        raise OSError(failure, "copy failed")

    monkeypatch.setattr(staging.os, "copy_file_range", broken, raising=False)
    with pytest.raises(staging.StagingRefused) as caught:
        staging.stage_environment(environment)
    assert caught.value.transient
    assert "changed while being copied; try again" in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)
    assert caught.value.__cause__.errno == failure
    assert not list(tmp_path.glob(f"{staging.STAGE_PREFIX}*"))


@pytest.mark.parametrize("replacement", ["symlink", "fifo"])
def test_kernel_copy_does_not_bypass_source_open_checks(
    tmp_path, monkeypatch, replacement
):
    """Commit 599d0f44: a swapped source is refused first."""
    source = tmp_path / "source"
    source.mkdir()
    victim = source / "data"
    victim.write_bytes(b"task bytes")
    outside = tmp_path / "operator"
    outside.write_bytes(b"operator bytes")
    original_open = staging.os.open

    def swapping(path, flags, *args, **kwargs):
        if (
            path == "data"
            and kwargs.get("dir_fd") is not None
            and not flags & os.O_CREAT
        ):
            victim.unlink()
            if replacement == "symlink":
                victim.symlink_to(outside)
            else:
                os.mkfifo(victim)
        return original_open(path, flags, *args, **kwargs)

    def must_not_copy(*args):
        pytest.fail("a refused source reached the kernel copy")

    monkeypatch.setattr(staging.os, "open", swapping)
    monkeypatch.setattr(staging.os, "copy_file_range", must_not_copy, raising=False)
    with pytest.raises(staging.StagingRefused) as caught:
        staging.copy_tree(source, tmp_path / "destination")
    assert caught.value.transient
    expected = (
        "changed while being copied; try again"
        if replacement == "symlink"
        else "changed into something that is not a file"
    )
    assert expected in str(caught.value)
    assert not (tmp_path / "destination" / "data").exists()
