"""Two publications of one destination: review B's finding on caae129ee (TASKMD-R16).

B-R15-M1: a directory commit moves the old output aside, and a failed
install renamed it back with os.rename, replacing a publication another
commit completed in between. The rule: a rollback never replaces a name it
did not vacate. Cases from review B (``test_review_r15.py``) and review A
(``test_review_r15_a.py``), with their interleavings unchanged.
"""

import errno
import io
import sys
import threading

import pytest

from benchflow._utils import reachable_write as rw
from tests.test_taskmd_r14 import _archive, _old_output, _staging_left

PREVIOUS = getattr(rw, "_PREVIOUS", "previous")


def _listing(out):
    if not out.exists():
        return None
    if out.is_dir():
        return sorted(p.name for p in out.iterdir())
    return out.read_text()


def _staged(held, *, directory=True, files=("a",)):
    stage = rw.StagedTar(held / "out", directory=directory)
    stage.__enter__()
    stage.extract(io.BytesIO(_archive(directory=directory, files=files)))
    return stage


def _assert_old_kept(root, stage, error, old):
    """A's old output survives only in its staging, which the error names."""
    assert error is not None
    kept = root / stage._work_name / PREVIOUS
    assert _listing(kept) == old
    assert any(
        stage._work_name in note and PREVIOUS in note for note in error.__notes__
    )
    assert stage._work_name in _staging_left(root)


# --- review B: real threads, A paused after moving the old entry aside --------


@pytest.mark.parametrize("second_kind", ["nonempty-dir", "empty-dir", "file"])
def test_concurrent_rollback_keeps_successful_publication(
    tmp_path, monkeypatch, second_kind
):
    """B publishes in A's window; A's install then fails; B's output stays."""
    target = tmp_path / "out"
    if second_kind == "file":
        target.write_text("OLD")
        old = "OLD"
    else:
        _old_output(tmp_path)
        old = ["old"]
    held = rw.pin_destination(tmp_path)
    moved, release = threading.Event(), threading.Event()
    rename = rw.os.rename
    errors = []

    def hooked(src, dst, *args, **kwargs):
        if threading.current_thread().name == "publication-a":
            if src == "out" and dst == PREVIOUS:
                result = rename(src, dst, *args, **kwargs)
                moved.set()
                assert release.wait(10)
                return result
            # A folder over B's file fails ENOTDIR by itself; inject otherwise.
            if src == "payload" and second_kind != "file":
                raise OSError(errno.ENOSPC, "injected ENOSPC on A's install")
        return rename(src, dst, *args, **kwargs)

    first = _staged(held, files=("a",))
    monkeypatch.setattr(rw.os, "rename", hooked)

    def commit_a():
        try:
            first.commit()
        except OSError as exc:
            errors.append(exc)

    thread = threading.Thread(target=commit_a, name="publication-a")
    thread.start()
    try:
        assert moved.wait(5)
        rw.publish_tar_stream(
            io.BytesIO(
                _archive(
                    directory=second_kind != "file",
                    files=() if second_kind == "empty-dir" else ("b",),
                )
            ),
            held / "out",
            directory=second_kind != "file",
        )
        published = _listing(target)
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive()
    (error,) = errors
    first.__exit__(type(error), error, None)
    assert _listing(target) == published, "failed A rolled back over successful B"
    expected = errno.ENOTDIR if second_kind == "file" else errno.ENOSPC
    assert error.errno == expected  # the primary error is A's install, unchanged
    assert not first.committed
    _assert_old_kept(tmp_path, first, error, old)


# --- review A: B commits inside A's window, single thread --------------------


@pytest.mark.parametrize(
    "case",
    [
        "B-in-between-nonempty",
        "A-fails-B-nonempty",
        "A-fails-B-empty",
        "A-fails-B-file-over-old-file",
    ],
)
def test_two_publications_one_destination(tmp_path, monkeypatch, case):
    """B commits while A's directory commit sits between its two renames."""
    if case == "A-fails-B-file-over-old-file":
        (tmp_path / "out").write_text("OLD")
        old = "OLD"
    else:
        _old_output(tmp_path)
        old = ["old"]
    held = rw.pin_destination(tmp_path)
    a = _staged(held, files=("from_a",))
    if case == "A-fails-B-empty":
        b = _staged(held, files=())
    elif case == "A-fails-B-file-over-old-file":
        b = _staged(held, directory=False, files=("x",))
    else:
        b = _staged(held, files=("from_b",))
    rename = rw.os.rename
    armed = [True]

    def hooked(src, dst, *args, **kwargs):
        if armed[0] and src == "payload" and kwargs.get("src_dir_fd") == a._work_fd:
            armed[0] = False
            b.commit()  # B runs to completion between A's aside and A's install
            if case != "B-in-between-nonempty":
                raise OSError(errno.ENOSPC, "injected ENOSPC on A's install")
        return rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(rw.os, "rename", hooked)
    a_error = None
    try:
        a.commit()
    except OSError as exc:
        a_error = exc
    finally:
        monkeypatch.setattr(rw.os, "rename", rename)
    a.__exit__(type(a_error) if a_error else None, a_error, None)
    b.__exit__(None, None, None)
    final = _listing(tmp_path / "out")
    assert b.committed
    b_output = {
        "B-in-between-nonempty": ["from_b"],
        "A-fails-B-nonempty": ["from_b"],
        "A-fails-B-empty": [],
        "A-fails-B-file-over-old-file": "NEW",
    }[case]
    assert not a.committed
    assert final == b_output, "B reported success but A's rollback replaced it"
    # Uninjected, A's own install fails: a folder never replaces a full one.
    expected = (
        {errno.ENOTEMPTY, errno.EEXIST}
        if case == "B-in-between-nonempty"
        else {errno.ENOSPC}
    )
    assert a_error is not None and a_error.errno in expected
    _assert_old_kept(tmp_path, a, a_error, old)


def test_single_failed_commit_still_restores_old_folder(tmp_path, monkeypatch):
    """Control: with nobody else publishing, the old folder is put back."""
    out = _old_output(tmp_path)
    held = rw.pin_destination(tmp_path)
    rename = rw.os.rename

    def fail(src, dst, *args, **kwargs):
        if src == "payload":
            raise OSError(errno.ENOSPC, "injected ENOSPC on install")
        return rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(rw.os, "rename", fail)
    with pytest.raises(OSError) as error:
        rw.publish_tar_stream(io.BytesIO(_archive()), held / "out", directory=True)
    assert error.value.errno == errno.ENOSPC
    assert _listing(out) == ["old"]
    assert _staging_left(tmp_path) == []


# --- the no-replace rename itself --------------------------------------------


@pytest.mark.skipif(
    not (sys.platform.startswith("linux") or sys.platform == "darwin"),
    reason="kernel no-replace rename is Linux and macOS only",
)
def test_kernel_noreplace_rename_never_replaces(tmp_path):
    """The real syscall is reached here, so the fallback is not what passes."""
    import os

    fd = os.open(tmp_path, os.O_RDONLY)
    try:
        (tmp_path / "src").write_text("SRC")
        (tmp_path / "file").write_text("TAKEN")
        (tmp_path / "empty").mkdir()
        for taken in ("file", "empty"):
            with pytest.raises(FileExistsError):
                rw._renameat_noreplace("src", taken, src_dir_fd=fd, dst_dir_fd=fd)
        assert (tmp_path / "file").read_text() == "TAKEN"
        assert (tmp_path / "empty").is_dir()
        assert rw._renameat_noreplace("src", "free", src_dir_fd=fd, dst_dir_fd=fd)
        assert (tmp_path / "free").read_text() == "SRC"
        assert not (tmp_path / "src").exists()
    finally:
        os.close(fd)


class _FakeLibc:
    """A libc whose no-replace rename fails ``err`` (or that has none)."""

    def __init__(self, err=None):
        self.err = err
        if err is not None:
            self.renameat2 = self.renameatx_np = self.syscall = self._fail

    def _fail(self, *args):
        import ctypes

        ctypes.set_errno(self.err)
        return -1


@pytest.mark.skipif(
    not (sys.platform.startswith("linux") or sys.platform == "darwin"),
    reason="kernel no-replace rename is Linux and macOS only",
)
@pytest.mark.parametrize("err", [None, errno.ENOSYS, errno.EINVAL, errno.ENOTSUP])
def test_unavailable_kernel_noreplace_reports_false(tmp_path, monkeypatch, err):
    """No wrapper, an old kernel, or a filesystem without the flag: False."""
    import ctypes
    import os

    if err == errno.ENOTSUP and sys.platform.startswith("linux"):
        err = errno.EPERM  # what an older seccomp profile answers
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: _FakeLibc(err))
    if err is None:
        monkeypatch.setattr(rw, "_SYS_RENAMEAT2", {})
    (tmp_path / "src").write_text("SRC")
    fd = os.open(tmp_path, os.O_RDONLY)
    try:
        assert not rw._renameat_noreplace("src", "dst", src_dir_fd=fd, dst_dir_fd=fd)
    finally:
        os.close(fd)
    assert (tmp_path / "src").read_text() == "SRC"


def _without_kernel_noreplace(monkeypatch):
    """No no-replace rename on this host: the real fallback branch runs."""
    monkeypatch.setattr(rw, "_renameat_noreplace", lambda *a, **k: False)


@pytest.mark.parametrize("taken", [False, True], ids=["free", "taken-by-b"])
def test_without_noreplace_a_file_rolls_back_by_link(tmp_path, monkeypatch, taken):
    """Old file, folder payload, no syscall: link-and-unlink never replaces B."""
    out = tmp_path / "out"
    out.write_text("OLD")
    held = rw.pin_destination(tmp_path)
    a = _staged(held, files=("from_a",))
    b = _staged(held, directory=False, files=("x",))
    rename = rw.os.rename

    def hooked(src, dst, *args, **kwargs):
        if src == "payload" and kwargs.get("src_dir_fd") == a._work_fd:
            if taken:
                b.commit()
            raise OSError(errno.ENOSPC, "injected ENOSPC on A's install")
        return rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(rw.os, "rename", hooked)
    _without_kernel_noreplace(monkeypatch)
    with pytest.raises(OSError) as error:
        a.commit()
    monkeypatch.undo()
    a.__exit__(OSError, error.value, None)
    b.__exit__(None, None, None)
    assert error.value.errno == errno.ENOSPC
    if taken:
        assert out.read_text() == "NEW"
        _assert_old_kept(tmp_path, a, error.value, "OLD")
    else:
        assert out.read_text() == "OLD"
        assert _staging_left(tmp_path) == []


@pytest.mark.parametrize("taken", [False, True], ids=["free", "taken-by-b"])
def test_without_noreplace_a_folder_stays_in_staging(tmp_path, monkeypatch, taken):
    """No call can move a folder without maybe replacing: keep it, never delete."""
    out = _old_output(tmp_path)
    held = rw.pin_destination(tmp_path)
    a = _staged(held, files=("from_a",))
    b = _staged(held, files=())
    rename = rw.os.rename

    def hooked(src, dst, *args, **kwargs):
        if src == "payload" and kwargs.get("src_dir_fd") == a._work_fd:
            if taken:
                b.commit()
            raise OSError(errno.ENOSPC, "injected ENOSPC on A's install")
        return rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(rw.os, "rename", hooked)
    _without_kernel_noreplace(monkeypatch)
    with pytest.raises(OSError) as error:
        a.commit()
    monkeypatch.undo()
    a.__exit__(OSError, error.value, None)
    b.__exit__(None, None, None)
    assert error.value.errno == errno.ENOSPC
    assert _listing(out) == ([] if taken else None)
    _assert_old_kept(tmp_path, a, error.value, ["old"])
