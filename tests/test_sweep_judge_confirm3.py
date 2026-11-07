"""Guard CONFIRM-taskmd-rewards-3 against reviewed commit 11fc9381."""

import ctypes
import os
import shutil
import stat
import struct

import pytest

from benchflow.rewards.file_readers import find_deliverables


def test_deliverables_do_not_assume_posix_flags(tmp_path, monkeypatch):
    """CONFIRM-3: missing POSIX capabilities must not raise AttributeError."""
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "answer.txt").write_text("correct")
    monkeypatch.delattr(os, "O_DIRECTORY")
    # Unsupported non-Windows filesystems fail closed; Windows gets its own
    # no-follow implementation. Neither path may assume POSIX attributes.
    try:
        result = find_deliverables(evidence)
    except OSError:
        result = {}
    assert result in ({}, {"answer.txt": "correct"})


# Exercise the actual Windows API adapter on Linux using a handle-level kernel
# stand-in. This validates the flags, lifetime and attribute decisions; it is
# not a claim of a native Windows execution on this Linux VM.
@pytest.fixture
def windows_api(monkeypatch):
    from benchflow.rewards import _safe_evidence as safe

    class Function:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args):
            return self.fn(*args)

    class Kernel:
        def __init__(self):
            self.handles = {}
            self.calls = []
            self.reparse = set()
            self.file_type = 1
            self.next_handle = 10
            self.after_open = None
            self.NtCreateFile = Function(self.nt_create)
            self.NtQueryDirectoryFile = Function(self.query)
            self.CreateFileW = Function(self.create)
            self.GetFileInformationByHandle = Function(self.info)
            self.GetFileType = Function(lambda handle: self.file_type)
            self.CloseHandle = Function(self.close)
            self.directory_flag = os.O_DIRECTORY

        def create(self, path, access, sharing, security, disposition, flags, template):
            from pathlib import Path

            assert access == 0x80000000
            assert sharing == 1  # no FILE_SHARE_WRITE or FILE_SHARE_DELETE
            assert disposition == 3
            assert flags == 0x02200000
            path = Path(path)
            info = path.lstat()
            handle = self.next_handle
            self.next_handle += 1
            fd = None
            if stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode):
                # Read through the opened object, independently of later names.
                fd = os.open(path, os.O_RDONLY)
            self.handles[handle] = (path, info, fd)
            self.calls.append(path)
            if self.after_open and stat.S_ISREG(info.st_mode):
                self.after_open(path)
            return handle

        def nt_create(
            self,
            opened,
            access,
            attrs,
            io,
            alloc,
            attr,
            share,
            disposition,
            options,
            ea,
            size,
        ):
            assert access == 0x80100000
            assert share == 1 and disposition == 1
            assert options in (0x00200021, 0x00200060)
            attrs = attrs._obj
            parent_path, _, _ = self.handles[attrs.root]
            name = attrs.name.contents.buffer
            assert attrs.name.contents.length == len(name.encode("utf-16-le"))
            assert "/" not in name and "\\" not in name
            opened._obj.value = self.create(
                str(parent_path / name), 0x80000000, 1, None, 3, 0x02200000, None
            )
            return 0

        def query(
            self, handle, event, apc, ctx, io, buffer, size, kind, single, name, restart
        ):
            assert kind == 12 and not single and name is None
            if not restart:
                return -2147483642  # STATUS_NO_MORE_FILES
            _, _, fd = self.handles[handle]
            names = [".", "..", *os.listdir(fd)]
            entries = []
            for index, name in enumerate(names):
                encoded = name.encode("utf-16-le")
                count = (12 + len(encoded) + 7) // 8 * 8
                entries.append(
                    struct.pack(
                        "<III", count if index < len(names) - 1 else 0, 0, len(encoded)
                    )
                    + encoded
                    + b"\0" * (count - 12 - len(encoded))
                )
            data = b"".join(entries)
            ctypes.memmove(buffer, data, len(data))
            io._obj.information = len(data)
            return 0

        def info(self, handle, pointer):
            path, info, _fd = self.handles[handle]
            attrs = 0x10 if stat.S_ISDIR(info.st_mode) else 0
            if stat.S_ISLNK(info.st_mode) or path in self.reparse:
                attrs |= 0x400
            pointer._obj.attributes = attrs
            return 1

        def close(self, handle):
            _path, _info, fd = self.handles.pop(handle)
            if fd is not None:
                os.close(fd)
            return 1

    kernel = Kernel()
    kernel.safe = safe
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **kw: kernel, raising=False)
    monkeypatch.setattr(safe, "_posix_available", lambda: False)
    monkeypatch.setattr(safe, "_is_windows", lambda: True)

    def file(self, handle):
        path, _info, fd = kernel.handles.pop(handle)
        assert all(parent in kernel.calls for parent in path.parents)
        # Every ancestor is still held while the file is consumed.
        held = [item[0] for item in kernel.handles.values()]
        assert all(parent in held for parent in path.parents)
        return os.fdopen(fd, "rb")

    monkeypatch.setattr(safe._WindowsAPI, "file", file)
    return kernel


def test_windows_reads_regular_evidence_without_posix_flags(
    tmp_path, monkeypatch, windows_api
):
    """CONFIRM-3: supported Windows capture keeps grading ordinary evidence."""
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "answer.txt").write_text("correct")
    for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"):
        monkeypatch.delattr(os, flag)
    # Linux shutil uses O_NONBLOCK internally; Windows uses its path variant.
    monkeypatch.setattr(shutil, "_use_fd_functions", False)
    assert find_deliverables(evidence) == {"answer.txt": "correct"}
    assert windows_api.handles == {}


@pytest.mark.parametrize(
    "link_at", ["root", "middle", "leaf", "junction-root", "junction-leaf"]
)
def test_windows_refuses_all_reparse_components(tmp_path, windows_api, link_at):
    """CONFIRM-3: links and junctions at every hop fail closed."""
    evidence = tmp_path / "evidence"
    output = evidence / "output"
    output.mkdir(parents=True)
    answer = output / "answer.txt"
    answer.write_text("correct")
    controller = tmp_path / "controller"
    controller.mkdir()
    (controller / "answer.txt").write_text("DUMMY_CONTROLLER_MARKER")
    if link_at.startswith("junction"):
        windows_api.reparse.add(evidence if link_at.endswith("root") else answer)
    elif link_at == "leaf":
        answer.unlink()
        answer.symlink_to(controller / "answer.txt")
    elif link_at == "middle":
        answer.unlink()
        output.rmdir()
        output.symlink_to(controller)
    else:
        answer.unlink()
        output.rmdir()
        evidence.rmdir()
        evidence.symlink_to(controller)
    assert find_deliverables(output) == {}
    assert windows_api.handles == {}


@pytest.mark.parametrize("kind", [2, 3])
def test_windows_rejects_non_disk_handles(tmp_path, windows_api, kind):
    windows_api.file_type = kind
    assert find_deliverables(tmp_path) == {}
    assert windows_api.handles == {}


@pytest.mark.parametrize(
    "name",
    [
        "..",
        "x:y.txt",
        "x\\y.txt",
        "NUL.txt",
        "COM1.csv",
        "LPT\u00b9.txt",
        "answer.txt.",
        "answer.txt ",
    ],
)
def test_windows_rejects_name_aliases(windows_api, tmp_path, name):
    with pytest.raises(ValueError):
        windows_api.safe._WindowsDirectory(0, windows_api.safe._WindowsAPI()).open_file(
            name
        )
    assert windows_api.handles == {}


def test_windows_snapshot_uses_open_handle_after_rename(tmp_path, windows_api):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    answer = evidence / "answer.txt"
    answer.write_text("correct")
    controller = tmp_path / "controller.txt"
    controller.write_text("DUMMY_CONTROLLER_MARKER")

    def swap(path):
        if path == answer:
            answer.unlink()
            answer.symlink_to(controller)

    windows_api.after_open = swap
    assert find_deliverables(evidence) == {"answer.txt": "correct"}
    assert windows_api.handles == {}


def test_archive_extraction_needs_no_posix_file_flags(tmp_path, monkeypatch):
    """CONFIRM-3: native bounded capture can extract on Windows too."""
    import io
    import tarfile

    from benchflow.taskmd.verify import unpack_kept

    archive = tmp_path / "input.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("app/answer.txt")
        info.size = 7
        tar.addfile(info, io.BytesIO(b"correct"))
    for flag in ("O_NOFOLLOW", "O_CLOEXEC", "fchmod"):
        monkeypatch.delattr(os, flag)
    dest = tmp_path / "private"
    assert unpack_kept(archive, "/app", dest, {}) is None
    assert (dest / "app/answer.txt").read_text() == "correct"


@pytest.mark.parametrize(
    "names",
    [
        ["app/a.txt", "app/A.txt"],
        ["app/x:y.txt"],
        ["app/x\\y.txt"],
        ["app/NUL.txt"],
        ["app/NUL .txt"],
        ["app/CON.csv"],
        ["app/answer.txt."],
        ["app/answer.txt "],
        ["app/sub/C:outside.txt"],
    ],
)
def test_windows_archive_aliases_are_rejected_before_writing(
    tmp_path, monkeypatch, names
):
    """CONFIRM-3: Win32 aliases cannot escape or collide during bounded extraction."""
    import io
    import tarfile

    from benchflow.taskmd import verify

    class WindowsOS:
        name = "nt"

        def __getattr__(self, name):
            return getattr(os, name)

    archive = tmp_path / "input.tar"
    with tarfile.open(archive, "w") as tar:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = 7
            tar.addfile(info, io.BytesIO(b"correct"))
    from benchflow._utils import archive_paths

    monkeypatch.setattr(archive_paths, "os", WindowsOS())
    dest = tmp_path / "private"
    with pytest.raises(verify.TaskMdVerifierError):
        verify.unpack_kept(archive, "/app", dest, {})
    assert not dest.exists()
