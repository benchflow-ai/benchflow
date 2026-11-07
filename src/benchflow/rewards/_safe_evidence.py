"""No-follow handles for controller-side evidence, kept open through use."""

from __future__ import annotations

import os
import stat
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePath
from types import ModuleType
from typing import BinaryIO


def _platform_member(module: ModuleType, name: str):
    """Load a Windows-only API only when that filesystem backend is selected."""
    return getattr(module, name)


def _location(
    directory: Path, evidence_root: Path | None
) -> tuple[Path, tuple[str, ...]]:
    directory = directory.absolute()
    if evidence_root is None:
        return Path(directory.anchor), directory.parts[1:]
    root = evidence_root.absolute()
    relative = directory.relative_to(root)
    if ".." in relative.parts:
        raise ValueError("evidence must stay under its downloaded root")
    parent, components = _root_components(root)
    # Only the controller's parent may resolve OS aliases, such as macOS /var.
    anchor = Path(root.anchor) if parent is None else Path(parent).resolve()
    return anchor, (*components, *relative.parts)


def _root_components(root: PurePath) -> tuple[PurePath | None, tuple[str, ...]]:
    """The folder to open first and the components below it for an evidence root.

    A drive (``D:\\``), a UNC share (``\\\\server\\share\\``) or ``/`` has no name:
    it is opened itself, with zero child components. Appending its empty name
    would ask for a component ``""`` (CONFIRM-taskmd-rewards-R9-3).
    """
    if not root.name:
        return None, ()
    return root.parent, (root.name,)


class _PosixDirectory:
    def __init__(self, fd: int):
        self.fd = fd

    def names(self) -> list[str]:
        return os.listdir(self.fd)

    def open_file(self, name: str) -> BinaryIO:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("evidence requires a single filename")
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError("evidence must be a regular file")
            return os.fdopen(fd, "rb")
        except BaseException:
            os.close(fd)
            raise


@contextmanager
def open_directory(
    directory: Path, *, evidence_root: Path | None = None
) -> Iterator[_PosixDirectory | _WindowsDirectory]:
    anchor, parts = _location(directory, evidence_root)
    if not _posix_available():
        if not _is_windows():
            raise OSError(
                "safe evidence reads require no-follow filesystem handles; use a supported Windows, Linux or macOS controller"
            )
        api = _WindowsAPI()
        handles = []
        here = anchor
        try:
            handles.append(api.open(here, directory=True))
            for part in parts:
                _windows_component(part)
                handles.append(api.open(Path(part), directory=True, parent=handles[-1]))
            yield _WindowsDirectory(handles[-1], api)
        finally:
            for handle in reversed(handles):
                api.close(handle)
        return
    folder = os.open(anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=folder
            )
            os.close(folder)
            folder = child
        yield _PosixDirectory(folder)
    finally:
        os.close(folder)


@contextmanager
def open_file(path: Path, *, evidence_root: Path | None = None) -> Iterator[BinaryIO]:
    with (
        open_directory(path.parent, evidence_root=evidence_root) as directory,
        directory.open_file(path.name) as source,
    ):
        yield source


class _WindowsAPI:
    """Open the reparse object itself and deny write/delete sharing.

    Child opens and enumeration use the directory handle, never its old path.
    Renames or reparse metadata changes cannot redirect an ancestor lookup.
    """

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        class FileInfo(ctypes.Structure):
            _fields_ = [
                ("attributes", wintypes.DWORD),
                ("created", wintypes.FILETIME),
                ("accessed", wintypes.FILETIME),
                ("written", wintypes.FILETIME),
                ("volume", wintypes.DWORD),
                ("size_high", wintypes.DWORD),
                ("size_low", wintypes.DWORD),
                ("links", wintypes.DWORD),
                ("index_high", wintypes.DWORD),
                ("index_low", wintypes.DWORD),
            ]

        loader = _platform_member(ctypes, "WinDLL")
        kernel = loader("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.GetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(FileInfo),
        ]
        kernel.GetFileInformationByHandle.restype = wintypes.BOOL
        kernel.GetFileType.argtypes = [wintypes.HANDLE]
        kernel.GetFileType.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel = kernel
        self.info_type = FileInfo

        class UnicodeString(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_uint16),
                ("maximum", ctypes.c_uint16),
                ("buffer", ctypes.c_wchar_p),
            ]

        class ObjectAttributes(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_uint32),
                ("root", wintypes.HANDLE),
                ("name", ctypes.POINTER(UnicodeString)),
                ("attributes", ctypes.c_uint32),
                ("security", ctypes.c_void_p),
                ("qos", ctypes.c_void_p),
            ]

        class IOStatus(ctypes.Structure):
            _fields_ = [("status", ctypes.c_void_p), ("information", ctypes.c_size_t)]

        self.unicode_type = UnicodeString
        self.object_type = ObjectAttributes
        self.io_type = IOStatus
        self.nt = loader("ntdll")
        self.nt.NtCreateFile.argtypes = [
            ctypes.POINTER(wintypes.HANDLE),
            ctypes.c_uint32,
            ctypes.POINTER(ObjectAttributes),
            ctypes.POINTER(IOStatus),
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.nt.NtCreateFile.restype = ctypes.c_int32
        self.nt.NtQueryDirectoryFile.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(IOStatus),
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_ubyte,
            ctypes.c_void_p,
            ctypes.c_ubyte,
        ]
        self.nt.NtQueryDirectoryFile.restype = ctypes.c_int32

    def open(self, path: Path, *, directory: bool, parent: int | None = None) -> int:
        import ctypes
        from ctypes import wintypes

        # GENERIC_READ, FILE_SHARE_READ only, OPEN_EXISTING,
        # FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT.
        if parent is None:
            handle = self.kernel.CreateFileW(
                str(path), 0x80000000, 1, None, 3, 0x02200000, None
            )
            if handle == ctypes.c_void_p(-1).value:
                raise _platform_member(ctypes, "WinError")()
        else:
            name = str(path)
            _windows_component(name)
            length = len(name.encode("utf-16-le"))
            string = self.unicode_type(length, length, name)
            attrs = self.object_type(
                ctypes.sizeof(self.object_type),
                parent,
                ctypes.pointer(string),
                0x40,
                None,
                None,
            )
            io = self.io_type()
            opened = wintypes.HANDLE()
            status = self.nt.NtCreateFile(
                ctypes.byref(opened),
                0x80100000,
                ctypes.byref(attrs),
                ctypes.byref(io),
                None,
                0,
                1,
                1,
                0x00200020 | (1 if directory else 0x40),
                None,
                0,
            )
            if status != 0:
                raise OSError(
                    f"no-follow evidence open failed (NTSTATUS {status & 0xFFFFFFFF:#x})"
                )
            handle = opened.value
        if handle is None:
            raise OSError("no-follow evidence open returned no handle")
        try:
            info = self.info_type()
            if not self.kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
                raise _platform_member(ctypes, "WinError")()
            if (
                info.attributes & 0x400
                or bool(info.attributes & 0x10) != directory
                or self.kernel.GetFileType(handle) != 1
            ):
                raise OSError(
                    "evidence must be a disk file or directory without reparse points"
                )
            return handle
        except BaseException:
            self.close(handle)
            raise

    def names(self, handle: int) -> list[str]:
        import ctypes

        names: list[str] = []
        buffer = ctypes.create_string_buffer(65536)
        restart = True
        while True:
            io = self.io_type()
            status = self.nt.NtQueryDirectoryFile(
                handle,
                None,
                None,
                None,
                ctypes.byref(io),
                buffer,
                len(buffer),
                12,
                False,
                None,
                restart,
            )
            restart = False
            if status & 0xFFFFFFFF == 0x80000006:  # STATUS_NO_MORE_FILES
                return names
            if status != 0 or not 12 <= io.information <= len(buffer):
                raise OSError("no-follow evidence directory enumeration failed")
            data = buffer.raw[: io.information]
            offset = 0
            while True:
                next_entry, _, length = struct.unpack_from("<III", data, offset)
                if length % 2 or offset + 12 + length > len(data):
                    raise OSError("invalid evidence directory entry")
                name = data[offset + 12 : offset + 12 + length].decode("utf-16-le")
                if name not in (".", ".."):
                    names.append(name)
                if not next_entry:
                    break
                if next_entry < 12 + length or offset + next_entry + 12 > len(data):
                    raise OSError("invalid evidence directory entry offset")
                offset += next_entry

    def close(self, handle: int) -> None:
        self.kernel.CloseHandle(handle)

    def file(self, handle: int) -> BinaryIO:
        import msvcrt

        # Ownership transfers to the CRT descriptor only on successful conversion.
        try:
            fd = _platform_member(msvcrt, "open_osfhandle")(
                handle, os.O_RDONLY | _platform_member(os, "O_BINARY")
            )
        except BaseException:
            self.close(handle)
            raise
        try:
            return os.fdopen(fd, "rb")
        except BaseException:
            os.close(fd)
            raise


def _windows_component(name: str) -> None:
    # Refuse path aliases, alternate data streams and Win32 device names.
    stem = name.split(".", 1)[0].rstrip(" .").upper()
    if (
        not name
        or name in (".", "..")
        or any(c in name for c in '\x00\\/:*?"<>|')
        or name[-1] in " ."
        or stem in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
        or stem
        in {
            f"{prefix}{digit}"
            for prefix in ("COM", "LPT")
            for digit in "123456789\u00b9\u00b2\u00b3"
        }
    ):
        raise ValueError("evidence contains a Windows path alias")


class _WindowsDirectory:
    def __init__(self, handle: int, api: _WindowsAPI):
        self.handle = handle
        self.api = api

    def names(self) -> list[str]:
        return self.api.names(self.handle)

    def open_file(self, name: str) -> BinaryIO:
        _windows_component(name)
        return self.api.file(
            self.api.open(Path(name), directory=False, parent=self.handle)
        )


def _posix_available() -> bool:
    return (
        all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"))
        and os.open in os.supports_dir_fd
        and os.listdir in os.supports_fd
    )


def _is_windows() -> bool:
    return os.name == "nt"
