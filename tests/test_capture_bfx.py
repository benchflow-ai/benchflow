"""Workspace capture through ``bfx tar-out`` (TRUSTED-RUNTIME v2 §4,
capture row): the same manifest, exclusions and limits as today's capture,
and today's capture wherever bfx cannot archive the workspace completely."""

from __future__ import annotations

import contextlib
import io
import os
import socket
import stat
import tarfile
from pathlib import Path
from typing import Any

import pytest

from benchflow._bfx.client import BfxAmbiguous, BfxError, BfxNotStarted
from benchflow._bfx.locate import BfxUnavailable
from benchflow.review.evidence import (
    EvidenceError,
    EvidenceLimitError,
    capture_workspace,
)
from tests.test_evidence_without_python import LocalSandbox, _entries, _tree


def _tar_out(root: Path, args: dict | None = None) -> tuple[bytes, dict[str, Any]]:
    """What bfx tar-out writes for ``{root, paths: [""]}``: every entry
    beneath *root*, no link followed, no mount entered, special files
    skipped and listed."""
    args = args or {}
    buffer = io.BytesIO()
    report: dict[str, Any] = {"entries": 0, "skipped": [], "mounts": [], "changed": []}
    device = os.lstat(root).st_dev
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:

        def visit(path: Path, rel: str) -> None:
            st = os.lstat(path)
            if st.st_dev != device:
                report["mounts"].append(rel)
                return
            info = tarfile.TarInfo(rel)
            info.mode, info.mtime = stat.S_IMODE(st.st_mode), int(st.st_mtime)
            if stat.S_ISDIR(st.st_mode):
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
                if args.get("recursive", True):
                    for child in sorted(os.listdir(path)):
                        visit(path / child, f"{rel}/{child}")
            elif stat.S_ISLNK(st.st_mode):
                info.type, info.linkname = tarfile.SYMTYPE, os.readlink(path)
                tar.addfile(info)
            elif stat.S_ISREG(st.st_mode):
                data = path.read_bytes()
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                report["skipped"].append({"path": rel, "type": "special"})
                return
            report["entries"] += 1

        for path in args.get("paths", [""]):
            if path:
                visit(root / path, path)
            else:
                for child in sorted(os.listdir(root)):
                    visit(root / child, child)
    return buffer.getvalue(), report


class BfxSandbox(LocalSandbox):
    """LocalSandbox with bfx delivered: tar-out reads this machine."""

    bfx_delivery = "bind"

    def __init__(self, *, failure: BaseException | None = None, **kwargs) -> None:
        super().__init__(python=True, **kwargs)
        self.failure = failure
        self.tar_outs: list[dict] = []
        self.report_changes: dict[str, Any] = {}

    async def bfx_call(self, command, args, *, timeout_sec: int):
        assert command == "scan"
        if self.failure is not None:
            raise self.failure
        entries = []
        errors = []
        for walk in args["walks"]:
            root = Path(walk["root"])
            # As bfx/src/scan.rs answers a walk: a root that is no directory
            # is an error of the walk, spelled by errno_name; a missing root
            # is an entry that does not exist.
            if not root.exists() and not root.is_symlink():
                entries.append({"path": str(root), "exists": False})
                continue
            if not root.is_dir() or root.is_symlink():
                errors.append(
                    {
                        "path": str(root),
                        "op": "walk",
                        "error": "link" if root.is_symlink() else "not-directory",
                    }
                )
                continue
            paths = [root, *sorted(root.iterdir())] if walk["depth"] else [root]
            for path in paths:
                info = path.lstat()
                kind = (
                    "link"
                    if stat.S_ISLNK(info.st_mode)
                    else "dir"
                    if stat.S_ISDIR(info.st_mode)
                    else "file"
                    if stat.S_ISREG(info.st_mode)
                    else "fifo"
                )
                item = {
                    "path": str(path),
                    "exists": True,
                    "type": kind,
                    "size": info.st_size,
                    "mode": stat.S_IMODE(info.st_mode),
                }
                if path == root:
                    item["listing"] = True
                if kind == "link":
                    item["link"] = os.readlink(path)
                entries.append(item)
        if len(entries) > args["max_entries"]:
            raise BfxError("too-many", "metadata entries exceed capture limit")
        return {"entries": entries, "errors": errors}

    async def bfx_tar_out(self, args, destination: Path, *, timeout_sec: int):
        self.tar_outs.append(dict(args))
        if self.failure is not None:
            raise self.failure
        data, report = _tar_out(Path(args["root"]), args)
        with open(destination, "xb") as out:
            out.write(data)
        report.update(self.report_changes)
        return report


async def test_bfx_capture_gives_the_same_manifest_as_the_python_capture(tmp_path):
    workspace = _tree(tmp_path / "box").resolve()
    before = await capture_workspace(
        LocalSandbox(python=True), str(workspace), tmp_path / "a", exclude=["*.tmp"]
    )
    box = BfxSandbox()
    after = await capture_workspace(
        box, str(workspace), tmp_path / "b", exclude=["*.tmp"]
    )
    assert len(box.tar_outs) == 1
    assert box.tar_outs[0]["root"] == str(workspace)
    assert box.tar_outs[0]["recursive"] is False
    assert ".codex/auth.json" not in box.tar_outs[0]["paths"]
    assert not any("python3" in c for c in box.commands)
    assert after.workspace == before.workspace
    assert _entries(after) == _entries(before)
    assert sorted(x.model_dump_json() for x in after.exclusions) == sorted(
        x.model_dump_json() for x in before.exclusions
    )
    assert (tmp_path / "b" / "workspace" / "hardlink.txt").read_text() == "42\n"
    assert os.access(tmp_path / "b" / "workspace" / "run.sh", os.X_OK)
    assert not (tmp_path / "b" / "workspace" / ".codex" / "auth.json").exists()


async def test_special_files_bfx_skips_are_recorded_as_exclusions(tmp_path):
    workspace = _tree(tmp_path / "box").resolve()
    os.mkfifo(workspace / "pipe")
    with socket.socket(socket.AF_UNIX) as sock:
        with contextlib.suppress(OSError):
            sock.bind(str(workspace / "sock"))
        manifest = await capture_workspace(BfxSandbox(), str(workspace), tmp_path / "b")
    reasons = {e.original_path: e.reason for e in manifest.exclusions}
    assert reasons[str(workspace / "pipe")] == "special_file"


async def test_an_escaping_link_is_recorded_not_kept(tmp_path):
    workspace = _tree(tmp_path / "box").resolve()
    (workspace / "out").symlink_to("/etc/hostname")
    manifest = await capture_workspace(BfxSandbox(), str(workspace), tmp_path / "b")
    escaped = [e for e in manifest.exclusions if e.reason == "symlink_escape"]
    assert [(e.original_path, e.link_target) for e in escaped] == [
        (str(workspace / "out"), "/etc/hostname")
    ]
    assert not (tmp_path / "b" / "workspace" / "out").is_symlink()


async def test_limits_hold_on_the_bfx_capture(tmp_path):
    workspace = _tree(tmp_path / "box").resolve()
    with pytest.raises(EvidenceLimitError):
        await capture_workspace(
            BfxSandbox(), str(workspace), tmp_path / "b", max_entries=2
        )
    assert not (tmp_path / "b").exists()


async def test_a_file_changed_while_archived_fails_the_capture(tmp_path):
    workspace = _tree(tmp_path / "box").resolve()
    box = BfxSandbox()
    box.report_changes = {"changed": ["answer.txt"]}
    with pytest.raises(EvidenceError, match="changed while it was captured"):
        await capture_workspace(box, str(workspace), tmp_path / "b")
    assert not (tmp_path / "b").exists()


async def test_a_workspace_holding_a_mount_is_captured_as_before(tmp_path):
    workspace = _tree(tmp_path / "box").resolve()
    box = BfxSandbox()
    box.report_changes = {"mounts": ["data"]}
    manifest = await capture_workspace(box, str(workspace), tmp_path / "b")
    assert len(box.tar_outs) == 1
    assert any("python3" in c for c in box.commands)
    assert manifest.workspace == str(workspace)


@pytest.mark.parametrize(
    "failure",
    [
        BfxUnavailable("not delivered"),
        BfxNotStarted("not-started", "withdrawn"),
        BfxError("link", "a link in the path"),
        BfxError("not-dir", "a single file"),
        BfxError("mount", "another mount"),
        BfxError("missing", "no such path"),
    ],
)
async def test_where_bfx_cannot_archive_the_workspace_today_capture_runs(
    tmp_path, failure
):
    workspace = _tree(tmp_path / "box").resolve()
    box = BfxSandbox(failure=failure)
    manifest = await capture_workspace(box, str(workspace), tmp_path / "b")
    assert any("python3" in c for c in box.commands)
    assert manifest.workspace == str(workspace)


async def test_a_single_file_bfx_scans_as_not_a_directory_is_captured(tmp_path):
    """A declared artifact that is one file (Terminal-Bench 4.1's
    `artifacts = ["/app/cracker.py"]`): bfx's scan answers its root with a
    `not-directory` error, and today's capture takes the file."""
    workspace = _tree(tmp_path / "box").resolve()
    box = BfxSandbox()
    manifest = await capture_workspace(
        box, str(workspace / "answer.txt"), tmp_path / "b"
    )
    assert [(e.kind, e.size) for e in manifest.entries] == [("file", 3)]
    assert any("python3" in c for c in box.commands)
    assert box.tar_outs == []


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        (BfxError("too-large", "over the limit"), "failed in bfx"),
        (BfxAmbiguous("channel died", 0, True), "no bfx answer"),
    ],
)
async def test_other_bfx_failures_fail_the_capture(tmp_path, failure, match):
    workspace = _tree(tmp_path / "box").resolve()
    box = BfxSandbox(failure=failure)
    with pytest.raises(EvidenceError, match=match):
        await capture_workspace(box, str(workspace), tmp_path / "b")
    assert not any("python3" in c for c in box.commands)


async def test_without_bfx_delivery_today_capture_runs(tmp_path):
    workspace = _tree(tmp_path / "box").resolve()
    box = BfxSandbox()
    box.bfx_delivery = None
    await capture_workspace(box, str(workspace), tmp_path / "b")
    assert box.tar_outs == []


async def test_a_system_folder_is_refused(tmp_path):
    with pytest.raises(EvidenceError, match="system directory"):
        await capture_workspace(BfxSandbox(), "/etc", tmp_path / "b")
