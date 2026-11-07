"""The review's 400 source/archive combinations keep SDK cache keys."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("daytona")

from daytona._async.object_storage import AsyncObjectStorage

from benchflow.sandbox import daytona_context
from benchflow.sandbox.daytona_build_keys import context_hash


async def _SDK_HASH(self, path_str, archive_base_path=None):
    """The SDK's context hash as BenchFlow installs it (#35, _sdk_ops): the
    SDK's function line for line with each folder listed sorted, so a key is
    the same on every filesystem (daytona_build_keys.context_hash)."""
    return context_hash(path_str, archive_base_path, self.compute_archive_base_path)


_ARCHIVES = [
    None,
    "",
    ".",
    "..",
    "/",
    "//a//b/",
    "context/seed",
    r"C:\path\file",
    "é/文件",
    "a/../b",
]


@pytest.mark.parametrize("archive", _ARCHIVES)
@pytest.mark.parametrize("form", ["absolute", "slash", "dot", "relative", "path"])
@pytest.mark.parametrize(
    "source",
    [
        "",
        "tree",
        "tree/a",
        "tree/empty",
        "file-link",
        "dir-link",
        "dangling",
        "missing",
    ],
)
async def test_review_matrix_matches_sdk(tmp_path, monkeypatch, source, form, archive):
    """Review: all 400 path forms match the pinned SDK or its error."""
    root = tmp_path / "context"
    (root / "tree/empty").mkdir(parents=True)
    (root / "tree/nested").mkdir()
    (root / "tree/nested/é").write_bytes(b"\0" * 8193)
    (root / "tree/a").write_bytes(b"abc")
    for name, target in [
        ("file-link", "tree/a"),
        ("dir-link", "tree"),
        ("dangling", "missing"),
    ]:
        (root / name).symlink_to(target)
    path = str(root / source)
    path = {
        "absolute": path,
        "slash": path + "/",
        "dot": str(Path(path).parent) + "/./" + Path(path).name,
        "relative": os.path.relpath(path),
        "path": Path(path),
    }[form]
    storage = object.__new__(AsyncObjectStorage)
    monkeypatch.setattr(AsyncObjectStorage, "_compute_hash_for_path_md5", _SDK_HASH)

    async def outcome():
        try:
            return ("hash", await storage._compute_hash_for_path_md5(path, archive))
        except Exception as exc:
            return (type(exc), exc.args, getattr(exc, "filename", None))

    expected = await outcome()
    daytona_context.patch_context_hashing()
    assert await outcome() == expected
