"""A task keeps one Daytona build key on every machine.

Daytona keys a build by its Dockerfile text and one hash per COPY source,
which the SDK computes (``ObjectStorage._compute_hash_for_path_md5``, sync
and async) by walking a folder in ``os.walk`` order: the filesystem's own
listing order, which differs between ext4 and APFS. The same task got one
key on a laptop and another on a CI runner, and was built again on each.

Guards the fix for the gap a benchmark's own infrastructure works around as
a patch. The listing order is simulated by wrapping ``os.walk``; the
SDK's own functions are the reference, and the real SDK classes are put
back after each test.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from pathlib import Path

import pytest

pytest.importorskip("daytona")

from daytona._async.object_storage import AsyncObjectStorage
from daytona._async.process import AsyncProcess
from daytona._sync.object_storage import ObjectStorage

from benchflow.sandbox import _sdk_ops

# The SDK's own functions, also when an earlier test's apply() replaced them.
_SDK_SYNC = inspect.unwrap(ObjectStorage._compute_hash_for_path_md5)
_SDK_ASYNC = inspect.unwrap(AsyncObjectStorage._compute_hash_for_path_md5)
_REAL_WALK = os.walk


def _listing(order: str):
    """``os.walk`` as a filesystem that lists each folder in *order*."""

    def walk(top, *args, **kwargs):
        for root, dirs, files in _REAL_WALK(top, *args, **kwargs):
            dirs.sort(reverse=order == "reversed")
            files.sort(reverse=order == "reversed")
            yield root, dirs, files

    return walk


@pytest.fixture
def context(tmp_path) -> Path:
    """A COPY folder: files, a nested folder and an empty one."""
    root = tmp_path / "context"
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "empty-b").mkdir()
    (root / "empty-a").mkdir()
    for name, data in {
        "a.txt": b"alpha",
        "b.txt": b"beta",
        "pkg/__init__.py": b"",
        "pkg/z.py": b"z" * 9000,
        "pkg/sub/m.py": b"m",
    }.items():
        (root / name).write_bytes(data)
    return root


@pytest.fixture(autouse=True)
def sdk_restored(monkeypatch):
    """The SDK's own context hashes during the test, and every SDK
    attribute apply() patches back as it was afterwards."""
    monkeypatch.setattr(ObjectStorage, "_compute_hash_for_path_md5", _SDK_SYNC)
    monkeypatch.setattr(AsyncObjectStorage, "_compute_hash_for_path_md5", _SDK_ASYNC)
    for name in ("_open_ws", "get_session_command_logs"):
        monkeypatch.setattr(AsyncProcess, name, getattr(AsyncProcess, name))
    monkeypatch.setattr(_sdk_ops, "_PATCHED", _sdk_ops._PATCHED)


@pytest.fixture
def build_keys():
    """Imported here, so this file also runs where the module is missing."""
    return importlib.import_module("benchflow.sandbox.daytona_build_keys")


def _storage(cls):
    return object.__new__(cls)


async def _hashes(monkeypatch, path: str, archive: str | None) -> dict[str, str]:
    """The installed sync and async hashes of *path*, under each listing."""
    out = {}
    for order in ("sorted", "reversed"):
        monkeypatch.setattr(os, "walk", _listing(order))
        out[f"sync-{order}"] = ObjectStorage._compute_hash_for_path_md5(
            _storage(ObjectStorage), path, archive
        )
        out[f"async-{order}"] = await AsyncObjectStorage._compute_hash_for_path_md5(
            _storage(AsyncObjectStorage), path, archive
        )
    monkeypatch.setattr(os, "walk", _REAL_WALK)
    return out


async def test_a_folder_gets_one_key_whatever_the_listing_order(monkeypatch, context):
    """After apply(), the sync and async hashes agree under both listings,
    and equal the SDK's own hash on a filesystem that lists sorted."""
    before = await _hashes(monkeypatch, str(context), "context")
    assert before["sync-sorted"] != before["sync-reversed"], "the SDK's own walk"

    _sdk_ops.apply()
    after = await _hashes(monkeypatch, str(context), "context")

    assert set(after.values()) == {before["sync-sorted"]}
    assert before["async-sorted"] == before["sync-sorted"]


async def test_a_file_source_keeps_the_sdks_key(context):
    _sdk_ops.apply()
    for path, archive in ((context / "a.txt", "a.txt"), (context / "pkg/z.py", None)):
        assert ObjectStorage._compute_hash_for_path_md5(
            _storage(ObjectStorage), str(path), archive
        ) == _SDK_SYNC(_storage(ObjectStorage), str(path), archive)
        assert await AsyncObjectStorage._compute_hash_for_path_md5(
            _storage(AsyncObjectStorage), str(path), archive
        ) == await _SDK_ASYNC(_storage(AsyncObjectStorage), str(path), archive)


@pytest.mark.parametrize(
    "spelling", ["absolute", "trailing-slash", "relative", "dot", "missing"]
)
@pytest.mark.parametrize("archive", [None, "", "ctx", "/abs/ctx/", "a/../b"])
def test_context_hash_is_the_sdks_hash_on_a_sorted_listing(
    monkeypatch, tmp_path, context, spelling, archive, build_keys
):
    """Path spellings and archive paths the SDK accepts give its own bytes."""
    monkeypatch.chdir(tmp_path)
    path = {
        "absolute": str(context),
        "trailing-slash": f"{context}/",
        "relative": os.path.relpath(context),
        "dot": f"{context.parent}/./{context.name}",
        "missing": str(tmp_path / "missing"),
    }[spelling]
    monkeypatch.setattr(os, "walk", _listing("sorted"))
    expected = _SDK_SYNC(_storage(ObjectStorage), path, archive)
    monkeypatch.setattr(os, "walk", _listing("reversed"))
    assert (
        build_keys.context_hash(path, archive, ObjectStorage.compute_archive_base_path)
        == expected
    )


def test_sorted_walk_orders_the_descent_for_a_collected_walk(
    monkeypatch, context, build_keys
):
    monkeypatch.setattr(os, "walk", _listing("reversed"))
    walked = [
        (os.path.relpath(root, context), dirs, files)
        for root, dirs, files in list(build_keys.sorted_walk(str(context)))
    ]
    assert [root for root, _, _ in walked] == [
        ".",
        "empty-a",
        "empty-b",
        "pkg",
        os.path.join("pkg", "sub"),
    ]
    assert walked[0][1:] == (["empty-a", "empty-b", "pkg"], ["a.txt", "b.txt"])


def test_another_sdk_source_keeps_the_sdks_walk(monkeypatch, caplog):
    """A changed SDK function is left alone in both classes, said once."""

    class ChangedSync(ObjectStorage):
        def _compute_hash_for_path_md5(self, path_str, archive_base_path=None):
            return "changed"

    class Async(AsyncObjectStorage):
        pass

    monkeypatch.setattr(_sdk_ops, "_context_hash_refusal_logged", False)
    with caplog.at_level(logging.WARNING, logger=_sdk_ops.logger.name):
        _sdk_ops._patch_context_hash(ChangedSync, Async)
        _sdk_ops._patch_context_hash(ChangedSync, Async)

    assert _storage(ChangedSync)._compute_hash_for_path_md5("x") == "changed"
    assert "_compute_hash_for_path_md5" not in Async.__dict__
    warnings = [r for r in caplog.records if "SDK's folder order" in r.getMessage()]
    assert len(warnings) == 1
    assert "ObjectStorage._compute_hash_for_path_md5" in warnings[0].getMessage()


def test_the_patch_is_installed_once():
    class Sync(ObjectStorage):
        pass

    class Async(AsyncObjectStorage):
        pass

    _sdk_ops._patch_context_hash(Sync, Async)
    installed = (
        Sync.__dict__["_compute_hash_for_path_md5"],
        Async.__dict__["_compute_hash_for_path_md5"],
    )
    _sdk_ops._patch_context_hash(Sync, Async)
    assert (
        Sync.__dict__["_compute_hash_for_path_md5"],
        Async.__dict__["_compute_hash_for_path_md5"],
    ) == installed
    assert installed[0].__wrapped__ is _SDK_SYNC
    assert installed[1].__wrapped__ is _SDK_ASYNC
    assert ObjectStorage._compute_hash_for_path_md5 is _SDK_SYNC
