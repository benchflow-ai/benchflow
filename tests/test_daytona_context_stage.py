"""Differential custody and SDK-key tests for context staging."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import os
import socket
import stat
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("daytona")
from daytona import Image
from daytona._async.object_storage import AsyncObjectStorage

from benchflow.sandbox import daytona_context_stage as contexts
from benchflow.sandbox.daytona import _load_daytona_sdk
from benchflow.sandbox.daytona_build_keys import context_hash, sorted_walk
from benchflow.task import staging


@pytest.fixture(autouse=True)
def sdk(monkeypatch):
    _load_daytona_sdk()
    monkeypatch.delenv("BENCHFLOW_DAYTONA_SNAPSHOTS", raising=False)
    monkeypatch.setattr(contexts, "_MIN_DEFERRED_BYTES", 64)


def task(tmp_path, recipe="COPY payload /payload\n"):
    environment = tmp_path / "task" / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM example.invalid/task\n" + recipe)
    (environment / "payload").write_bytes(bytes(range(256)) * 513)
    (environment / "payload").chmod(0o551)
    os.utime(environment / "payload", ns=(1_700_000_000, 1_800_000_000))
    return environment


async def sdk_keys(stage):
    """The keys Daytona sees: the SDK's hash with each folder listed sorted
    (daytona_build_keys.context_hash, which _sdk_ops installs)."""
    return {
        context.archive_path: context_hash(
            str(context.source_path),
            context.archive_path,
            AsyncObjectStorage.compute_archive_base_path,
        )
        for context in Image.from_dockerfile(
            stage.environment_dir / "Dockerfile"
        )._context_list
    }


def manifest(root):
    result = {}
    for path in root.rglob("*"):
        info = path.lstat()
        value = (
            os.readlink(path)
            if path.is_symlink()
            else hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else None
        )
        result[str(path.relative_to(root))] = (
            stat.S_IFMT(info.st_mode),
            stat.S_IMODE(info.st_mode),
            # Ordinary staging recreates links with the current timestamp.
            None if path.is_symlink() else info.st_mtime_ns,
            value,
        )
    return result


@pytest.mark.parametrize(
    "case",
    ["internal-links", "hard-links", "empty", "large", "newline", "nonutf8", "order"],
)
async def test_source_keys_equal_original_copy_and_completed_tree(tmp_path, case):
    """Native SDK order, link behavior, names and bytes agree."""
    environment = task(tmp_path, "COPY . /workspace\n")
    if case == "internal-links":
        (environment / "link").symlink_to(environment / "payload")
        nested = environment / "nested"
        nested.mkdir()
        (nested / "link").symlink_to("../payload")
        (environment / "directory-link").symlink_to("nested")
    elif case == "hard-links":
        os.link(environment / "payload", environment / "hardlink")
    elif case == "empty":
        (environment / "empty").touch()
        (environment / "empty-directory").mkdir()
    elif case == "large":
        (environment / "large").write_bytes(b"large" * 500_000)
    elif case == "newline":
        (environment / "file\nname").write_bytes(b"newline" * 100)
    elif case == "nonutf8":
        (environment / os.fsdecode(b"non-utf8-\xff")).write_bytes(b"bytes" * 100)
    else:
        for index in reversed(range(400)):
            (environment / f"entry-{index:04d}").write_bytes(str(index).encode())
    old = staging.stage_environment(environment)
    new = contexts.stage_daytona_environment(environment)
    try:
        assert (
            Image.from_dockerfile(old.environment_dir / "Dockerfile").dockerfile()
            == Image.from_dockerfile(new.environment_dir / "Dockerfile").dockerfile()
        )
        if case == "nonutf8":
            errors = []
            for stage in (old, new):
                with pytest.raises(UnicodeEncodeError) as caught:
                    await sdk_keys(stage)
                errors.append(caught.value.args)
            assert errors[0] == errors[1]
            assert not new.pending
        else:
            expected = await sdk_keys(old)
            assert {
                archive: value for (_, archive), value in new.keys.items()
            } == expected
            assert (new.environment_dir / "payload").stat().st_size == 0
            new.materialize()
            assert await sdk_keys(new) == expected
        assert manifest(new.task_dir) == manifest(old.task_dir)
    finally:
        old.remove()
        new.remove()


@pytest.mark.parametrize("kind", ["escape", "fifo", "socket", "device"])
def test_refusals_equal_original_copy(tmp_path, monkeypatch, kind):
    """Every original refusal and diagnostic is retained."""
    environment = task(tmp_path)
    special = environment / "bad"
    listener = None
    if kind == "escape":
        special.symlink_to(tmp_path / "operator-file")
    elif kind == "fifo":
        os.mkfifo(special)
    elif kind == "socket":
        listener = socket.socket(socket.AF_UNIX)
        listener.bind(str(special))
    else:
        special.touch()
        original_stat = staging.os.stat

        def device(path, *args, **kwargs):
            result = original_stat(path, *args, **kwargs)
            if path == "bad" and kwargs.get("dir_fd") is not None:
                fields = list(result)
                fields[0] = stat.S_IFCHR | 0o600
                return os.stat_result(fields)
            return result

        monkeypatch.setattr(staging.os, "stat", device)
    errors = []
    try:
        for stage in (staging.stage_environment, contexts.stage_daytona_environment):
            with pytest.raises(staging.StagingRefused) as caught:
                stage(environment)
            errors.append(
                (str(caught.value), caught.value.refused, caught.value.transient)
            )
        assert errors[0] == errors[1]
    finally:
        if listener is not None:
            listener.close()


async def test_hit_keeps_captured_key_without_reopening_source(tmp_path, monkeypatch):
    """A hit consumes the frozen identity, like a private copy."""
    environment = task(tmp_path)
    old = staging.stage_environment(environment)
    stage = contexts.stage_daytona_environment(environment)
    storage = object.__new__(AsyncObjectStorage)
    storage._file_exists_in_s3 = AsyncMock(return_value=True)
    storage._upload_as_tar = AsyncMock()
    try:
        expected = (await sdk_keys(old))["payload"]
        (environment / "payload").chmod(0o751)
        (environment / "payload").write_bytes(b"changed after capture")
        monkeypatch.setattr(
            stage, "source", lambda *_: pytest.fail("hit reread source")
        )
        with contexts.staged_contexts(stage):
            key = await storage.upload(
                str(stage.environment_dir / "payload"), "org", "payload"
            )
        assert key == expected
        storage._file_exists_in_s3.assert_awaited_once_with(
            f"org/{expected}/context.tar"
        )
        storage._upload_as_tar.assert_not_awaited()
        assert (stage.environment_dir / "payload").stat().st_size == 0
    finally:
        old.remove()
        stage.remove()


async def test_miss_upload_matches_original_copy_bytes_and_metadata(tmp_path):
    """A miss uploads the same complete tree under the same key."""
    environment = task(tmp_path, "COPY . /workspace\n")
    (environment / "alias").symlink_to("payload")
    old = staging.stage_environment(environment)
    new = contexts.stage_daytona_environment(environment)
    uploads = []
    try:
        for stage in (old, new):
            storage = object.__new__(AsyncObjectStorage)
            storage._file_exists_in_s3 = AsyncMock(return_value=False)

            async def upload(key, path, archive):
                uploads.append((key, archive, manifest(Path(path))))

            storage._upload_as_tar = upload
            context = Image.from_dockerfile(
                stage.environment_dir / "Dockerfile"
            )._context_list[0]
            with contexts.staged_contexts(getattr(stage, "daytona_contexts", None)):
                await storage.upload(context.source_path, "org", context.archive_path)
            assert storage._file_exists_in_s3.await_count == 1
        assert uploads[0] == uploads[1]
        assert not new.pending
    finally:
        old.remove()
        new.remove()


@pytest.mark.parametrize("change", ["bytes", "link", "ancestor", "fifo"])
async def test_changed_source_refuses_miss_and_never_retries_a_refusal(
    tmp_path, change
):
    """Changed bytes or path custody cannot authorize an upload."""
    environment = task(tmp_path)
    stage = contexts.stage_daytona_environment(environment)
    source = environment / "payload"
    original = source.read_bytes()
    storage = object.__new__(AsyncObjectStorage)
    storage._file_exists_in_s3 = AsyncMock(return_value=False)
    storage._upload_as_tar = AsyncMock()
    try:
        if change == "bytes":
            info = source.stat()
            source.chmod(0o751)
            source.write_bytes(b"x" * len(original))
            os.utime(source, ns=(info.st_atime_ns, info.st_mtime_ns))
        elif change == "ancestor":
            environment.rename(environment.with_name("moved"))
            environment.symlink_to(environment.with_name("moved"))
        else:
            source.unlink()
            if change == "link":
                outside = tmp_path / "outside"
                outside.write_bytes(original)
                source.symlink_to(outside)
            else:
                os.mkfifo(source)
        with contexts.staged_contexts(stage):
            with pytest.raises(
                staging.StagingRefused, match="changed while being copied; try again"
            ) as first:
                await storage.upload(
                    str(stage.environment_dir / "payload"), "org", "payload"
                )
            if change == "bytes":
                source.write_bytes(original)
            with pytest.raises(staging.StagingRefused) as second:
                await storage.upload(
                    str(stage.environment_dir / "payload"), "org", "payload"
                )
            assert second.value is first.value
        storage._upload_as_tar.assert_not_awaited()
        assert storage._file_exists_in_s3.await_count == 1
    finally:
        stage.remove()


def test_change_between_capture_and_source_hash_refuses(tmp_path, monkeypatch):
    """A second source pass must match its initial pinned read."""
    environment = task(tmp_path)
    monkeypatch.setattr(staging.tempfile, "tempdir", str(tmp_path))
    original = contexts._ContextStage.context_key

    def changed(stage, path, archive):
        (environment / "payload").chmod(0o751)
        (environment / "payload").write_bytes(b"a different payload")
        return original(stage, path, archive)

    monkeypatch.setattr(contexts._ContextStage, "context_key", changed)
    with pytest.raises(staging.StagingRefused, match="changed while being copied"):
        contexts.stage_daytona_environment(environment)
    assert not list(tmp_path.glob(f"{staging.STAGE_PREFIX}*"))


async def test_materialized_key_mismatch_is_sticky_and_never_uploads(tmp_path):
    """Completed-copy MD5 disagreement is a permanent refusal."""
    stage = contexts.stage_daytona_environment(task(tmp_path))
    storage = object.__new__(AsyncObjectStorage)
    storage._file_exists_in_s3 = AsyncMock(return_value=False)
    storage._compute_hash_for_path_md5 = AsyncMock(return_value="different")
    storage._upload_as_tar = AsyncMock()
    try:
        with contexts.staged_contexts(stage):
            for _ in range(2):
                with pytest.raises(staging.StagingRefused):
                    await storage.upload(
                        str(stage.environment_dir / "payload"), "org", "payload"
                    )
        storage._compute_hash_for_path_md5.assert_awaited_once()
        storage._upload_as_tar.assert_not_awaited()
    finally:
        stage.remove()


@pytest.mark.parametrize(
    "fallback", ["compose", "sdk", "host-access", "snapshots", "missing-dockerfile"]
)
def test_unsupported_paths_keep_the_complete_original_snapshot(
    tmp_path, monkeypatch, fallback
):
    """Unsupported consumers retain ordinary complete staging."""
    environment = task(tmp_path)
    if fallback == "compose":
        (environment / "compose.yaml").write_text(
            "services: {main: {image: example}}\n"
        )
    elif fallback == "sdk":
        monkeypatch.setattr(contexts, "version", lambda _: "unknown")
    elif fallback == "snapshots":
        monkeypatch.setenv("BENCHFLOW_DAYTONA_SNAPSHOTS", "content")
    elif fallback == "missing-dockerfile":
        (environment / "Dockerfile").unlink()
    old = staging.stage_environment(
        environment, allow_outside=fallback == "host-access"
    )
    new = contexts.stage_daytona_environment(
        environment, allow_outside=fallback == "host-access"
    )
    try:
        assert manifest(old.task_dir) == manifest(new.task_dir)
        assert not getattr(new, "pending", {})
    finally:
        old.remove()
        new.remove()


async def test_unknown_context_is_materialized_before_original_sdk_upload(tmp_path):
    """Additional recipe contexts cannot upload placeholders."""
    stage = contexts.stage_daytona_environment(task(tmp_path))
    storage = object.__new__(AsyncObjectStorage)
    storage._file_exists_in_s3 = AsyncMock(return_value=False)
    storage._upload_as_tar = AsyncMock()
    try:
        with contexts.staged_contexts(stage):
            await storage.upload(str(stage.environment_dir), "org", "other")
        assert not stage.pending
        assert (stage.environment_dir / "payload").stat().st_size > 0
        storage._upload_as_tar.assert_awaited_once()
    finally:
        stage.remove()


async def test_cancellation_keeps_staging_until_physical_copy_finishes(
    tmp_path, monkeypatch
):
    """Cancellation neither races cleanup nor waits for a slow disk."""
    stage = contexts.stage_daytona_environment(task(tmp_path))
    entered = threading.Event()
    resume = threading.Event()
    original = stage.materialize

    def blocked():
        entered.set()
        assert resume.wait(10)
        original()

    monkeypatch.setattr(stage, "materialize", blocked)
    job = asyncio.create_task(stage.complete())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        stage.release()
        assert stage.root.exists()
    finally:
        resume.set()
    for _ in range(100):
        if not stage.root.exists():
            break
        await asyncio.sleep(0.01)
    assert not stage.root.exists()


async def test_snapshot_enabled_after_staging_receives_complete_bytes(
    tmp_path, monkeypatch
):
    """A newly enabled snapshot consumer never captures placeholders."""
    from contextlib import asynccontextmanager

    from benchflow.sandbox import daytona_content_snapshots as snapshots
    from benchflow.sandbox.daytona_layers import LayeredRecipe
    from benchflow.sandbox.daytona_strategies import _DaytonaDirect

    stage = contexts.stage_daytona_environment(task(tmp_path))
    env = SimpleNamespace(
        _daytona_contexts=stage,
        _install_layers_record={},
        task_env_config=SimpleNamespace(build_timeout_sec=60),
    )
    monkeypatch.setattr(snapshots, "enabled", lambda: True)
    monkeypatch.setattr(snapshots, "owner_prefix", lambda: "test-owner")

    @asynccontextmanager
    async def captured(image):
        assert not stage.pending
        assert (stage.environment_dir / "payload").read_bytes() == (
            stage.original_environment_dir / "payload"
        ).read_bytes()
        raise RuntimeError("capture reached with complete bytes")
        yield  # pragma: no cover

    monkeypatch.setattr(snapshots, "captured_image_async", captured)
    try:
        with pytest.raises(RuntimeError, match="capture reached with complete bytes"):
            await _DaytonaDirect(env)._content_snapshot(
                None,
                None,
                None,
                LayeredRecipe(),
                build_allowance_sec=0,
                force_build=False,
                fresh=False,
            )
    finally:
        stage.remove()


async def test_copy_submission_failure_releases_the_private_tree(tmp_path, monkeypatch):
    """Failed worker creation does not strand a cleanup reader."""
    stage = contexts.stage_daytona_environment(task(tmp_path))

    def fail(**kwargs):
        raise RuntimeError("worker unavailable")

    monkeypatch.setattr(contexts, "ThreadPoolExecutor", fail)
    with pytest.raises(RuntimeError, match="worker unavailable"):
        await stage.complete()
    stage.release()
    assert not stage.root.exists()


async def test_storage_error_retains_diagnostic_and_never_materializes(tmp_path):
    """Provider errors retain the original exception and boundary."""
    stage = contexts.stage_daytona_environment(task(tmp_path))
    storage = object.__new__(AsyncObjectStorage)
    error = RuntimeError("object lookup diagnostic")
    storage._file_exists_in_s3 = AsyncMock(side_effect=error)
    storage._upload_as_tar = AsyncMock()
    try:
        with contexts.staged_contexts(stage), pytest.raises(RuntimeError) as caught:
            await storage.upload(
                str(stage.environment_dir / "payload"), "org", "payload"
            )
        assert caught.value is error
        assert (stage.environment_dir / "payload").stat().st_size == 0
        storage._upload_as_tar.assert_not_awaited()
    finally:
        stage.remove()


async def test_miss_write_error_preserves_original_staging_refusal(
    tmp_path, monkeypatch
):
    """Disk failures during deferred copying retain staging diagnostics."""
    environment = task(tmp_path)
    stage = contexts.stage_daytona_environment(environment)

    def full_disk(*args):
        raise OSError(errno.ENOSPC, "disk full")

    with monkeypatch.context() as patch:
        patch.setattr(staging, "_copy_file_data", full_disk)
        with pytest.raises(staging.StagingRefused) as old:
            staging.stage_environment(environment)
    original_open = os.open

    def cannot_write(path, flags, *args, **kwargs):
        if str(path) == str(stage.environment_dir / "payload") and flags & os.O_WRONLY:
            raise OSError(errno.ENOSPC, "disk full")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(contexts.os, "open", cannot_write)
    try:
        with pytest.raises(staging.StagingRefused) as new:
            await stage.complete()
        assert str(new.value) == str(old.value)
        assert new.value.refused == old.value.refused
        assert new.value.transient == old.value.transient
        assert new.value.__cause__.errno == old.value.__cause__.errno == errno.ENOSPC
    finally:
        stage.remove()


async def test_cache_lookup_path_check_stays_off_the_event_loop(tmp_path, monkeypatch):
    """Cache hits retain the SDK's off-loop filesystem check."""
    stage = contexts.stage_daytona_environment(task(tmp_path))
    path = str(stage.environment_dir / "payload")
    main_thread = threading.get_ident()
    checked = []
    original = os.path.exists

    def exists(candidate):
        if candidate == path:
            checked.append(threading.get_ident())
        return original(candidate)

    storage = object.__new__(AsyncObjectStorage)
    storage._file_exists_in_s3 = AsyncMock(return_value=True)
    monkeypatch.setattr(contexts.os.path, "exists", exists)
    try:
        with contexts.staged_contexts(stage):
            await storage.upload(path, "org", "payload")
        assert checked and main_thread not in checked
    finally:
        stage.remove()


def _session1_context_key(stage, path, archive):
    """Frozen key algorithm from commit f5a69b05, before path/hash
    overlap, with each folder listed sorted as the build key is
    (daytona_build_keys.sorted_walk)."""
    digest = hashlib.md5()
    absolute = os.path.abspath(path)
    if archive is None:
        archive = AsyncObjectStorage.compute_archive_base_path(path)
    digest.update(archive.encode("utf-8"))

    def read_file(filename):
        relative = os.path.relpath(os.path.realpath(filename), stage.task_dir)
        if relative in stage.pending:
            verification = hashlib.sha256()
            with stage.source(relative) as source:
                while chunk := source.read(1024 * 1024):
                    verification.update(chunk)
                    digest.update(chunk)
            if verification.digest() != stage.pending[relative].sha256:
                raise stage.changed()
        else:
            with open(filename, "rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)

    if os.path.isfile(absolute):
        read_file(absolute)
    else:
        for root, directories, files in list(sorted_walk(absolute)):
            if not directories and not files:
                digest.update(os.path.relpath(root, path).encode("utf-8"))
            for filename in files:
                full = os.path.join(root, filename)
                digest.update(os.path.relpath(full, absolute).encode("utf-8"))
                read_file(full)
    return digest.hexdigest()


@pytest.mark.parametrize(
    "case",
    [
        "links",
        "hardlinks",
        "names",
        "nonutf8",
        "reverse-walk",
        "link-root",
        "task-root",
    ],
)
def test_keys_and_captures_equal_session1_on_adversarial_trees(
    tmp_path, monkeypatch, case
):
    """F5a69b05: path/hash overlap keeps session 1 keys and captures."""
    environment = task(tmp_path, "COPY . /workspace\n")
    nested = environment / "nested"
    nested.mkdir()
    (nested / "multiblock").write_bytes(bytes(range(256)) * 12_345)
    (nested / "empty-directory").mkdir()
    (nested / "empty-file").touch()
    if case == "links":
        (nested / "alias").symlink_to("../payload")
        (environment / "alias").symlink_to(environment / "payload")
        (environment / "directory-link").symlink_to("nested")
    elif case == "hardlinks":
        os.link(environment / "payload", nested / "hardlink")
    elif case == "names":
        (nested / "line\nbreak").write_bytes(b"newline" * 37)
    elif case == "nonutf8":
        (nested / os.fsdecode(b"non-utf8-\xff")).write_bytes(b"bytes" * 37)
    elif case == "reverse-walk":
        for index in reversed(range(100)):
            (nested / f"entry-{index:03d}").write_bytes(str(index).encode())
        original_walk = os.walk

        def reordered(*args, **kwargs):
            for root, directories, files in reversed(
                list(original_walk(*args, **kwargs))
            ):
                yield root, directories[::-1], files[::-1]

        monkeypatch.setattr(contexts.os, "walk", reordered)
    elif case == "link-root":
        (environment / "directory-link").symlink_to("nested")
        (environment / "Dockerfile").write_text(
            "FROM example.invalid/task\nCOPY directory-link /workspace\n"
        )
    ordinary = staging.stage_environment(environment)
    stage = contexts.stage_daytona_environment(environment)
    try:
        for relative, payload in stage.pending.items():
            with (ordinary.task_dir / relative).open("rb") as source:
                # Session 1 captured this exact SHA-256 at the checked copy boundary.
                assert payload.sha256 == hashlib.file_digest(source, "sha256").digest()
        if case == "nonutf8":
            errors = []
            assert not stage.pending and not stage.keys
            for key in (_session1_context_key, contexts._ContextStage.context_key):
                with pytest.raises(UnicodeEncodeError) as caught:
                    key(stage, str(stage.environment_dir), "workspace")
                errors.append(caught.value.args)
            assert errors[0] == errors[1]
        else:
            assert stage.pending
            for (path, archive), key in stage.keys.items():
                assert key == _session1_context_key(stage, path, archive)
            if case == "task-root":
                path = str(stage.task_dir)
                assert stage.context_key(path, None) == _session1_context_key(
                    stage, path, None
                )
            stage.materialize()
        assert manifest(stage.task_dir) == manifest(ordinary.task_dir)
    finally:
        ordinary.remove()
        stage.remove()


async def test_hash_worker_finishes_verification_before_returning_a_key(
    tmp_path, monkeypatch
):
    """F5a69b05: overlapping MD5 cannot bypass a delayed SHA refusal."""
    stage = contexts.stage_daytona_environment(task(tmp_path))
    entered = threading.Event()
    resume = threading.Event()
    original_sha = hashlib.sha256

    class DelayedMismatch:
        def __init__(self):
            self.digest_state = original_sha()

        def update(self, chunk):
            entered.set()
            assert resume.wait(10)
            self.digest_state.update(chunk)

        def digest(self):
            return b"mismatch"

    monkeypatch.setattr(contexts.hashlib, "sha256", DelayedMismatch)
    job = asyncio.create_task(
        asyncio.to_thread(
            stage.context_key, str(stage.environment_dir / "payload"), "payload"
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert not job.done()
        resume.set()
        with pytest.raises(staging.StagingRefused, match="changed while being copied"):
            await job
    finally:
        resume.set()
        await asyncio.gather(job, return_exceptions=True)
        stage.remove()
