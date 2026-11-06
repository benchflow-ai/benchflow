"""Production-boundary regressions from the review of commit 259b33be."""

from __future__ import annotations

import errno
import io
import os
import stat
import tarfile
from contextlib import asynccontextmanager, contextmanager, suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("daytona")
from daytona import Image
from daytona._async.object_storage import AsyncObjectStorage

from benchflow.sandbox import daytona_context_stage as contexts
from benchflow.sandbox.daytona import _load_daytona_sdk
from benchflow.sandbox.daytona_build_keys import context_hash
from benchflow.sandbox.setup import _create_sandbox_environment, _stage_environment
from benchflow.task import Task, staging
from benchflow.task.env import TaskEnvNotAllowedError
from benchflow.task.paths import RolloutPaths


@pytest.fixture(autouse=True)
def sdk(monkeypatch, tmp_path):
    _load_daytona_sdk()
    monkeypatch.delenv("BENCHFLOW_DAYTONA_SNAPSHOTS", raising=False)
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    monkeypatch.setattr(staging.tempfile, "tempdir", str(tmp_path))


def task(root, recipe="COPY payload /payload\n"):
    environment = root / "named-task/environment"
    environment.mkdir(parents=True)
    (environment.parent / "task.toml").write_text('version = "1.0"\n')
    (environment.parent / "instruction.md").write_text("Local staging test.\n")
    (environment / "Dockerfile").write_text("FROM example.invalid/task\n" + recipe)
    (environment / "payload").write_bytes(bytes(range(256)) * 1025)
    (environment / "payload").chmod(0o551)
    os.utime(environment / "payload", ns=(1700000000123456789, 1700000001987654321))
    return environment


def sandbox_for(environment):
    return _create_sandbox_environment(
        "daytona",
        Task(environment.parent),
        environment.parent,
        "context-review",
        RolloutPaths(environment.parent.parent / "rollout"),
    )


def context_for(stage):
    return Image.from_dockerfile(stage.environment_dir / "Dockerfile")._context_list[0]


async def sdk_key(context):
    """The key Daytona sees: the SDK's hash with each folder listed sorted
    (daytona_build_keys.context_hash, installed over the SDK's by _sdk_ops),
    so it is the same on every filesystem."""
    return context_hash(
        str(context.source_path),
        context.archive_path,
        AsyncObjectStorage.compute_archive_base_path,
    )


class Store:
    def __init__(self, hit=False, error=None):
        self.hit = hit
        self.error = error
        self.heads = []
        self.uploads = []

    async def head_async(self, key):
        self.heads.append(key)
        if self.error is not None:
            raise self.error
        if not self.hit:
            raise FileNotFoundError(key)

    async def put_async(self, key, stream):
        self.uploads.append((key, b"".join([part async for part in stream])))


def storage_for(hit=False, error=None):
    storage = object.__new__(AsyncObjectStorage)
    storage.store = Store(hit, error)
    return storage


async def upload(stage, storage):
    context = context_for(stage)
    with contexts.staged_contexts(getattr(stage, "daytona_contexts", None)):
        return await storage.upload(context.source_path, "org", context.archive_path)


def archive_metadata(data):
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        return {
            member.name: (member.get_info(), member.pax_headers)
            for member in archive.getmembers()
        }


@pytest.mark.parametrize("hit", [True, False])
async def test_setup_and_start_use_real_threshold_and_sdk_key(tmp_path, hit):
    """Guards 259b33be: setup/start wiring and the 64 KiB boundary matter."""
    environment = task(tmp_path, "COPY . /workspace\n")
    sizes = (0, 32768, 65535, 65536, 65537, 3 * 1024 * 1024 + 1)
    for size in sizes:
        (environment / str(size)).write_bytes(b"z" * size)
    ordinary = staging.stage_environment(environment)
    sandbox = sandbox_for(environment)
    stage = getattr(sandbox, "_daytona_contexts", None)
    try:
        assert stage is not None
        assert set(stage.pending) == {
            "environment/payload",
            "environment/65536",
            "environment/65537",
            f"environment/{3 * 1024 * 1024 + 1}",
        }
        for size in sizes:
            assert (stage.environment_dir / str(size)).stat().st_size == (
                0 if size >= 65536 else size
            )
        expected = await sdk_key(context_for(ordinary))
        storage = storage_for(hit)

        async def exercise_sdk(force_build):
            context = context_for(stage)
            return await storage.upload(
                context.source_path, "org", context.archive_path
            )

        sandbox._strategy = SimpleNamespace(start=exercise_sdk)
        assert await sandbox.start(False) == expected
        assert contexts._CURRENT.get() is None
        assert storage.store.heads == [f"org/{expected}/context.tar"]
        assert len(storage.store.uploads) == (0 if hit else 1)
        if hit:
            assert stage.pending
            assert (stage.environment_dir / "65536").stat().st_size == 0
        else:
            assert not stage.pending
            baseline = storage_for()
            await upload(ordinary, baseline)
            assert archive_metadata(storage.store.uploads[0][1]) == archive_metadata(
                baseline.store.uploads[0][1]
            )
            assert storage.store.uploads == baseline.store.uploads
    finally:
        ordinary.remove()
        if stage is not None:
            stage.remove()


@pytest.mark.parametrize("mode", [0o4755, 0o2755, 0o6755])
@pytest.mark.parametrize("consumer", ["miss", "compose"])
async def test_deferred_archive_preserves_special_modes(tmp_path, mode, consumer):
    """Guards 259b33be: real miss/Compose tar headers and bytes match staging."""
    environment = task(tmp_path)
    (environment / "payload").chmod(mode)
    if consumer == "compose":
        (environment / "compose.yaml").write_text(
            "services: {main: {image: example.invalid/task}}\n"
        )
    ordinary = staging.stage_environment(environment)
    stage = contexts.stage_daytona_environment(environment)
    try:
        assert stat.S_IMODE((stage.environment_dir / "payload").stat().st_mode) == mode
        if consumer == "compose":
            assert not stage.pending and not stage.keys
        archives = []
        for current in (ordinary, stage):
            storage = storage_for()
            await upload(current, storage)
            archives.append(storage.store.uploads[0])
        before = archive_metadata(archives[0][1])
        after = archive_metadata(archives[1][1])
        assert before["payload"][0]["mode"] == after["payload"][0]["mode"] == mode
        assert before == after
        assert archives[0] == archives[1]
    finally:
        ordinary.remove()
        stage.remove()


@pytest.mark.parametrize("links", [False, True])
async def test_real_tar_matches_for_tree_metadata(tmp_path, monkeypatch, links):
    """Guards 259b33be with SDK tar bytes for links and unusual entries."""
    environment = task(tmp_path, "COPY . /workspace\n")
    (environment / "empty").touch()
    (environment / "emptydir").mkdir()
    (environment / "line\nbreak").write_bytes(b"x" * 65536)
    os.link(environment / "payload", environment / "hardlink")
    if links:
        (environment / "alias").symlink_to("payload")
        original_symlink = os.symlink

        def fixed_symlink(source, destination, **kwargs):
            original_symlink(source, destination, **kwargs)
            os.utime(destination, ns=(1700000000000000000,) * 2, follow_symlinks=False)

        # Ordinary staging recreates symlinks at current time. Fix creation time
        # equally in both arms; no archive headers or regular files are adjusted.
        monkeypatch.setattr(os, "symlink", fixed_symlink)
    ordinary = staging.stage_environment(environment)
    stage = contexts.stage_daytona_environment(environment)
    try:
        archives = []
        for current in (ordinary, stage):
            storage = storage_for()
            storage._compute_hash_for_path_md5 = AsyncMock(
                wraps=storage._compute_hash_for_path_md5
            )
            await upload(current, storage)
            context = context_for(current)
            storage._compute_hash_for_path_md5.assert_awaited_once_with(
                context.source_path, context.archive_path
            )
            archives.append(storage.store.uploads[0])
        assert archive_metadata(archives[0][1]) == archive_metadata(archives[1][1])
        assert archives[0] == archives[1]
    finally:
        ordinary.remove()
        stage.remove()


def ordinary_disk_error(environment, monkeypatch):
    original_copy = staging._copy_file_data

    def no_space(source, destination):
        if os.fstat(source).st_size >= 65536:
            raise OSError(errno.ENOSPC, "disk full")
        original_copy(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(staging, "_copy_file_data", no_space)
        with pytest.raises(TaskEnvNotAllowedError) as caught:
            _stage_environment(environment, environment.parent.name)
    return caught.value


def fail_deferred_write(monkeypatch):
    original_open = os.open

    def no_space(path, flags, *args, **kwargs):
        if (
            str(path).endswith("/environment/payload")
            and flags & os.O_WRONLY
            and not flags & os.O_CREAT
        ):
            raise OSError(errno.ENOSPC, "disk full")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", no_space)


def assert_same_diagnostic(before, after):
    assert type(after) is type(before)
    assert str(after) == str(before)
    assert after.refused == before.refused
    assert after.__suppress_context__


@pytest.mark.parametrize("boundary", ["stage", "setup"])
def test_compose_disk_failure_retains_public_diagnostic(
    tmp_path, monkeypatch, boundary
):
    """Guards 259b33be: synchronous Compose fallback keeps the public remedy."""
    environment = task(tmp_path)
    (environment / "compose.yaml").write_text(
        "services: {main: {image: example.invalid/task}}\n"
    )
    before = ordinary_disk_error(environment, monkeypatch)
    fail_deferred_write(monkeypatch)
    with pytest.raises(TaskEnvNotAllowedError) as after:
        if boundary == "stage":
            _stage_environment(environment, environment.parent.name, "daytona")
        else:
            sandbox_for(environment)
    assert_same_diagnostic(before, after.value)
    assert not list(tmp_path.glob(staging.STAGE_PREFIX + "*"))


@pytest.mark.parametrize("consumer", ["upload", "unknown-context", "snapshot"])
async def test_start_disk_failure_retains_public_diagnostic(
    tmp_path, monkeypatch, consumer
):
    """Guards 259b33be: upload and late snapshots retain setup's refusal."""
    from benchflow.sandbox import daytona_content_snapshots as snapshots
    from benchflow.sandbox.daytona_layers import LayeredRecipe
    from benchflow.sandbox.daytona_strategies import _DaytonaDirect

    environment = task(tmp_path)
    before = ordinary_disk_error(environment, monkeypatch)
    sandbox = sandbox_for(environment)
    stage = sandbox._daytona_contexts
    assert stage is not None and stage.pending
    storage = storage_for()
    capture = AsyncMock(side_effect=AssertionError("snapshot read placeholders"))
    monkeypatch.setattr(snapshots, "enabled", lambda: True)
    monkeypatch.setattr(snapshots, "owner_prefix", lambda: "test-owner")
    monkeypatch.setattr(snapshots, "captured_image_async", capture)
    sandbox._install_layers_record = {}

    async def exercise_sdk(force_build):
        if consumer == "snapshot":
            return await _DaytonaDirect(sandbox)._content_snapshot(
                None,
                None,
                None,
                LayeredRecipe(),
                build_allowance_sec=0,
                force_build=False,
                fresh=False,
            )
        context = context_for(stage)
        path, archive = context.source_path, context.archive_path
        if consumer == "unknown-context":
            path, archive = str(stage.environment_dir), "additional"
        return await storage.upload(path, "org", archive)

    sandbox._strategy = SimpleNamespace(start=exercise_sdk)
    fail_deferred_write(monkeypatch)
    try:
        for _ in range(2):
            with pytest.raises(TaskEnvNotAllowedError) as after:
                await sandbox.start(False)
            assert_same_diagnostic(before, after.value)
            assert contexts._CURRENT.get() is None
        assert storage.store.uploads == []
        capture.assert_not_called()
    finally:
        stage.remove()


async def test_late_snapshot_preserves_special_modes(tmp_path, monkeypatch):
    """Guards 259b33be: a late snapshot receives the complete captured mode."""
    from benchflow.sandbox import daytona_content_snapshots as snapshots
    from benchflow.sandbox.daytona_layers import LayeredRecipe
    from benchflow.sandbox.daytona_strategies import _DaytonaDirect

    environment = task(tmp_path)
    (environment / "payload").chmod(0o6755)
    sandbox = sandbox_for(environment)
    stage = sandbox._daytona_contexts
    assert stage is not None and stage.pending
    monkeypatch.setattr(snapshots, "enabled", lambda: True)
    monkeypatch.setattr(snapshots, "owner_prefix", lambda: "test-owner")
    sandbox._install_layers_record = {}

    @asynccontextmanager
    async def captured(image):
        assert not stage.pending
        assert (
            stat.S_IMODE((stage.environment_dir / "payload").stat().st_mode) == 0o6755
        )
        assert (stage.environment_dir / "payload").read_bytes() == (
            environment / "payload"
        ).read_bytes()
        raise RuntimeError("capture reached with original metadata")
        yield  # pragma: no cover

    monkeypatch.setattr(snapshots, "captured_image_async", captured)

    async def exercise_snapshot(force_build):
        return await _DaytonaDirect(sandbox)._content_snapshot(
            None,
            None,
            None,
            LayeredRecipe(),
            build_allowance_sec=0,
            force_build=False,
            fresh=False,
        )

    sandbox._strategy = SimpleNamespace(start=exercise_snapshot)
    try:
        with pytest.raises(
            RuntimeError, match="capture reached with original metadata"
        ):
            await sandbox.start(False)
        assert contexts._CURRENT.get() is None
    finally:
        stage.remove()


async def test_direct_strategy_keeps_wrapped_staging_diagnostic(tmp_path, monkeypatch):
    """Guards 259b33be through the SDK error decorator and real direct start."""
    from daytona._utils.errors import intercept_errors

    from benchflow.sandbox import daytona as daytona_module
    from benchflow.sandbox.protocol import SandboxStartupError

    environment = task(tmp_path)
    before = ordinary_disk_error(environment, monkeypatch)
    sandbox = sandbox_for(environment)
    stage = sandbox._daytona_contexts
    assert stage is not None
    storage = storage_for()

    class Client:
        @intercept_errors(message_prefix="Failed to create sandbox: ")
        async def create(self, params, timeout):
            context = params.image._context_list[0]
            await storage.upload(context.source_path, "org", context.archive_path)
            pytest.fail("a refused upload reached provider creation")

        async def list(self, query=None, request_timeout=None):
            for item in ():
                yield item

    manager = SimpleNamespace(get_client=AsyncMock(return_value=Client()))
    monkeypatch.setattr(
        daytona_module,
        "DaytonaClientManager",
        SimpleNamespace(get_instance=AsyncMock(return_value=manager)),
    )
    fail_deferred_write(monkeypatch)
    try:
        with pytest.raises(TaskEnvNotAllowedError) as after:
            await sandbox.start(False)
        assert_same_diagnostic(before, after.value)
        assert isinstance(after.value.__context__, SandboxStartupError)
        assert contexts._CURRENT.get() is None
        assert len(storage.store.heads) == 1
        assert storage.store.uploads == []
    finally:
        stage.remove()


async def test_start_keeps_unrelated_provider_error(tmp_path):
    """Guards 259b33be: unwrapping staging refusals leaves provider errors intact."""
    environment = task(tmp_path)
    sandbox = sandbox_for(environment)
    error = RuntimeError("provider startup diagnostic")
    error.__cause__ = OSError(errno.EIO, "provider transport failed")
    sandbox._strategy = SimpleNamespace(start=AsyncMock(side_effect=error))
    try:
        with pytest.raises(RuntimeError) as caught:
            await sandbox.start(False)
        assert caught.value is error
        assert contexts._CURRENT.get() is None
    finally:
        sandbox._daytona_contexts.remove()


@pytest.mark.parametrize("outcome", ["hit", "miss", "error"])
async def test_sdk_upload_tracing_and_duration_are_preserved(
    tmp_path, monkeypatch, outcome
):
    """Guards 259b33be: the SDK decorator records spans and upload duration."""
    from daytona._utils import otel_decorator

    environment = task(tmp_path)
    ordinary = staging.stage_environment(environment)
    stage = contexts.stage_daytona_environment(environment)
    events = []
    durations = []

    class Tracer:
        @contextmanager
        def start_as_current_span(self, name, **kwargs):
            events.append(("span", name, kwargs))
            yield SimpleNamespace(
                set_status=lambda status: events.append(("status", status.status_code)),
                record_exception=lambda error: events.append(("error", str(error))),
            )

    class Meter:
        def create_histogram(self, name, **kwargs):
            def record(duration, labels):
                durations.append(duration)
                events.append(("metric", name, kwargs, labels))

            return SimpleNamespace(record=record)

    monkeypatch.setattr(otel_decorator, "get_tracer", lambda: Tracer())
    monkeypatch.setattr(otel_decorator, "get_meter", lambda: Meter())
    monkeypatch.setattr(otel_decorator, "_execution_histograms", {})
    records = []
    try:
        for current in (ordinary, stage):
            events.clear()
            error = (
                RuntimeError("object lookup diagnostic") if outcome == "error" else None
            )
            storage = storage_for(outcome == "hit", error)
            if error is None:
                await upload(current, storage)
            else:
                with pytest.raises(RuntimeError) as caught:
                    await upload(current, storage)
                assert caught.value is error
            records.append(list(events))
        assert records[0] == records[1]
        assert [event[1] for event in records[1] if event[0] == "span"] == [
            "AsyncObjectStorage.upload"
        ]
        assert len(durations) == 2 and all(value >= 0 for value in durations)
        assert sum(event[0] == "metric" for event in records[1]) == 1
    finally:
        ordinary.remove()
        stage.remove()


@pytest.mark.parametrize("phase", ["source", "destination"])
def test_stream_construction_failure_closes_descriptors(tmp_path, monkeypatch, phase):
    """Guards 259b33be: failed fdopen transfers cannot leak either raw fd."""
    stage = contexts.stage_daytona_environment(task(tmp_path))
    original_fdopen = os.fdopen
    opened = []
    before_fds = set(os.listdir("/proc/self/fd"))

    def no_buffer(fd, mode, *args, **kwargs):
        if mode == ("rb" if phase == "source" else "wb"):
            opened.append(fd)
            raise MemoryError("stream buffer allocation failed")
        return original_fdopen(fd, mode, *args, **kwargs)

    monkeypatch.setattr(os, "fdopen", no_buffer)
    try:
        with pytest.raises(MemoryError, match="stream buffer allocation failed"):
            stage.materialize()
        assert len(opened) == 1
        stage.remove()
        with pytest.raises(OSError) as closed:
            os.fstat(opened[0])
        assert closed.value.errno == errno.EBADF
        assert set(os.listdir("/proc/self/fd")) == before_fds
    finally:
        stage.remove()
        for fd in opened:
            with suppress(OSError):
                os.close(fd)
