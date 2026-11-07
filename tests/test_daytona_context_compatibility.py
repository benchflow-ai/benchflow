"""Require the stdlib executor contract before reader custody."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("daytona")

from daytona import Image
from daytona._async.object_storage import AsyncObjectStorage

from benchflow.sandbox import daytona_content_snapshots as content
from benchflow.sandbox import daytona_context

_SDK_HASH = inspect.unwrap(AsyncObjectStorage._compute_hash_for_path_md5)


class _AlternateExecutorLoop(asyncio.SelectorEventLoop):
    """Use qasync's separate default-pool layout with a real executor."""

    def __init__(self):
        super().__init__()
        self._alternate_pool = None

    def run_in_executor(self, executor, function, *args):
        self._check_closed()
        if executor is None:
            if self._alternate_pool is None:
                self._alternate_pool = ThreadPoolExecutor(max_workers=2)
            executor = self._alternate_pool
        return asyncio.wrap_future(executor.submit(function, *args), loop=self)

    def set_default_executor(self, executor):
        self._alternate_pool = executor

    async def shutdown_default_executor(self, timeout=None):
        if self._alternate_pool is not None:
            self._alternate_pool.shutdown(wait=True)

    def close(self):
        if self._alternate_pool is not None:
            self._alternate_pool.shutdown(wait=True)
        super().close()


def _check_upload(loop, tmp_path, monkeypatch, caplog, *, optimized):
    source = tmp_path / "seed"
    source.write_bytes(b"seed")
    image = Image.base("alpine@sha256:" + "1" * 64).add_local_file(str(source), "/seed")
    storage = object.__new__(AsyncObjectStorage)
    sdk_calls = []
    acquisitions = []
    stages = []
    checks = AsyncMock(return_value=True)
    uploaded = AsyncMock()
    monkeypatch.setattr(storage, "_file_exists_in_s3", checks)
    monkeypatch.setattr(storage, "_upload_as_tar", uploaded)
    original_acquire = daytona_context._ContextReaders.acquire

    @functools.wraps(_SDK_HASH)
    async def sdk(self, path_str, archive_base_path=None):
        sdk_calls.append(path_str)
        return await _SDK_HASH(self, path_str, archive_base_path)

    def acquire(self):
        acquisitions.append(self)
        original_acquire(self)

    monkeypatch.setattr(AsyncObjectStorage, "_compute_hash_for_path_md5", sdk)
    monkeypatch.setattr(daytona_context._ContextReaders, "acquire", acquire)
    reader = Mock(wraps=daytona_context._HashReader)
    monkeypatch.setattr(daytona_context, "_HashReader", reader)
    daytona_context.patch_context_hashing()
    caplog.set_level(logging.DEBUG, logger="benchflow")

    async def upload():
        expected = await _SDK_HASH(storage, str(source), "seed")
        async with content.captured_image_async(image) as staged:
            path = staged._context_list[0].source_path
            stages.append(path)
            assert await storage.upload(path, "review", "seed") == expected

    try:
        loop.run_until_complete(upload())
        assert reader.call_count == len(acquisitions) == int(optimized)
        assert len(sdk_calls) == int(not optimized)
        assert all(readers._count == 0 for readers in acquisitions)
        checks.assert_awaited_once()
        uploaded.assert_not_awaited()
        assert not any(Path(path).exists() for path in stages)
        messages = [
            record.getMessage()
            for record in caplog.records
            if record.name == "benchflow"
            and record.getMessage().startswith("Using Daytona SDK context hashing")
        ]
        assert messages == (
            []
            if optimized
            else [
                "Using Daytona SDK context hashing for event loop "
                f"{type(loop).__module__}.{type(loop).__qualname__}"
            ]
        )
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()


def test_alternate_pool_layout_uses_whole_sdk(tmp_path, monkeypatch, caplog):
    """An alternative BaseEventLoop never acquires reader custody."""
    loop = _AlternateExecutorLoop()
    _check_upload(loop, tmp_path, monkeypatch, caplog, optimized=False)
    assert loop._default_executor is None


@pytest.mark.parametrize(
    "method",
    [
        "run_in_executor",
        "set_default_executor",
        "shutdown_default_executor",
        "_check_default_executor",
        "_check_closed",
        "_check_callback",
        "_do_shutdown",
    ],
)
def test_overridden_executor_contract_uses_sdk(tmp_path, monkeypatch, caplog, method):
    """Every executor-method override selects the SDK before work."""
    original = getattr(asyncio.BaseEventLoop, method)

    def overridden(self, *args, **kwargs):
        return original(self, *args, **kwargs)

    loop_class = type(
        "AlternateLoop", (asyncio.SelectorEventLoop,), {method: overridden}
    )
    _check_upload(loop_class(), tmp_path, monkeypatch, caplog, optimized=False)


@pytest.mark.parametrize("kind", ["stdlib", "uvloop", "qasync"])
def test_real_loop_upload_selects_supported_contract(
    tmp_path, monkeypatch, caplog, kind
):
    """Real loops keep SDK keys and select only a proven executor."""
    if kind == "stdlib":
        loop = asyncio.new_event_loop()
    elif kind == "uvloop":
        loop = pytest.importorskip("uvloop").new_event_loop()
    else:
        qasync = pytest.importorskip("qasync")
        qt = pytest.importorskip("PyQt5.QtCore")
        app = qt.QCoreApplication.instance() or qt.QCoreApplication([])
        loop = qasync.QEventLoop(app)
    _check_upload(loop, tmp_path, monkeypatch, caplog, optimized=kind == "stdlib")
