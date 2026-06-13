"""Task-directory targets for consumers that need BenchFlow-native packages."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NativeTaskTarget:
    """A native task directory selected for validation or execution."""

    path: Path
    source_path: Path
    adapter_source: str | None = None


class InboundTaskMaterializer:
    """Materialize foreign inbound task dirs for a longer-lived consumer."""

    def __init__(self, *, prefix: str = "benchflow-task-target-") -> None:
        self._temp_dir = tempfile.TemporaryDirectory(prefix=prefix)
        self._root = Path(self._temp_dir.name)
        self._cache: dict[Path, NativeTaskTarget] = {}

    @property
    def root(self) -> Path:
        return self._root

    def close(self) -> None:
        self._temp_dir.cleanup()

    def __enter__(self) -> InboundTaskMaterializer:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def native_target(self, task_dir: Path | str) -> NativeTaskTarget:
        """Return a native task target for ``task_dir``.

        Native/Harbor task packages return their original path. Other recognized
        inbound task directories are materialized once and cached for the life of
        this materializer. Unsupported recognized tasks raise their adapter's
        ``UnsupportedInboundTaskError``.
        """
        from benchflow.adapters.harbor import HarborAdapter
        from benchflow.adapters.inbound import (
            detect_adapter,
            materialize_inbound_task_md,
        )

        source_path = Path(task_dir)
        try:
            adapter = detect_adapter(source_path)
        except ValueError:
            return NativeTaskTarget(path=source_path, source_path=source_path)

        if adapter is HarborAdapter:
            return NativeTaskTarget(path=source_path, source_path=source_path)

        cache_key = _cache_key(source_path)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        inbound = adapter.from_task_dir(source_path)
        native_task = materialize_inbound_task_md(
            inbound,
            self._next_destination(inbound.name),
        )
        target = NativeTaskTarget(
            path=native_task,
            source_path=source_path,
            adapter_source=inbound.source,
        )
        self._cache[cache_key] = target
        return target

    def _next_destination(self, name: str) -> Path:
        dest = self._root / name
        if not dest.exists():
            return dest
        index = 2
        while True:
            candidate = self._root / f"{name}-{index}"
            if not candidate.exists():
                return candidate
            index += 1


@contextmanager
def native_task_target(task_dir: Path) -> Iterator[NativeTaskTarget]:
    """Yield a native task directory for a short-lived operation."""
    with InboundTaskMaterializer() as materializer:
        yield materializer.native_target(task_dir)


def _cache_key(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()
