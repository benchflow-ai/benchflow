"""Hugging Face dataset snapshot helpers for task-tree sources."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow._utils.benchmark_repos import ResolvedSource, task_file_hashes

SOURCE_SIDECAR = ".benchflow-source.json"


@dataclass(frozen=True, slots=True)
class HfDatasetSnapshot:
    """A local Hugging Face dataset snapshot plus source provenance."""

    path: Path
    provenance: dict[str, Any]


@contextmanager
def _staged_output(source: Path, output: Path, *, overwrite: bool):
    # A destination inside the source can recursively copy itself; an ancestor
    # can destroy the download cache. Resolve aliases before either operation.
    src, dst = source.resolve(), output.resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError("HF snapshot source and output trees must not overlap")
    if output.is_symlink():
        raise ValueError("HF snapshot output must not be a symlink")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage on the destination filesystem, keeping the previous snapshot until
    # every copy/write succeeds. A failed publish restores the previous entry.
    with tempfile.TemporaryDirectory(prefix=".benchflow-hf-", dir=output.parent) as tmp:
        stage, backup = Path(tmp) / "snapshot", Path(tmp) / "previous"
        yield stage
        try:
            if output.exists():
                output.rename(backup)
            stage.rename(output)
        except BaseException:
            # Cancellation must not let TemporaryDirectory delete the old
            # snapshot while its public name is still absent.
            if backup.exists() and not output.exists():
                backup.rename(output)
            raise


def _copy_tree(src: Path, dst: Path, *, overwrite: bool) -> None:
    with _staged_output(src, dst, overwrite=overwrite) as stage:
        _populate_snapshot(src, stage, ())


def _populate_snapshot(source: Path, output: Path, task_ids: Sequence[str]) -> None:
    if not task_ids:
        shutil.copytree(source, output, ignore=shutil.ignore_patterns(".git"))
        return
    output.mkdir()
    for task_id in task_ids:
        shutil.copytree(
            source / task_id,
            output / task_id,
            ignore=shutil.ignore_patterns(".git"),
        )


def _copy_selected_tasks(
    source_root: Path,
    output_dir: Path,
    task_ids: Sequence[str],
    *,
    overwrite: bool,
) -> None:
    with _staged_output(source_root, output_dir, overwrite=overwrite) as stage:
        _populate_snapshot(source_root, stage, task_ids)


def _read_hf_revision(repo_id: str, revision: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for HF dataset snapshots. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    info = HfApi().repo_info(repo_id, repo_type="dataset", revision=revision)
    sha = getattr(info, "sha", None)
    if not sha:
        raise ValueError(f"Could not resolve HF dataset revision for {repo_id!r}")
    return str(sha)


def _download_snapshot(
    repo_id: str,
    *,
    revision: str | None,
    cache_dir: Path | None,
    allow_patterns: Sequence[str] = (),
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for HF dataset snapshots. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if allow_patterns:
        kwargs["allow_patterns"] = list(allow_patterns)
    return Path(snapshot_download(**kwargs))


def hf_dataset_provenance(
    *,
    repo_id: str,
    requested_revision: str | None,
    resolved_revision: str,
    source_path: str,
    local_path: Path,
    include_tasks: Sequence[str] = (),
) -> dict[str, Any]:
    """Build generic HF dataset source provenance for a local task tree."""

    path = source_path.strip("/")
    provenance: dict[str, Any] = {
        "type": "huggingface_dataset",
        "repo": repo_id,
        "repo_type": "dataset",
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "path": path,
        "local_path": str(local_path),
        "dirty": False,
        "file_hashes": task_file_hashes(local_path)
        if (local_path / "task.toml").is_file() or (local_path / "task.md").is_file()
        else {},
    }
    if include_tasks:
        provenance["include_tasks"] = list(include_tasks)
    return provenance


def write_source_sidecar(root: Path, provenance: dict[str, Any]) -> Path:
    """Persist source provenance beside a materialized task snapshot."""

    sidecar = root / SOURCE_SIDECAR
    sidecar.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return sidecar


def load_source_sidecar(path: Path) -> dict[str, Any] | None:
    """Load a source sidecar from *path* or one of its parents."""

    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    candidates = [resolved] if resolved.is_dir() else [resolved.parent]
    candidates.extend(resolved.parents)
    for candidate in candidates:
        sidecar = candidate / SOURCE_SIDECAR
        if not sidecar.is_file():
            continue
        try:
            data = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        data = dict(data)
        data["local_path"] = str(candidate)
        return data
    return None


def snapshot_hf_dataset(
    repo_id: str,
    *,
    output_dir: Path,
    revision: str | None = None,
    path: str | None = None,
    cache_dir: Path | None = None,
    overwrite: bool = False,
    include_tasks: Sequence[str] = (),
) -> HfDatasetSnapshot:
    """Materialize a HF dataset snapshot and stamp local source metadata."""

    resolved_revision = _read_hf_revision(repo_id, revision)
    source_path = (path or "").strip("/")
    selected_tasks = tuple(dict.fromkeys(include_tasks))
    allow_patterns = tuple(
        f"{source_path}/{task_id}/**" if source_path else f"{task_id}/**"
        for task_id in selected_tasks
    )
    snapshot_root = _download_snapshot(
        repo_id,
        revision=resolved_revision,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
    )
    source_root = snapshot_root / source_path if source_path else snapshot_root
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"Path {source_path!r} not found in HF dataset {repo_id!r}"
        )

    missing_tasks = [
        task_id for task_id in selected_tasks if not (source_root / task_id).is_dir()
    ]
    if missing_tasks:
        raise FileNotFoundError(
            f"Tasks not found under {source_path or '.'!r} in HF dataset "
            f"{repo_id!r}: {', '.join(missing_tasks)}"
        )

    with _staged_output(source_root, output_dir, overwrite=overwrite) as stage:
        _populate_snapshot(source_root, stage, selected_tasks)
        provenance = hf_dataset_provenance(
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            source_path=source_path,
            local_path=stage,
            include_tasks=selected_tasks,
        )
        provenance["local_path"] = str(output_dir)
        write_source_sidecar(stage, provenance)
    return HfDatasetSnapshot(path=output_dir, provenance=provenance)


def resolved_source_from_hf_snapshot(
    repo_id: str,
    *,
    output_dir: Path,
    revision: str | None = None,
    path: str | None = None,
    cache_dir: Path | None = None,
    overwrite: bool = False,
) -> ResolvedSource:
    """Return a ``ResolvedSource`` for a materialized HF task snapshot."""

    snapshot = snapshot_hf_dataset(
        repo_id,
        output_dir=output_dir,
        revision=revision,
        path=path,
        cache_dir=cache_dir,
        overwrite=overwrite,
    )
    return ResolvedSource(path=snapshot.path, provenance=snapshot.provenance)
