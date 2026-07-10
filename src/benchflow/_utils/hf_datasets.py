"""Hugging Face dataset snapshot helpers for task-tree sources."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
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


def _copy_tree(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {dst}")
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git"))


def _copy_selected_tasks(
    source_root: Path,
    output_dir: Path,
    task_ids: Sequence[str],
    *,
    overwrite: bool,
) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}")
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    output_dir.mkdir(parents=True)
    for task_id in task_ids:
        shutil.copytree(
            source_root / task_id,
            output_dir / task_id,
            ignore=shutil.ignore_patterns(".git"),
        )


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
    return str(sha or revision or "main")


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
        revision=revision,
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

    if selected_tasks:
        _copy_selected_tasks(
            source_root,
            output_dir,
            selected_tasks,
            overwrite=overwrite,
        )
    else:
        _copy_tree(source_root, output_dir, overwrite=overwrite)
    provenance = hf_dataset_provenance(
        repo_id=repo_id,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        source_path=source_path,
        local_path=output_dir,
        include_tasks=selected_tasks,
    )
    write_source_sidecar(output_dir, provenance)
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
