"""Hugging Face publishing helpers with optional read-after-write checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass(frozen=True)
class HfPublishResult:
    repo_id: str
    repo_type: str
    path_in_repo: str
    url: str
    commit_url: str | None = None


def _require_hf_api():
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise ValueError(
            "huggingface_hub is required for --publish-hf/--publish-model"
        ) from exc
    return HfApi()


def _tree_url(repo_id: str, repo_type: str, path_in_repo: str) -> str:
    kind = "datasets/" if repo_type == "dataset" else ""
    suffix = f"/tree/main/{path_in_repo.strip('/')}" if path_in_repo else "/tree/main"
    return f"https://huggingface.co/{kind}{repo_id}{suffix}"


def _resolve_url(repo_id: str, repo_type: str, path_in_repo: str) -> str:
    kind = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{kind}{repo_id}/resolve/main/{path_in_repo}"


def _check_public_files(repo_id: str, repo_type: str, path_in_repo: str) -> None:
    index_url = _tree_url(repo_id, repo_type, path_in_repo)
    response = httpx.get(index_url, follow_redirects=True, timeout=20)
    if response.status_code >= 400:
        raise ValueError(
            f"HF public read check failed: {index_url} -> {response.status_code}"
        )


def publish_folder_to_hf(
    folder: Path,
    *,
    repo_id: str,
    path_in_repo: str,
    repo_type: str = "dataset",
    public_read_check: bool = False,
    commit_message: str | None = None,
) -> HfPublishResult:
    if not folder.is_dir():
        raise ValueError(f"publish source folder not found: {folder}")
    api = _require_hf_api()
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(folder),
        path_in_repo=path_in_repo.strip("/"),
        commit_message=commit_message
        or f"Upload BenchFlow artifacts to {path_in_repo}",
    )
    if public_read_check:
        _check_public_files(repo_id, repo_type, path_in_repo)
    return HfPublishResult(
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        url=_tree_url(repo_id, repo_type, path_in_repo),
        commit_url=getattr(commit, "commit_url", None),
    )


def publish_file_to_hf(
    file_path: Path,
    *,
    repo_id: str,
    path_in_repo: str,
    repo_type: str = "dataset",
    public_read_check: bool = False,
    commit_message: str | None = None,
) -> HfPublishResult:
    if not file_path.is_file():
        raise ValueError(f"publish source file not found: {file_path}")
    api = _require_hf_api()
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
    commit = api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_or_fileobj=str(file_path),
        path_in_repo=path_in_repo.strip("/"),
        commit_message=commit_message or f"Upload BenchFlow artifact {path_in_repo}",
    )
    if public_read_check:
        url = _resolve_url(repo_id, repo_type, path_in_repo.strip("/"))
        response = httpx.head(url, follow_redirects=True, timeout=20)
        if response.status_code >= 400:
            raise ValueError(
                f"HF public read check failed: {url} -> {response.status_code}"
            )
    return HfPublishResult(
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        url=_tree_url(repo_id, repo_type, str(Path(path_in_repo).parent)),
        commit_url=getattr(commit, "commit_url", None),
    )
