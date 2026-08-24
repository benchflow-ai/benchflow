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


def _bucket_tree_url(bucket_id: str, path_in_repo: str) -> str:
    # Buckets are non-versioned (no "main" branch), so the browsable listing
    # is /tree/<path> directly, not /resolve/<path> (which 404s) or
    # /tree/main/<path> (the git-repo convention _tree_url uses).
    suffix = f"/tree/{path_in_repo.strip('/')}" if path_in_repo else ""
    return f"https://huggingface.co/buckets/{bucket_id}{suffix}"


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


def publish_folder_to_bucket(
    folder: Path,
    *,
    bucket_id: str,
    path_in_repo: str = "",
    private: bool = False,
) -> HfPublishResult:
    if not folder.is_dir():
        raise ValueError(f"publish source folder not found: {folder}")
    try:
        from huggingface_hub import create_bucket, sync_bucket
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ValueError(
            "huggingface_hub with bucket support (create_bucket/sync_bucket) is "
            "required for --publish-bucket; upgrade huggingface_hub"
        ) from exc
    try:
        create_bucket(bucket_id, private=private)
    except HfHubHTTPError as exc:
        if exc.response is None or exc.response.status_code != 409:
            raise
    prefix = path_in_repo.strip("/")
    remote = (
        f"hf://buckets/{bucket_id}/{prefix}" if prefix else f"hf://buckets/{bucket_id}"
    )
    sync_bucket(str(folder), remote)
    return HfPublishResult(
        repo_id=bucket_id,
        repo_type="bucket",
        path_in_repo=prefix,
        url=_bucket_tree_url(bucket_id, prefix),
    )


_EVAL_RESULTS_PATH = ".eval_results/benchflow.yaml"


def _existing_eval_results(model_repo: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        downloaded = hf_hub_download(
            repo_id=model_repo, filename=_EVAL_RESULTS_PATH, repo_type="model"
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return []
    import yaml

    data = yaml.safe_load(Path(downloaded).read_text())
    return data if isinstance(data, list) else []


def _upsert_eval_result(existing: list[dict], entry: dict) -> list[dict]:
    key = (entry["dataset"]["id"], entry["dataset"]["task_id"])
    kept = [
        row
        for row in existing
        if (row.get("dataset", {}).get("id"), row.get("dataset", {}).get("task_id"))
        != key
    ]
    return [*kept, entry]


def open_eval_results_pr(
    *,
    model_repo: str,
    dataset_id: str,
    task_id: str,
    value: float,
    source_url: str | None = None,
    notes: str | None = None,
) -> str | None:
    try:
        import yaml
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ValueError(
            "huggingface_hub is required for --eval-results-model"
        ) from exc
    entry: dict = {"dataset": {"id": dataset_id, "task_id": task_id}, "value": value}
    if source_url:
        entry["source"] = {"url": source_url}
    if notes:
        entry["notes"] = notes
    merged = _upsert_eval_result(_existing_eval_results(model_repo), entry)
    content = yaml.safe_dump(merged, sort_keys=False).encode("utf-8")
    commit = HfApi().create_commit(
        repo_id=model_repo,
        repo_type="model",
        operations=[
            CommitOperationAdd(path_in_repo=_EVAL_RESULTS_PATH, path_or_fileobj=content)
        ],
        commit_message=f"Add {dataset_id} eval results",
        create_pr=True,
    )
    return commit.pr_url


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
