"""Unit tests for the HF bucket publish and eval-results PR helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import httpx
import pytest
import yaml

from benchflow.publish.huggingface import open_eval_results_pr, publish_folder_to_bucket


def _folder(tmp_path: Path) -> Path:
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "result.json").write_text("{}")
    return folder


def test_publish_folder_to_bucket_missing_bucket_api_raises_clear_error(
    tmp_path, monkeypatch
):
    """Guards against an installed huggingface_hub too old for bucket support.

    huggingface_hub lazily resolves top-level attributes (no create_bucket/
    sync_bucket in __dict__ until accessed), so delattr can't simulate a
    missing name — swap in a plain stub module instead.
    """
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.ModuleType("huggingface_hub")
    )
    with pytest.raises(ValueError, match="bucket support"):
        publish_folder_to_bucket(_folder(tmp_path), bucket_id="org/bucket")


def test_publish_folder_to_bucket_url_uses_tree_not_resolve(tmp_path, monkeypatch):
    """Buckets are non-versioned; /resolve/<path> 404s, /tree/<path> is correct."""
    calls: dict = {}
    monkeypatch.setattr(
        "huggingface_hub.create_bucket",
        lambda bucket_id, private=False: calls.setdefault(
            "create", (bucket_id, private)
        ),
    )
    monkeypatch.setattr(
        "huggingface_hub.sync_bucket",
        lambda local, remote: calls.setdefault("sync", (local, remote)),
    )

    result = publish_folder_to_bucket(
        _folder(tmp_path), bucket_id="org/bucket", path_in_repo="runs/x"
    )

    assert result.url == "https://huggingface.co/buckets/org/bucket/tree/runs/x"
    assert calls["sync"][1] == "hf://buckets/org/bucket/runs/x"


def test_publish_folder_to_bucket_ignores_existing_bucket_conflict(
    tmp_path, monkeypatch
):
    from huggingface_hub.errors import HfHubHTTPError

    response = httpx.Response(409, request=httpx.Request("POST", "https://example.com"))

    def fake_create_bucket(bucket_id, private=False):
        raise HfHubHTTPError("exists", response=response)

    monkeypatch.setattr("huggingface_hub.create_bucket", fake_create_bucket)
    monkeypatch.setattr("huggingface_hub.sync_bucket", lambda local, remote: None)

    result = publish_folder_to_bucket(_folder(tmp_path), bucket_id="org/bucket")
    assert result.repo_id == "org/bucket"


def test_open_eval_results_pr_missing_api_raises_clear_error(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.ModuleType("huggingface_hub")
    )
    with pytest.raises(ValueError, match="huggingface_hub is required"):
        open_eval_results_pr(
            model_repo="org/model", dataset_id="org/bench", task_id="t1", value=1.0
        )


class _FakeCommitInfo:
    pr_url = "https://huggingface.co/org/model/discussions/1"


def _fake_hf_api(monkeypatch, captured: dict):
    class FakeHfApi:
        def create_commit(
            self, *, repo_id, repo_type, operations, commit_message, create_pr
        ):
            captured["content"] = operations[0].path_or_fileobj
            return _FakeCommitInfo()

    monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)


def test_open_eval_results_pr_upserts_without_clobbering_other_entries(
    tmp_path, monkeypatch
):
    """Guards against overwriting unrelated (dataset, task_id) entries."""
    existing_path = tmp_path / "existing.yaml"
    existing_path.write_text(
        "- dataset:\n    id: org/bench\n    task_id: other-task\n  value: 42.0\n"
        "- dataset:\n    id: org/bench\n    task_id: t1\n  value: 1.0\n"
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo_id, filename, repo_type=None: str(existing_path),
    )
    captured: dict = {}
    _fake_hf_api(monkeypatch, captured)

    pr_url = open_eval_results_pr(
        model_repo="org/model", dataset_id="org/bench", task_id="t1", value=88.0
    )

    assert pr_url == _FakeCommitInfo.pr_url
    entries = yaml.safe_load(captured["content"])
    by_key = {
        (e["dataset"]["id"], e["dataset"]["task_id"]): e["value"] for e in entries
    }
    assert by_key == {("org/bench", "other-task"): 42.0, ("org/bench", "t1"): 88.0}


def test_open_eval_results_pr_no_existing_file_writes_single_entry(monkeypatch):
    from huggingface_hub.errors import EntryNotFoundError

    def fake_hf_hub_download(repo_id, filename, repo_type=None):
        raise EntryNotFoundError("nope")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    captured: dict = {}
    _fake_hf_api(monkeypatch, captured)

    open_eval_results_pr(
        model_repo="org/model", dataset_id="org/bench", task_id="t1", value=1.0
    )

    entries = yaml.safe_load(captured["content"])
    assert len(entries) == 1
    assert entries[0]["dataset"] == {"id": "org/bench", "task_id": "t1"}
