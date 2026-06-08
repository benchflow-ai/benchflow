from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest


def _load_publish_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments" / "skillsbench-fill" / "publish.py"
    spec = importlib.util.spec_from_file_location("skillsbench_fill_publish", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publish_replaces_empty_hf_token(monkeypatch) -> None:
    """Guards PR #644 against an empty HF_TOKEN breaking PR5 uploads."""
    publish = _load_publish_module()
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("HUGGING_FACE_TOKEN", "hf_abcdefghijklmnopqrstuvwxyz")

    token = publish.configure_hf_token_env()

    assert token == "hf_abcdefghijklmnopqrstuvwxyz"
    assert token == os.environ["HF_TOKEN"]


def test_publish_writes_marker_only_after_hf_commit(tmp_path, monkeypatch) -> None:
    """Guards PR #644 against local dashboard credit when the HF commit fails."""
    publish = _load_publish_module()
    publish.ROOT = tmp_path
    runs_root = tmp_path / "runs"
    cell = "gemini-3.5-flash__with__citation-check__t1"
    rollout = runs_root / cell / "citation-check__trialabc"
    (tmp_path / "review").mkdir()
    (rollout / "trajectory").mkdir(parents=True)
    (tmp_path / "review" / f"{cell}.json").write_text(
        json.dumps({"cell_id": cell, "verdict": "pass"})
    )
    (rollout / "config.json").write_text("{}")
    (rollout / "result.json").write_text(
        json.dumps({"timing": {"total_s": 1.0}, "rewards": {"reward": 1.0}})
    )
    (rollout / "timing.json").write_text(json.dumps({"total_s": 1.0}))
    (rollout / "trajectory" / "acp_trajectory.jsonl").write_text("{}\n")
    (rollout / "trajectory" / "llm_trajectory.jsonl").write_text("{}\n")

    class FakeApi:
        def __init__(self, token):
            self.token = token

        def list_repo_tree(self, *args, **kwargs):
            return []

        def create_commit(self, *args, **kwargs):
            raise RuntimeError("HF unavailable")

    class FakeCommitOperationAdd:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_hf = types.SimpleNamespace(
        HfApi=FakeApi, CommitOperationAdd=FakeCommitOperationAdd
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setenv("HUGGING_FACE_TOKEN", "hf_abcdefghijklmnopqrstuvwxyz")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish.py",
            "--runs-root",
            str(runs_root),
            "--ts",
            "2026-06-08__hotfix",
        ],
    )

    with pytest.raises(RuntimeError, match="HF unavailable"):
        publish.main()

    assert not (tmp_path / "published" / f"{cell}.json").exists()
