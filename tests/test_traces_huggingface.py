"""Tests for benchflow.traces.huggingface — split-aware dataset download."""

from __future__ import annotations

from pathlib import Path

import benchflow.traces.huggingface as hf
from benchflow.traces.huggingface import (
    _download_hf_dataset,
    _pick_split_file,
    _split_filename_candidates,
)


def test_pick_split_file_matches_split_specific_parquet() -> None:
    """A non-train split resolves to its own parquet file, not train."""
    repo_files = [
        "data/train-00000-of-00001.parquet",
        "data/test-00000-of-00001.parquet",
        "README.md",
    ]
    picked = _pick_split_file(repo_files, "test", ".parquet")
    assert picked == "data/test-00000-of-00001.parquet"
    # The train file must never be returned for split="test".
    assert picked is not None
    assert "train" not in picked


def test_pick_split_file_matches_plain_jsonl() -> None:
    """Plain `data/{split}.jsonl` layout resolves to the split file."""
    repo_files = ["data/train.jsonl", "data/validation.jsonl"]
    assert _pick_split_file(repo_files, "validation", ".jsonl") == (
        "data/validation.jsonl"
    )
    assert _pick_split_file(repo_files, "train", ".jsonl") == "data/train.jsonl"


def test_pick_split_file_returns_none_when_no_match() -> None:
    """Missing split returns None so the caller can fall back to guesses."""
    repo_files = ["data/train-00000-of-00001.parquet"]
    assert _pick_split_file(repo_files, "test", ".parquet") is None


def test_pick_split_file_ignores_sibling_subset() -> None:
    """A subset like `test-small-*` must not be picked for split="test".

    The sharded match is anchored on the `{split}-NNNNN-of-NNNNN` convention,
    so only the genuine `test` shard resolves — never a `test-small` subset.
    """
    repo_files = [
        "data/test-small-00000-of-00001.parquet",
        "data/test-00000-of-00001.parquet",
    ]
    picked = _pick_split_file(repo_files, "test", ".parquet")
    assert picked == "data/test-00000-of-00001.parquet"


def test_pick_split_file_no_match_when_only_subset_present() -> None:
    """When only a `test-small` subset exists, split="test" finds nothing."""
    repo_files = ["data/test-small-00000-of-00001.parquet"]
    assert _pick_split_file(repo_files, "test", ".parquet") is None


def test_split_filename_candidates_are_all_split_specific() -> None:
    """Constructed candidates for a non-train split never reference train."""
    candidates = _split_filename_candidates(None, "test", ".parquet")
    assert candidates  # non-empty
    for filename in candidates:
        assert "train" not in filename
        assert "test" in filename
    # Conventional sharded + plain layouts are both attempted.
    assert "data/test-00000-of-00001.parquet" in candidates
    assert "data/test.parquet" in candidates


def test_split_filename_candidates_prefers_matched_file() -> None:
    """A matched repo file is tried before constructed guesses."""
    matched = "data/test-00000-of-00002.parquet"
    candidates = _split_filename_candidates(matched, "test", ".parquet")
    assert candidates[0] == matched


def test_parquet_conversion_failure_falls_through_to_jsonl(monkeypatch, tmp_path):
    """Guards bug G from PR #323: a parquet conversion failure must not abort.

    When a parquet file downloads but the conversion fails (e.g. ``pyarrow``
    missing or decode error), ``_download_hf_dataset`` must fall through to the
    JSONL candidates instead of letting the exception propagate immediately —
    a regression from the prior behavior that alternated formats. Before the
    fix, ``_parquet_to_jsonl`` ran outside the per-candidate try/except.
    """
    import sys
    import types

    cache = tmp_path / "cache"
    cache.mkdir()

    # A JSONL source the fallback path should successfully copy.
    jsonl_src = tmp_path / "src.jsonl"
    jsonl_src.write_text('{"ok": true}\n')

    def fake_list_repo_files(repo_id, repo_type=None):
        return ["data/train-00000-of-00001.parquet", "data/train.jsonl"]

    def fake_hf_hub_download(repo_id, filename, repo_type=None):
        # Parquet "downloads" fine; JSONL "downloads" fine too.
        if filename.endswith(".parquet"):
            p = tmp_path / "downloaded.parquet"
            p.write_bytes(b"not-real-parquet")
            return str(p)
        return str(jsonl_src)

    # Inject a stub `huggingface_hub` so `_download_hf_dataset` takes the
    # hub branch even when the real package is not installed.
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.list_repo_files = fake_list_repo_files
    fake_hub.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    # Parquet conversion fails — exactly the pyarrow-missing / decode-error case.
    def boom(parquet_path, out_path, *, max_rows=None):
        raise ImportError("pyarrow is required for parquet datasets.")

    monkeypatch.setattr(hf, "_parquet_to_jsonl", boom)

    out = _download_hf_dataset("some/repo", split="train", cache=cache)

    # The JSONL fallback ran; the exception did not propagate.
    assert Path(out).exists()
    assert Path(out).read_text() == '{"ok": true}\n'
