from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pytest


def _load_runner_module():
    path = Path("benchmarks/hilbench/run_hilbench.py").resolve()
    spec = importlib.util.spec_from_file_location("hilbench_runner_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_hilbench_bucket_uri_downloads_from_bucket_resolve_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Guards PR #279 HILBench bucket downloads do not use dataset APIs."""
    runner = _load_runner_module()
    captured = {}

    def fake_urlopen(request, timeout: int):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["authorization"] = request.get_header("Authorization")
        return _FakeResponse(b"docker image bytes")

    monkeypatch.setattr(runner, "urlopen", fake_urlopen)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    local_path = runner._download_hf_bucket_file(
        "hf://buckets/ScaleAI/hil-bench-swe-images/images/69bc1094b455a91fa20fb868.tar.zst",
        tmp_path,
    )

    assert captured == {
        "url": (
            "https://huggingface.co/buckets/ScaleAI/hil-bench-swe-images/"
            "resolve/images/69bc1094b455a91fa20fb868.tar.zst"
        ),
        "timeout": 60,
        "authorization": None,
    }
    assert local_path == (
        tmp_path
        / "ScaleAI--hil-bench-swe-images"
        / "images"
        / "69bc1094b455a91fa20fb868.tar.zst"
    )
    assert local_path.read_bytes() == b"docker image bytes"


def test_hilbench_bucket_download_uses_optional_hf_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Guards PR #279 optional private-bucket auth without requiring it."""
    runner = _load_runner_module()
    captured = {}

    def fake_urlopen(request, timeout: int):
        captured["authorization"] = request.get_header("Authorization")
        return _FakeResponse(b"docker image bytes")

    monkeypatch.setattr(runner, "urlopen", fake_urlopen)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test-token")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    runner._download_hf_bucket_file(
        "hf://buckets/ScaleAI/hil-bench-swe-images/images/sample.tar.zst",
        tmp_path,
    )

    assert captured["authorization"] == "Bearer test-token"
