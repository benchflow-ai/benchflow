"""Guards the fix at 363d815b."""

import json

import pytest

from benchflow.traces import huggingface as hf


@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("suffix", [".jsonl", ".parquet"])
@pytest.mark.parametrize("limit", [None, 1, 3, 4, 10])
def test_loads_all_split_shards_with_one_global_limit(
    tmp_path, monkeypatch, suffix, limit, stale
):
    """Guards FF-sw-trajectories against first-shard-only caches."""
    names = [f"data/test-{i:05}-of-00002{suffix}" for i in range(2)]
    sources = {}
    for index, name in enumerate(names):
        path = tmp_path / f"shard-{index}"
        path.write_text(
            "".join(json.dumps({"n": index * 2 + row}) + "\n" for row in range(2))
        )
        sources[name] = str(path)
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        lambda *a, **kw: ["data/test-small-00000-of-00001" + suffix, *reversed(names)],
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *, filename, **kw: sources[filename]
    )
    monkeypatch.setattr(
        hf,
        "_parquet_to_jsonl",
        lambda src, dst, max_rows=None: hf._copy_jsonl(src, dst, max_rows=max_rows),
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    # A pre-fix cache must not preserve the first-shard truncation forever.
    old = cache / ("dummy__traces__test" + (f"_n{limit}" if limit else "") + ".jsonl")
    if stale:
        old.write_text('{"n": 999}\n')
    result = hf._download_hf_dataset(
        "dummy/traces", split="test", max_rows=limit, cache=cache
    )
    assert [json.loads(line)["n"] for line in result.read_text().splitlines()] == list(
        range(4)
    )[:limit]
    assert (
        hf._download_hf_dataset(
            "dummy/traces", split="test", max_rows=limit, cache=cache
        )
        == result
    )


def test_failed_later_shard_never_publishes_partial_cache(tmp_path, monkeypatch):
    """Guards FF-sw-trajectories: a failed shard must not look complete."""
    first = tmp_path / "first"
    first.write_text('{"n": 0}\n')
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        lambda *a, **kw: [f"data/train-{i:05}-of-00002.jsonl" for i in range(2)],
    )

    def download(*, filename, **kwargs):
        if filename == "data/train-00000-of-00002.jsonl":
            return str(first)
        raise OSError("dummy unavailable shard")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)

    def unavailable(*args, **kwargs):
        raise RuntimeError("dummy API unavailable")

    monkeypatch.setattr(hf, "_download_via_api", unavailable)
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(RuntimeError, match="API unavailable"):
        hf._download_hf_dataset("dummy/traces", cache=cache)
    assert list(cache.glob("*.jsonl")) == []


@pytest.mark.parametrize("limit", [None, 101, 220])
def test_api_fallback_paginates_to_requested_population(tmp_path, monkeypatch, limit):
    """Guards FF-sw-trajectories against equivalent truncation in API fallback."""
    from urllib.parse import parse_qs, urlsplit

    import httpx

    def get(url, **kwargs):
        query = parse_qs(urlsplit(url).query)
        offset, length = int(query["offset"][0]), int(query["length"][0])
        rows = [{"row": {"n": n}} for n in range(220)][offset : offset + length]
        return httpx.Response(
            200,
            json={"rows": rows, "num_rows_total": 220},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr("httpx.get", get)
    output = tmp_path / "rows.jsonl"
    hf._download_via_api("dummy/traces", out_path=output, max_rows=limit)
    assert len(output.read_text().splitlines()) == (limit or 220)
