"""Guards FF-sw-other Atlas judge outages at 363d815b; no network."""

import io
import json
import sys
import urllib.error

import pytest

from benchflow.adapters.resources import mcp_atlas_judge as atlas


@pytest.mark.parametrize(
    "failure", ["http", "network", "json", "shape", "reward", "missing_key"]
)
def test_atlas_never_publishes_score_without_judge_verdict(
    tmp_path, monkeypatch, failure
):
    payload = tmp_path / "case.json"
    payload.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["judge", str(payload)])
    monkeypatch.setattr(atlas, "read_final_answer", lambda: "dummy answer")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    writes = []
    monkeypatch.setattr(atlas, "write_reward", lambda *args: writes.append(args))

    def urlopen(*args, **kwargs):
        if failure == "http":
            raise urllib.error.HTTPError(
                "https://example.test", 503, "unavailable", {}, io.BytesIO(b"dummy")
            )
        if failure == "network":
            raise OSError("dummy outage")
        content = {
            "json": "not json",
            "shape": "{}",
            "reward": '{"reward": "nan"}',
        }.get(failure, '{"reward": 1}')
        data = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        return io.BytesIO(data)

    monkeypatch.setattr(atlas.urllib.request, "urlopen", urlopen)
    if failure == "missing_key":
        monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises((RuntimeError, SystemExit)):
        atlas.main()
    assert writes == []
