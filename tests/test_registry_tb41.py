"""Guard the versioned registry addition against 61b4d227."""

import json
import re
import unittest
from pathlib import Path
from urllib.parse import urlsplit

REGISTRY = (
    Path(__file__).resolve().parents[1] / "src/benchflow/_utils/dataset_registry.json"
)


class TestTerminalBench41Registry(unittest.TestCase):
    def test_release_is_pinned(self):
        entries = json.loads(REGISTRY.read_text())
        entry = next(
            e
            for e in entries
            if e["name"] == "terminal-bench" and e["version"] == "4.1"
        )
        assert entry["git_tag"] == "v4.1.0"
        assert len(entry["tasks"]) == 66
        assert len({t["name"] for t in entry["tasks"]}) == 66
        for task in entry["tasks"]:
            assert task["git_commit_id"] == "4070d6cef56a7d20cb5dc98d5e10ada88a6b9440"
            assert task["path"] == f"tasks/{task['name']}"
            assert re.fullmatch(r"sha256:[0-9a-f]{64}", task["digest"])
        assert any(t["name"] == "interleaved-vigenere" for t in entry["tasks"])

    def test_repository_url_is_ready_for_resolution(self):
        entries = json.loads(REGISTRY.read_text())
        entry = next(
            e
            for e in entries
            if e["name"] == "terminal-bench" and e["version"] == "4.1"
        )
        urls = {task["git_url"] for task in entry["tasks"]}
        assert len(urls) == 1
        url = urls.pop()
        assert "PENDING" not in url.upper()
        parsed = urlsplit(url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "github.com"
        assert re.fullmatch(r"/[A-Za-z0-9_.-]+/terminal-bench\.git", parsed.path)
        assert not parsed.query and not parsed.fragment
