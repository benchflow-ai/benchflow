"""Stale-audit refresh step for the no-web denylist."""

import json

from tools import hosted_retrieval_scan as scan


def test_the_scan_names_a_search_model_the_denylist_misses(tmp_path):
    table = {
        "acme/acme-web-research-2": {"mode": "chat", "litellm_provider": "acme"},
        "openai/gpt-6.1-sol": {"mode": "chat", "litellm_provider": "openai"},
        "empiriolabs/perplexity-deep-research": {
            "mode": "chat",
            "litellm_provider": "empiriolabs",
        },
        "acme/web-embed": {"mode": "embedding", "litellm_provider": "acme"},
        "gpt-rosalind-research": {"mode": "chat", "litellm_provider": "openai"},
    }
    assert scan.missed(table) == [("acme/acme-web-research-2", "acme")]
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(table))
    assert scan.main(["--table", str(path)]) == 1


def test_the_pinned_table_has_nothing_left_to_add(tmp_path):
    """The table LiteLLM's pinned wheel ships: every search-like chat model is
    refused or recorded as not reaching the web."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("litellm")
    assert spec and spec.submodule_search_locations
    folder = Path(next(iter(spec.submodule_search_locations)))
    table = json.loads(
        (folder / "model_prices_and_context_window_backup.json").read_text()
    )
    assert scan.missed(table) == []
