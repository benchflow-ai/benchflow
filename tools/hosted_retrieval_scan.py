"""List search-like models in a LiteLLM price table that the no-web denylist misses.

A model that searches or browses by itself has no tool or request field to
refuse, so a no-web run refuses it by name
(``litellm_request_guard.HOSTED_RETRIEVAL_MODELS``). That list only knows the
models someone added; this scan is its refresh step. Run it when LiteLLM is
bumped and before a no-web campaign::

    uv run python -m tools.hosted_retrieval_scan            # LiteLLM main's table
    uv run python -m tools.hosted_retrieval_scan --table prices.json

It prints each chat model whose name looks like a search, research or online
model and that no entry refuses, and exits 1 when there is one. Read each on
its vendor's page: add the ones that reach the web to the denylist (with an
example), and the ones that do not to ``NOT_WEB`` below, with the reason.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from typing import Any

from benchflow.providers.litellm_request_guard import hosted_retrieval_model

MAIN_TABLE = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
SEARCH_LIKE = re.compile(
    r"search|research|sonar|online|compound|browse|grounding|web", re.IGNORECASE
)
MODES = {None, "chat", "responses", "completion"}
# Names that look like search models and are not, each read on 2026-11-04.
NOT_WEB = {
    "gpt-rosalind-research": (
        "OpenAI's life-sciences model; neither OpenAI's pricing page nor LiteLLM "
        "says it searches"
    ),
    "firesearch-ocr-v6": "Fireworks' OCR model",
}


def missed(table: dict[str, Any]) -> list[tuple[str, str]]:
    """(model, provider) of each search-like chat model no entry refuses."""
    found = []
    for name, entry in sorted(table.items()):
        if not isinstance(entry, dict) or entry.get("mode") not in MODES:
            continue
        base = name.rsplit("/", 1)[-1]
        if not SEARCH_LIKE.search(base) or base in NOT_WEB:
            continue
        provider = str(entry.get("litellm_provider") or "")
        if not hosted_retrieval_model(name, provider or None):
            found.append((name, provider))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List search-like models the no-web denylist misses."
    )
    parser.add_argument("--table", help="a price table file (default: LiteLLM main)")
    args = parser.parse_args(argv)
    if args.table:
        with open(args.table, encoding="utf-8") as handle:
            table = json.load(handle)
    else:
        with urllib.request.urlopen(MAIN_TABLE, timeout=60) as response:
            table = json.load(response)
    rows = missed(table)
    for name, provider in rows:
        print(f"not refused: {name} (provider {provider or 'unknown'})")
    print(f"{len(rows)} search-like model(s) not refused of {len(table)} in the table")
    return 1 if rows else 0


if __name__ == "__main__":
    sys.exit(main())
