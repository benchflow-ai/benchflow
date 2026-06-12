from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pages(config: dict[str, Any]) -> list[str]:
    groups = config["navigation"]["groups"]
    return [page for group in groups for page in group["pages"]]


def test_mintlify_root_configs_cover_the_same_pages():
    """Guards the fix for PR #665 where Mintlify looked for config under docs/."""
    root_config = json.loads((REPO_ROOT / "docs.json").read_text(encoding="utf-8"))
    docs_config = json.loads(
        (REPO_ROOT / "docs" / "docs.json").read_text(encoding="utf-8")
    )

    root_pages = _pages(root_config)
    docs_pages = _pages(docs_config)

    assert root_pages == [f"docs/{page}" for page in docs_pages]
    for page in docs_pages:
        assert (REPO_ROOT / "docs" / f"{page}.md").is_file()
