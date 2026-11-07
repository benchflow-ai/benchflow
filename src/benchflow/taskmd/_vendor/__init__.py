"""Pinned task.md reference tools with reviewed local naming changes.

VENDOR.json retains the upstream commit and source hashes alongside installed
hashes. The parser uses neutral diagnostic wording and import.toml metadata.
The judge-prompt compiler is unchanged. Lint and type checks skip this folder.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

VENDOR_DIR = Path(__file__).resolve().parent


@cache
def vendor_pin() -> dict[str, Any]:
    """The pin: repository, commit, and each vendored file's SHA-256."""
    return json.loads((VENDOR_DIR / "VENDOR.json").read_text(encoding="utf-8"))
