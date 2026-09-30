"""Verbatim copies of task.md draft 2's reference tools.

``taskmd.py`` is the reference parser and checker (``tools/taskmd.py``) and
``judgeprompt.py`` the ``judge-prompt@1`` compiler (``tools/judgeprompt.py``)
of task.md's spec repository, at the commit ``VENDOR.json`` pins. They are
byte-identical to the pinned files, so BenchFlow parses a package exactly as
the reference parser does. Never edit them here: copy them again from the
pinned repository and update ``VENDOR.json``; ``tests/test_taskmd_vendor.py``
checks both. Lint and type checks skip this folder (``pyproject.toml``).
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
