"""The hosted client names the service by what it does, not by its product name.

The release leak gate refuses the hosted service's product name in what ships; 0.10.0rc5 shipped
`benchflow/hosted.py` with it on seven lines (docstrings, comments naming the service's own source files, and one
error message), under an allowlist exception that would have expired. The name is built at runtime here, so this
file carries none.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile("bench" + "hub", re.IGNORECASE)


def test_the_hosted_modules_do_not_name_the_service():
    for rel in ("src/benchflow/hosted.py", "src/benchflow/cli/hosted.py"):
        lines = [
            number
            for number, line in enumerate((ROOT / rel).read_text().splitlines(), 1)
            if NAME.search(line)
        ]
        assert lines == [], (rel, lines)


def test_the_hosted_docs_do_not_name_it_either():
    text = (ROOT / "docs" / "evaluate" / "hosted-evals.md").read_text()
    assert not NAME.search(text)
