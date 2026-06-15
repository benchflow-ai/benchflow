#!/usr/bin/env python3
"""Spot-check: list importers of a file (or its dotted module). Tags barrel importers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.py"


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: importers.py <file-or-module>\n")
        return 2
    arg = sys.argv[1]

    out = subprocess.run(
        [sys.executable, str(MANIFEST)], capture_output=True, text=True, check=False
    )
    rows = json.loads(out.stdout)
    row = next(
        (
            r
            for r in rows
            if r["file"] == arg or r["file"].endswith(arg) or r["module"] == arg
        ),
        None,
    )
    if row is None:
        sys.stderr.write(f"not in manifest: {arg}\n")
        return 1

    for imp in row.get("importers", []):
        print(imp)
    for imp in row.get("barrel_importers", []):
        print(f"# barrel: {imp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
