#!/usr/bin/env python3
"""Rule 1 ≥70% exemption check. VERDICT: dominant=X (N%) / NONE / INSUFFICIENT_IMPORTERS / NO_IMPORTERS."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.py"


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: dominant_symbol.py <file>\n")
        return 2
    target = sys.argv[1]
    out = subprocess.run(
        [sys.executable, str(MANIFEST)], capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        return out.returncode
    rows = json.loads(out.stdout)
    row = next(
        (r for r in rows if r["file"] == target or r["file"].endswith(target)), None
    )
    if row is None:
        sys.stderr.write(f"not in manifest: {target}\n")
        return 1

    selectors: dict[str, list[str]] = row.get("importer_selectors", {})
    total = sum(len(v) for v in selectors.values())
    if total == 0:
        print("VERDICT: NO_IMPORTERS")
        return 0
    if total < 3:
        print(f"VERDICT: INSUFFICIENT_IMPORTERS (total_specific={total})")
        return 0

    ranked = sorted(selectors.items(), key=lambda kv: -len(kv[1]))
    top_sym, top_hits = ranked[0]
    pct = round(100 * len(top_hits) / total)
    if pct >= 70:
        print(f"VERDICT: dominant={top_sym} ({pct}%)")
    else:
        print(f"VERDICT: NONE (top={top_sym} @ {pct}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
