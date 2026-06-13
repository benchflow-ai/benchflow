#!/bin/bash
set -euo pipefail

python3 - <<'PY'
import json
from pathlib import Path

html = Path("/app/browser_fixture/index.html").read_text()
expected = "browser-use-smoke: ready"
if expected not in html:
    raise SystemExit("fixture did not contain expected status")

Path("/app/final_result.txt").write_text(expected + "\n")
Path("/logs/artifacts/browser-use-smoke-trace.json").write_text(
    json.dumps(
        {
            "framework": "benchflow-oracle-browser-use-smoke",
            "steps": ["read local browser fixture", "write final result"],
            "screenshots_b64": [],
            "final_result": expected,
        },
        indent=2,
    )
    + "\n"
)
PY
