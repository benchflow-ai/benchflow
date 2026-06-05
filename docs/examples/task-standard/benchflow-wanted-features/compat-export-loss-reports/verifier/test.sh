#!/usr/bin/env bash
set -euo pipefail
cd /repo

python - <<'PY'
from pathlib import Path

candidates = [
    Path("src/benchflow/adapters/export.py"),
    Path("src/benchflow/adapters/harbor_export.py"),
    Path("src/benchflow/task/export.py"),
]
if not any(path.exists() for path in candidates):
    raise SystemExit("expected a native-to-Harbor/Pier export module")

tests = [
    Path("tests/test_task_export.py"),
    Path("tests/test_harbor_export.py"),
]
if not any(path.exists() for path in tests):
    raise SystemExit("expected export loss report tests")
PY

uv run python -m pytest \
  tests/test_task_config.py \
  tests/test_task_document.py \
  tests/test_tasks.py \
  -q

printf "1.0" > /logs/verifier/reward.txt
cat > /logs/verifier/reward.json <<'JSON'
{"reward": 1.0, "items": {"split_export": 1.0, "loss_report": 1.0}}
JSON

