#!/usr/bin/env bash
set -euo pipefail
cd /repo

python - <<'PY'
from pathlib import Path

required = [
    Path("src/benchflow/task/runtime_capabilities.py"),
    Path("tests/test_runtime_capabilities.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing runtime capability files: {missing}")

runtime_view = Path("src/benchflow/task/runtime_view.py")
task_package = Path("src/benchflow/task/package.py")
if not runtime_view.exists() and not task_package.exists():
    raise SystemExit("expected a TaskRuntimeView/TaskPackage module")
PY

uv run python -m pytest \
  tests/test_runtime_capabilities.py \
  tests/test_task_document.py \
  tests/test_task_config.py \
  tests/test_tasks.py \
  -q

printf "1.0" > /logs/verifier/reward.txt
cat > /logs/verifier/reward.json <<'JSON'
{"reward": 1.0, "status": "scored", "items": {"runtime_view": 1.0, "fail_closed": 1.0}}
JSON

