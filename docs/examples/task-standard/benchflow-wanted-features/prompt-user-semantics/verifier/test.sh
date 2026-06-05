#!/usr/bin/env bash
set -euo pipefail
cd /repo

uv run python -m pytest \
  tests/test_task_document.py \
  tests/test_scene_outbox_trial.py \
  -q

python - <<'PY'
from pathlib import Path
text = Path("src/benchflow/task/document.py").read_text()
if "composition" not in text and "prompt" not in text:
    raise SystemExit("expected prompt composition support in task document/runtime path")
PY

printf "1.0" > /logs/verifier/reward.txt
cat > /logs/verifier/reward.json <<'JSON'
{"reward": 1.0, "items": {"prompt_composition": 1.0, "user_runtime": 1.0}}
JSON

