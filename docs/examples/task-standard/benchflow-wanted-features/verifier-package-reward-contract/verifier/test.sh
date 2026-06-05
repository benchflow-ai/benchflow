#!/usr/bin/env bash
set -euo pipefail
cd /repo

python - <<'PY'
from pathlib import Path

expected = [
    Path("src/benchflow/task/verifier_document.py"),
    Path("tests/test_verifier_document.py"),
]
missing = [str(path) for path in expected if not path.exists()]
if missing:
    raise SystemExit(f"missing verifier package files: {missing}")
PY

uv run python -m pytest \
  tests/test_verifier_document.py \
  tests/test_llm_judge_verifier.py \
  tests/test_oracle_chokepoint.py \
  -q

printf "1.0" > /logs/verifier/reward.txt
cat > /logs/verifier/reward.json <<'JSON'
{"reward": 1.0, "items": {"verifier_document": 1.0, "reward_contract": 1.0}, "metadata": {"aggregate_policy": "reward"}}
JSON
cat > /logs/verifier/reward-details.json <<'JSON'
{"criteria": [{"id": "reward_contract", "score": 1.0, "reason": "rich reward artifacts preserved"}]}
JSON

