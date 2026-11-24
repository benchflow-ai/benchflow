#!/bin/bash

VERIFIER_DIR="${BENCHFLOW_VERIFIER_DIR:-/verifier}"
if [ ! -d "$VERIFIER_DIR" ] && [ -d /tests ]; then
  VERIFIER_DIR=/tests
fi

REWARD_TEXT="${BENCHFLOW_REWARD_TEXT:-/logs/verifier/reward.txt}"
REWARD_JSON="${BENCHFLOW_REWARD_JSON:-/logs/verifier/reward.json}"
CTRF_JSON="${BENCHFLOW_REWARD_DETAILS_JSON:-${BENCHFLOW_CTRF_JSON:-/logs/verifier/ctrf.json}}"
PYTEST_BIN="${BENCHFLOW_PYTEST_BIN:-pytest}"

mkdir -p "$(dirname "$REWARD_TEXT")" "$(dirname "$REWARD_JSON")" "$(dirname "$CTRF_JSON")"

# pytest and pytest-json-ctrf come with the image (environment/Dockerfile), so
# nothing is installed here and the verifier needs no network.
"$PYTEST_BIN" --ctrf "$CTRF_JSON" "$VERIFIER_DIR/test_outputs.py" -rA -v
PYTEST_EXIT_CODE=$?

if [ $PYTEST_EXIT_CODE -eq 0 ]; then
  REWARD=1.0
else
  REWARD=0.0
fi

printf '%s\n' "$REWARD" > "$REWARD_TEXT"
python3 - "$REWARD" "$REWARD_JSON" <<'PY'
import json
import sys

reward = float(sys.argv[1])
reward_json = sys.argv[2]
with open(reward_json, "w") as f:
    json.dump({"reward": reward}, f)
PY

exit 0
