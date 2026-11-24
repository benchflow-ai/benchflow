#!/bin/bash

# Don't use set -e so we can capture pytest exit code
WORKSPACE="${BENCHFLOW_WORKSPACE:-/root}"
VERIFIER_DIR="${BENCHFLOW_VERIFIER_DIR:-/verifier}"
if [ ! -d "$VERIFIER_DIR" ] && [ -d /tests ]; then
  VERIFIER_DIR=/tests
fi

REWARD_TEXT="${BENCHFLOW_REWARD_TEXT:-/logs/verifier/reward.txt}"
REWARD_JSON="${BENCHFLOW_REWARD_JSON:-/logs/verifier/reward.json}"
CTRF_JSON="${BENCHFLOW_REWARD_DETAILS_JSON:-${BENCHFLOW_CTRF_JSON:-/logs/verifier/ctrf.json}}"
PYTEST_BIN="${BENCHFLOW_PYTEST_BIN:-pytest}"
ARTIFACT_DIR="${BENCHFLOW_VERIFIER_ARTIFACT_DIR:-$(dirname "$CTRF_JSON")}"

# Recalculate formulas using gnumeric (ssconvert) - export to CSV only
# NOTE: Do NOT replace original xlsx with ssconvert output - it destroys formatting
if [ -f "$WORKSPACE/gdp.xlsx" ]; then
    echo "Evaluating Excel formulas with gnumeric..."
    cd "$WORKSPACE"

    # Export CSV with calculated values for test verification
    ssconvert -S "$WORKSPACE/gdp.xlsx" "$WORKSPACE/sheet.csv" 2>/dev/null || true

    echo "Formula evaluation complete."
fi

mkdir -p "$(dirname "$REWARD_TEXT")" "$(dirname "$REWARD_JSON")" "$(dirname "$CTRF_JSON")" "$ARTIFACT_DIR"

# CTRF produces a standard test report in JSON format which is useful for logging.
cd "$WORKSPACE"
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

# Copy artifacts for debugging
if [ -f "$WORKSPACE/gdp.xlsx" ]; then
  cp "$WORKSPACE/gdp.xlsx" "$ARTIFACT_DIR/gdp_modified.xlsx"
fi
# Copy CSV files
cp "$WORKSPACE"/sheet.csv.* "$ARTIFACT_DIR"/ 2>/dev/null || true

exit 0
