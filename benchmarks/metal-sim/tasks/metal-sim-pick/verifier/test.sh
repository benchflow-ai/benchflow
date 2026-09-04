#!/bin/bash
# Verifier: run the agent's /app/policy.py through the simulator on held-out seeds.
# Writes /logs/verifier/reward.txt (success rate 0.0-1.0), reward.json, reward-details.json.
# APP_DIR / LOG_ROOT overrides let this run outside a sandbox (see bench/README.md).
set -uo pipefail
APP="${APP_DIR:-/app}"
LOGS="${LOG_ROOT:-/logs}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$LOGS/verifier" "$LOGS/artifacts"
SUMMARY="$LOGS/verifier/metal-sim-eval.json"
EVAL_LOG="$LOGS/verifier/eval.log"
rm -f "$SUMMARY"
if [ ! -f "$APP/policy.py" ]; then
  echo "no policy at $APP/policy.py" > "$EVAL_LOG"
else
  metal-sim-eval --task metal-sim-pick -T num_scenes=6 -T max_steps=200 \
    --policy-file "$APP/policy.py" --seed 4242 --no-frames \
    --log-dir "$LOGS/artifacts/inspect-robots" -E trace_dir="$LOGS/artifacts/metal_sim_traces" \
    --json "$SUMMARY" > "$EVAL_LOG" 2>&1 || true
fi
python3 "$HERE/reward.py" "$SUMMARY" "$LOGS/verifier" "$EVAL_LOG"
