#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
reward=0.0
trace=/logs/artifacts/computer-use-smoke-trace.json

if [ -f /app/computer_use_result.txt ] \
  && [ -f /app/computer_use_roundtrip.txt ] \
  && [ "$(tr -d '\n' < /app/computer_use_result.txt)" = "computer-use-smoke: ready" ] \
  && [ "$(tr -d '\n' < /app/computer_use_roundtrip.txt)" = "computer-use-smoke: ready" ] \
  && [ -s /logs/artifacts/computer-use-smoke.png ] \
  && [ -f "$trace" ] \
  && grep -q '"final_result": "computer-use-smoke: ready"' "$trace" \
  && grep -q '"screenshots_b64":' "$trace"; then
  reward=1.0
fi

printf '%s\n' "$reward" > /logs/verifier/reward.txt
printf '{"reward": %s}\n' "$reward" > /logs/verifier/reward.json
