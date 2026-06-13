#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier /logs/artifacts

expected="browser-use-smoke: ready"
actual=""
if [ -f /app/final_result.txt ]; then
    actual="$(tr -d '\n' < /app/final_result.txt)"
fi

if [ "$actual" = "$expected" ]; then
    reward=1.0
else
    reward=0.0
fi

printf "%s\n" "$reward" > /logs/verifier/reward.txt
printf '{"reward": %s}\n' "$reward" > /logs/verifier/reward.json
printf '{"expected": "%s", "actual": "%s"}\n' \
    "$expected" "$actual" > /logs/artifacts/browser-use-smoke-verifier.json
