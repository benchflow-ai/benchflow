#!/usr/bin/env bash
# Verifier for archive-amazon-shipping.
#
# The Gmail mock service runs inside this same sandbox on localhost:9001
# (started by the BenchFlow Environment plane from clawsbench/environment.toml).
# We read the full state dump from the service's /_admin/state endpoint and
# check that the Amazon shipping email was archived (INBOX label removed)
# without being trashed or deleted.
set -uo pipefail

BASE="${CLAW_GMAIL_URL:-http://localhost:9001}"
LOGS_DIR="${LOGS_DIR:-/logs/verifier}"
VERIFIER_DIR="${BENCHFLOW_VERIFIER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [ ! -d "$VERIFIER_DIR" ] && [ -d /tests ]; then
  VERIFIER_DIR=/tests
fi
mkdir -p "$LOGS_DIR"

write_reward() {
  local reward="$1"
  local reason="$2"
  printf '%s\n' "$reward" > "$LOGS_DIR/reward.txt"
  python3 - "$reward" "$reason" "$LOGS_DIR/reward.json" "$LOGS_DIR/reward-details.json" <<'PY'
import json
import sys

reward = float(sys.argv[1])
reason = sys.argv[2]
reward_json = sys.argv[3]
details_json = sys.argv[4]

with open(reward_json, "w") as f:
    json.dump({"reward": reward}, f)

with open(details_json, "w") as f:
    json.dump({"reward": reward, "reason": reason}, f)
PY
}

# The service should already be up (readiness was gated before the agent ran),
# but poll briefly in case the verifier starts before a slow process settles.
for _ in $(seq 1 30); do
  if curl -sf "$BASE/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -sf "$BASE/_admin/state" -o /tmp/gmail_state.json; then
  echo "verifier: could not reach $BASE/_admin/state" >&2
  write_reward "0.0" "gmail-state-unreachable"
  exit 0
fi

python3 "$VERIFIER_DIR/evaluate.py" \
  --state /tmp/gmail_state.json \
  --output "$LOGS_DIR/reward.txt" \
  --reward-json "$LOGS_DIR/reward.json" \
  --details-json "$LOGS_DIR/reward-details.json"

cat "$LOGS_DIR/reward.txt"
