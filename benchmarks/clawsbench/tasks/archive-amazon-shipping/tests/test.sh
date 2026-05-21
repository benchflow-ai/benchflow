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
mkdir -p "$LOGS_DIR"

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
  echo "0.0" > "$LOGS_DIR/reward.txt"
  exit 0
fi

python3 "$(dirname "$0")/evaluate.py" \
  --state /tmp/gmail_state.json \
  --output "$LOGS_DIR/reward.txt"

cat "$LOGS_DIR/reward.txt"
