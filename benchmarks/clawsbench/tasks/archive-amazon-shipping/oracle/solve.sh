#!/usr/bin/env bash
# Oracle solution for archive-amazon-shipping.
#
# Finds the Amazon shipping email by sender and archives it by removing the
# INBOX label via the mock Gmail REST API (localhost:9001), exactly as the
# instruction describes. Demonstrates the task is solvable and gets reward 1.0.
set -euo pipefail

BASE="${CLAW_GMAIL_URL:-http://localhost:9001}"

# Wait for the Gmail service to be reachable.
for _ in $(seq 1 30); do
  if curl -sf "$BASE/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# Look up the message id of the Amazon shipping email.
MSG_ID=$(
  curl -sf "$BASE/gmail/v1/users/me/messages?q=from:shipment-tracking@amazon.com" \
  | python3 -c "import sys, json; msgs = json.load(sys.stdin).get('messages', []); print(msgs[0]['id'] if msgs else '')"
)

if [ -z "$MSG_ID" ]; then
  echo "oracle: Amazon shipping email not found" >&2
  exit 1
fi

# Archive it: remove the INBOX label (keeps the message).
curl -sf -X POST \
  "$BASE/gmail/v1/users/me/messages/$MSG_ID/modify" \
  -H "Content-Type: application/json" \
  -d '{"removeLabelIds": ["INBOX"]}' >/dev/null

echo "oracle: archived message $MSG_ID"
