#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${BENCHFLOW_WORKSPACE:-/workspace}"
mkdir -p "$WORKSPACE"

printf 'BF-1042\n' > "$WORKSPACE/order_id.txt"
cat > "$WORKSPACE/recovery.json" <<'JSON'
{
  "order_id": "BF-1042",
  "failure_point": "carrier handoff scan missing after warehouse release",
  "next_action": "open a carrier trace and send the customer the trace reference"
}
JSON
