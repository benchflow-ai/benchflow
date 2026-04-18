#!/usr/bin/env bash
# Run the NanoFirm demo: Gemini 3.1 Pro as the "firm brain" orchestrator,
# Claude Opus 4.6 playing both the adversary (landlord firm) and the
# tenant-firm counterpart agents inside a Docker environment.
set -euo pipefail

BENCHFLOW_REPO="/workspace/repos/benchflow"
BENCHFLOW_BIN="$BENCHFLOW_REPO/.venv/bin/benchflow"

cd "$BENCHFLOW_REPO"

if [[ ! -x "$BENCHFLOW_BIN" ]]; then
  echo "benchflow CLI not found at $BENCHFLOW_BIN" >&2
  exit 1
fi

exec "$BENCHFLOW_BIN" run \
  -t labs/nanofirm/task \
  -a gemini \
  -m gemini-3.1-pro-preview \
  -e docker \
  --ae ADVERSARY_MODEL=claude-opus-4-6 \
  --ae TENANT_FIRM_MODEL=claude-opus-4-6
