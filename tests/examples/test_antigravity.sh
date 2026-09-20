#!/usr/bin/env bash
# Test the antigravity agent (Google Antigravity CLI, `agy`) with a Gemini API key.
#
# Prerequisites:
#   - GEMINI_API_KEY set (agy runs in Gemini API-key mode inside the sandbox;
#     Google sign-in lives in the OS keyring and cannot be copied into a sandbox)
#   - Docker running, or DAYTONA_API_KEY set for --daytona
#
# Usage:
#   bash examples/test_antigravity.sh             # gemini-3.8-flash at the default (high) effort
#   bash examples/test_antigravity.sh low         # explicit --reasoning-effort low
#   bash examples/test_antigravity.sh --daytona   # use Daytona

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

TASK="$SCRIPT_DIR/hello-world-task"
ENV="${ENV:-docker}"
ARGS=()
for arg in "$@"; do
  case "$arg" in --daytona) ENV="daytona" ;; *) ARGS+=("$arg") ;; esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

AGENT="antigravity"
MODEL="${MODEL:-gemini-3.8-flash}"
JOBS_DIR="jobs/test-antigravity"

if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "SKIP: GEMINI_API_KEY not set"
  exit 0
fi
if [ "$ENV" = "daytona" ] && [ -z "${DAYTONA_API_KEY:-}" ]; then
  echo "ERROR: DAYTONA_API_KEY not set (check .env)"
  exit 1
fi

EXTRA=()
if [ $# -gt 0 ]; then
  EXTRA+=(--reasoning-effort "$1")
fi

echo "=== $AGENT smoke test ==="
echo "Task:   $TASK"
echo "Model:  $MODEL"
echo "Env:    $ENV"
echo ""

if uv run bench eval run \
  --tasks-dir "$TASK" \
  --agent "$AGENT" \
  --model "$MODEL" \
  --sandbox "$ENV" \
  --jobs-dir "$JOBS_DIR" \
  "${EXTRA[@]}"; then
  echo "PASS"
else
  echo "FAIL — check jobs output: ls $JOBS_DIR/"
  exit 1
fi
