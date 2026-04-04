#!/usr/bin/env bash
# Test OpenAI API key + gpt-5.4 with codex-acp agent.
#
# Prerequisites:
#   - OPENAI_API_KEY set in .env
#   - Docker running, or DAYTONA_API_KEY + DAYTONA_API_URL set for --daytona
#
# Usage:
#   bash examples/test_codex.sh
#   bash examples/test_codex.sh --daytona

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source .env from repo root if it exists
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

ENV="${ENV:-docker}"
for arg in "$@"; do
  case "$arg" in --daytona) ENV="daytona" ;; esac
done

TASK="examples/hello-world-task"
AGENT="codex-acp"
MODEL="gpt-5.4"
JOBS_DIR="jobs/test-codex"
REASONING="${REASONING:-high}"  # none, low, medium, high

# ── Helpers ──

show_failure() {
  local dir="$1"
  local latest
  latest=$(ls -td "$dir"/*/ 2>/dev/null | head -1)
  if [ -z "$latest" ]; then return; fi
  local agent_log
  agent_log=$(ls -t "$latest"/agent/*.txt 2>/dev/null | head -1)
  if [ -n "$agent_log" ]; then
    echo "  Last 20 lines of $agent_log:"
    tail -20 "$agent_log" | sed 's/^/    /'
  fi
  if [ -f "$latest/result.json" ]; then
    local err
    err=$(python3 -c "import json,sys; r=json.load(open('$latest/result.json')); print(r.get('error',''))" 2>/dev/null)
    if [ -n "$err" ]; then
      echo "  Error: $err"
    fi
  fi
}

# ── Pre-flight ──

if [ "$ENV" = "daytona" ]; then
  if [ -z "${DAYTONA_API_KEY:-}" ]; then
    echo "ERROR: DAYTONA_API_KEY not set (check .env)"
    exit 1
  fi
  if [ -z "${DAYTONA_API_URL:-}" ]; then
    echo "ERROR: DAYTONA_API_URL not set (check .env)"
    exit 1
  fi
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY not set (check .env)"
  exit 1
fi

# ── Run ──

echo "=== $AGENT + $MODEL ==="
echo "Task:   $TASK"
echo "Agent:  $AGENT"
echo "Model:  $MODEL"
echo "Env:    $ENV"
echo "Effort: $REASONING"
echo ""

# shellcheck disable=SC2086
if uv run benchflow run \
  -t "$TASK" \
  -a "$AGENT" \
  -m "$MODEL" \
  -e "$ENV" \
  --jobs-dir "$JOBS_DIR" \
  --ae OPENAI_REASONING_EFFORT="$REASONING"; then
  echo "PASS"
else
  echo "FAIL"
  show_failure "$JOBS_DIR"
  exit 1
fi
