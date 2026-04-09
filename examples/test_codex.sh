#!/usr/bin/env bash
# Test codex-acp agent: subscription, API key.
#
# Prerequisites:
#   - OPENAI_API_KEY set in .env, or logged in via `codex --login` (for subscription)
#   - Docker running, or DAYTONA_API_KEY + DAYTONA_API_URL set for --daytona
#
# Usage:
#   bash examples/test_codex.sh                  # run all
#   bash examples/test_codex.sh subscription     # subscription auth only
#   bash examples/test_codex.sh apikey           # API key only
#   bash examples/test_codex.sh --daytona        # use Daytona

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source .env from repo root if it exists
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

TASK="examples/hello-world-task"
ENV="${ENV:-docker}"
ARGS=()
for arg in "$@"; do
  case "$arg" in --daytona) ENV="daytona" ;; *) ARGS+=("$arg") ;; esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

AGENT="codex-acp"
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

# ── Model definitions ──
declare -A MODELS
MODELS=(
  [subscription]="gpt-5.4"
  [apikey]="gpt-5.4"
)

# Extra --ae flags per model
declare -A EXTRA_ARGS
EXTRA_ARGS=(
  [subscription]="--ae OPENAI_REASONING_EFFORT=$REASONING"
  [apikey]="--ae OPENAI_REASONING_EFFORT=$REASONING"
)

# ── Pre-flight checks ──

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

check_env() {
  local label="$1"
  case "$label" in
    subscription)
      if [ -n "${OPENAI_API_KEY:-}" ]; then
        echo "NOTE: $label — OPENAI_API_KEY is set, will use API key (not subscription)"
      elif [ ! -f "$HOME/.codex/auth.json" ]; then
        echo "SKIP: $label — no OPENAI_API_KEY and no ~/.codex/auth.json (run: codex --login)"
        return 1
      fi ;;
    apikey)
      if [ -n "${OPENAI_API_KEY:-}" ]; then
        echo "NOTE: $label — using OPENAI_API_KEY (API key auth)"
      else
        echo "SKIP: $label — OPENAI_API_KEY not set"
        return 1
      fi ;;
  esac
  return 0
}

# ── Determine which models to test ──

if [ $# -gt 0 ]; then
  SELECTED=("$@")
else
  SELECTED=("subscription" "apikey")
fi

echo "=== $AGENT provider sweep ==="
echo "Task:   $TASK"
echo "Agent:  $AGENT"
echo "Env:    $ENV"
echo "Effort: $REASONING"
echo "Models: ${SELECTED[*]}"
echo ""

PASS=0
FAIL=0
SKIP=0

for label in "${SELECTED[@]}"; do
  model="${MODELS[$label]:-}"
  if [ -z "$model" ]; then
    echo "ERROR: unknown model label '$label'"
    echo "  Available: ${!MODELS[*]}"
    exit 1
  fi

  echo "--- $label: $model ---"

  if ! check_env "$label"; then
    SKIP=$((SKIP + 1))
    echo ""
    continue
  fi

  extra="${EXTRA_ARGS[$label]:-}"

  # shellcheck disable=SC2086
  if uv run benchflow run \
    -t "$TASK" \
    -a "$AGENT" \
    -m "$model" \
    -e "$ENV" \
    --jobs-dir "$JOBS_DIR" \
    $extra; then
    echo "PASS: $label"
    PASS=$((PASS + 1))
  else
    echo "FAIL: $label"
    show_failure "$JOBS_DIR"
    FAIL=$((FAIL + 1))
  fi
  echo ""
done

# ── Summary ──

TOTAL=$((PASS + FAIL + SKIP))
echo "=== Summary ==="
echo "Passed:  $PASS / $TOTAL"
echo "Failed:  $FAIL / $TOTAL"
echo "Skipped: $SKIP / $TOTAL"

if [ "$FAIL" -gt 0 ]; then
  echo ""
  echo "Check jobs output: ls $JOBS_DIR/"
  exit 1
fi
