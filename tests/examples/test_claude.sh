#!/usr/bin/env bash
# Test claude-agent-acp across providers: subscription, Vertex Sonnet, Z.AI GLM-5.
#
# Prerequisites:
#   - ANTHROPIC_API_KEY set, or logged in via `claude login` (for subscription)
#   - gcloud ADC configured + GOOGLE_CLOUD_PROJECT set (for Vertex Sonnet)
#   - ZAI_API_KEY set (for Z.AI GLM-5)
#   - Docker running, or DAYTONA_API_KEY + DAYTONA_API_URL set for --daytona
#
# Usage:
#   bash examples/test_claude.sh                  # run all
#   bash examples/test_claude.sh subscription     # subscription auth only
#   bash examples/test_claude.sh zai-glm5         # run one
#   bash examples/test_claude.sh --daytona        # use Daytona

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source .env from repo root if it exists
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
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

AGENT="claude-agent-acp"
JOBS_DIR="jobs/test-claude"
PROJECT="${GOOGLE_CLOUD_PROJECT:-skillsbench}"

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
  [subscription]="claude-sonnet-4-6"
  [sonnet]="anthropic-vertex/claude-sonnet-4-6"
  [zai-glm5]="zai/glm-5.1"
)

# Extra --ae flags per model
declare -A EXTRA_ARGS
EXTRA_ARGS=(
  [subscription]=""
  [sonnet]="--ae CLAUDE_CODE_USE_VERTEX=1 --ae GOOGLE_CLOUD_PROJECT=$PROJECT --ae GOOGLE_CLOUD_LOCATION=global"
  [zai-glm5]="--ae ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic --ae ANTHROPIC_AUTH_TOKEN=$ZAI_API_KEY"
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

check_vertex() {
  local label="$1"
  if [ -z "${GOOGLE_CLOUD_PROJECT:-}" ]; then
    echo "SKIP: $label — GOOGLE_CLOUD_PROJECT not set"
    return 1
  fi
  local adc="$HOME/.config/gcloud/application_default_credentials.json"
  if [ ! -f "$adc" ]; then
    echo "SKIP: $label — ADC not found at $adc (run: gcloud auth application-default login)"
    return 1
  fi
  return 0
}

check_env() {
  local label="$1"
  case "$label" in
    subscription)
      if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
        echo "NOTE: $label — ANTHROPIC_API_KEY is set, will use API key (not subscription)"
      elif [ ! -f "$HOME/.claude/.credentials.json" ]; then
        echo "SKIP: $label — no ANTHROPIC_API_KEY and no ~/.claude/.credentials.json (run: claude login)"
        return 1
      fi ;;
    sonnet)
      check_vertex "$label" ;;
    zai-glm5)
      if [ -z "${ZAI_API_KEY:-}" ]; then
        echo "SKIP: $label — ZAI_API_KEY not set"
        return 1
      fi ;;
  esac
  return 0
}

# ── Determine which models to test ──

if [ $# -gt 0 ]; then
  SELECTED=("$@")
else
  SELECTED=("subscription" "sonnet" "zai-glm5")
fi

echo "=== $AGENT provider sweep ==="
echo "Task:   $TASK"
echo "Agent:  $AGENT"
echo "Env:    $ENV"
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
