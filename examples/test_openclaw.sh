#!/usr/bin/env bash
# Test openclaw agent across providers via Google Vertex AI and Z.AI.
#
# Prerequisites:
#   - gcloud ADC configured + GOOGLE_CLOUD_PROJECT set (for Gemini and Sonnet via Vertex AI)
#   - ZAI_API_KEY set (for Z.AI GLM-5)
#   - Docker running
#
# Usage:
#   source .env && bash examples/test_openclaw.sh           # run all
#   source .env && bash examples/test_openclaw.sh gemini     # run one
#   source .env && bash examples/test_openclaw.sh zai vertex # run subset

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
AGENT="openclaw"
ENV="docker"
JOBS_DIR="jobs/test-openclaw"
PROJECT="${GOOGLE_CLOUD_PROJECT:-skillsbench}"

# ── Model definitions ──
# key=label, value=model string + extra args
declare -A MODELS
MODELS=(
  [gemini]="google-vertex/gemini-3-flash-preview"
  [sonnet]="anthropic-vertex/claude-sonnet-4-6"
  [zai-glm5]="zai/glm-5"
)

# Extra --ae flags per model (if any)
declare -A EXTRA_ARGS
EXTRA_ARGS=(
  [gemini]="--ae GOOGLE_CLOUD_PROJECT=$PROJECT"
  [sonnet]="--ae GOOGLE_CLOUD_PROJECT=$PROJECT"
)

# ── Pre-flight checks ──

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
    gemini|sonnet)
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
  SELECTED=("gemini" "sonnet" "zai-glm5")
fi

echo "=== openclaw provider sweep ==="
echo "Task:   $TASK"
echo "Agent:  $AGENT"
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
