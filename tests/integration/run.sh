#!/usr/bin/env bash
# Integration test runner — drives bench eval create per agent.
#
# All agents launch in parallel; each agent runs its 9 tasks concurrently
# via Daytona (concurrency=64 by default). The script waits for every agent to finish,
# then runs check_results.py to validate outputs.
#
# Usage:
#   tests/integration/run.sh                    # all agents
#   tests/integration/run.sh gemini pi-acp      # specific agents
#   tests/integration/run.sh --check-only       # review existing results
#
# Required env vars:
#   GEMINI_API_KEY (or GOOGLE_API_KEY)
#   DAYTONA_API_KEY
#   CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY  (for claude-agent-acp)
#   OPENAI_API_KEY                                (for codex-acp)

set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

INTEGRATION_CONCURRENCY="${BENCHFLOW_INTEGRATION_CONCURRENCY:-64}"
ALLOW_SKIPS="${BENCHFLOW_INTEGRATION_ALLOW_SKIPS:-false}"

# The 9 selected SkillsBench tasks for integration testing.
SELECTED_TASKS=(
  jax-computing-basics
  python-scala-translation
  jpg-ocr-stat
  grid-dispatch-operator
  threejs-to-obj
  data-to-d3
  lake-warming-attribution
  weighted-gdp-calc
  shock-analysis-supply
)

# Agent → model mapping. Agents not listed use the default Gemini model.
DEFAULT_MODEL="gemini-3.1-flash-lite-preview"

model_for_agent() {
  case "$1" in
    claude-agent-acp) echo "claude-haiku-4-5-20251001" ;;
    codex-acp)        echo "gpt-5.4-nano" ;;
    mimo)             echo "xiaomi/mimo-v2.5-pro" ;;
    *)                echo "$DEFAULT_MODEL" ;;
  esac
}

ALL_AGENTS=(
  claude-agent-acp
  pi-acp
  openclaw
  codex-acp
  gemini
  opencode
  harvey-lab-harness
  openhands
  mimo
)

# ── Parse args ──────────────────────────────────────────────────────
CHECK_ONLY=false
AGENTS=()
for arg in "$@"; do
  case "$arg" in
    --check-only) CHECK_ONLY=true ;;
    *)            AGENTS+=("$arg") ;;
  esac
done

if [ ${#AGENTS[@]} -eq 0 ]; then
  AGENTS=("${ALL_AGENTS[@]}")
fi

if [ "$CHECK_ONLY" = true ]; then
  if [ -n "${BENCHFLOW_INTEGRATION_JOBS_ROOT:-}" ]; then
    JOBS_ROOT="$BENCHFLOW_INTEGRATION_JOBS_ROOT"
  elif [ -n "${BENCHFLOW_INTEGRATION_RUN_ID:-}" ]; then
    JOBS_ROOT="jobs/integration-$BENCHFLOW_INTEGRATION_RUN_ID"
  else
    echo "ERROR: --check-only requires BENCHFLOW_INTEGRATION_JOBS_ROOT or BENCHFLOW_INTEGRATION_RUN_ID"
    exit 1
  fi
else
  RUN_ID="${BENCHFLOW_INTEGRATION_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
  JOBS_ROOT="${BENCHFLOW_INTEGRATION_JOBS_ROOT:-jobs/integration-$RUN_ID}"
fi
LOG_DIR="$JOBS_ROOT/.logs"
CHECK_AGENTS=("${AGENTS[@]}")

# ── Credential checks ──────────────────────────────────────────────
# mimo additionally needs XIAOMI_API_KEY + XIAOMI_BASE_URL (registered
# `xiaomi` provider — resolve_agent_env fails closed without both).
has_gemini_key() {
  [ -n "${GEMINI_API_KEY:-}" ] || [ -n "${GOOGLE_API_KEY:-}" ]
}

has_creds_for() {
  case "$1" in
    claude-agent-acp)
      [ -n "${ANTHROPIC_API_KEY:-}" ] || \
      [ -n "${ANTHROPIC_AUTH_TOKEN:-}" ] || \
      [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]
      ;;
    codex-acp)
      [ -n "${OPENAI_API_KEY:-}" ]
      ;;
    mimo)
      # Xiaomi MiMo platform (registered `xiaomi` provider) — both required.
      [ -n "${XIAOMI_API_KEY:-}" ] && [ -n "${XIAOMI_BASE_URL:-}" ]
      ;;
    *)
      has_gemini_key
      ;;
  esac
}

# ── Run evals (all agents in parallel) ──────────────────────────────
if [ "$CHECK_ONLY" = false ]; then
  if [ -z "${DAYTONA_API_KEY:-}" ]; then
    echo "ERROR: DAYTONA_API_KEY required" >&2
    exit 1
  fi

  echo "Using ${#SELECTED_TASKS[@]} source-configured SkillsBench tasks"

  mkdir -p "$LOG_DIR"
  PIDS=()
  LAUNCHED=()
  RUNNABLE=()
  SKIPPED=()

  for agent in "${AGENTS[@]}"; do
    if ! has_creds_for "$agent"; then
      SKIPPED+=("$agent — no credentials")
      continue
    fi

    config_file="tests/integration/configs/$agent.yaml"
    if [ ! -f "$config_file" ]; then
      SKIPPED+=("$agent — missing $config_file")
      continue
    fi
    RUNNABLE+=("$agent")
  done

  if [ ${#SKIPPED[@]} -ne 0 ]; then
    for skipped in "${SKIPPED[@]}"; do
      echo "SKIP $skipped"
    done
    if [ "$ALLOW_SKIPS" != true ]; then
      echo "ERROR: requested agents were skipped; set BENCHFLOW_INTEGRATION_ALLOW_SKIPS=true for exploratory partial runs"
      exit 1
    fi
  fi

  if [ ${#RUNNABLE[@]} -eq 0 ]; then
    echo "ERROR: no agents launched"
    exit 1
  fi

  for agent in "${RUNNABLE[@]}"; do
    model="$(model_for_agent "$agent")"
    config_file="tests/integration/configs/$agent.yaml"
    echo "Launching $agent (model=$model, config=$config_file)..."
    uv run bench eval create \
      --config "$config_file" \
      --jobs-dir "$JOBS_ROOT/$agent" \
      --concurrency "$INTEGRATION_CONCURRENCY" \
      > "$LOG_DIR/$agent.log" 2>&1 &
    PIDS+=($!)
    LAUNCHED+=("$agent")
  done

  echo ""
  echo "${#LAUNCHED[@]} agents launched in parallel. Waiting..."
  echo ""
  CHECK_AGENTS=("${LAUNCHED[@]}")

  # Wait for all and report as each finishes. bench eval create may exit
  # non-zero when trials fail or verifiers reject agent output; audit anyway.
  EVAL_WARNINGS=0
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    agent="${LAUNCHED[$i]}"
    if wait "$pid"; then
      echo "✓ $agent finished — $(tail -1 "$LOG_DIR/$agent.log")"
    else
      echo "⚠ $agent exited $? — see $LOG_DIR/$agent.log (continuing to audit)"
      EVAL_WARNINGS=$((EVAL_WARNINGS + 1))
    fi
  done

  echo ""
  echo "${#LAUNCHED[@]} agents done, $EVAL_WARNINGS exited non-zero."
fi

# ── Check results ───────────────────────────────────────────────────
echo ""
echo "══════ Results ══════"
EXPECTED_CHECK_ARGS=("environment=daytona" "concurrency=$INTEGRATION_CONCURRENCY")
for agent in "${CHECK_AGENTS[@]}"; do
  EXPECTED_CHECK_ARGS+=("$agent.model=$(model_for_agent "$agent")")
done
uv run python tests/integration/check_results.py \
  "$JOBS_ROOT" \
  "${CHECK_AGENTS[@]}" \
  "${EXPECTED_CHECK_ARGS[@]}"
exit $?
