#!/usr/bin/env bash
# Integration test runner — drives bench eval create per agent.
#
# All agents launch in parallel; each agent runs its 9 tasks concurrently
# via Daytona (concurrency=30). The script waits for every agent to finish,
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

JOBS_ROOT="jobs/integration"
LOG_DIR="$JOBS_ROOT/.logs"

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
declare -A AGENT_MODELS=(
  [claude-agent-acp]="claude-haiku-4-5-20251001"
  [codex-acp]="gpt-5.4-nano"
)

ALL_AGENTS=(
  claude-agent-acp
  pi-acp
  openclaw
  codex-acp
  gemini
  opencode
  harvey-lab-harness
  openhands
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

# ── Credential checks ──────────────────────────────────────────────
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
    *)
      has_gemini_key
      ;;
  esac
}

# ── Prepare task subset ────────────────────────────────────────────
prepare_tasks_dir() {
  local full_dir
  full_dir=$(uv run python -c "
from benchflow.task_download import resolve_source
print(resolve_source('benchflow-ai/skillsbench', path='tasks', ref='main'))
")
  local subset_dir="$JOBS_ROOT/.tasks-subset"
  rm -rf "$subset_dir"
  mkdir -p "$subset_dir"
  for task in "${SELECTED_TASKS[@]}"; do
    if [ -d "$full_dir/$task" ]; then
      ln -s "$full_dir/$task" "$subset_dir/$task"
    else
      echo "WARN: task $task not found in $full_dir" >&2
    fi
  done
  echo "$subset_dir"
}

# ── Run evals (all agents in parallel) ──────────────────────────────
if [ "$CHECK_ONLY" = false ]; then
  if [ -z "${DAYTONA_API_KEY:-}" ]; then
    echo "ERROR: DAYTONA_API_KEY required" >&2
    exit 1
  fi

  echo "Resolving tasks..."
  TASKS_DIR=$(prepare_tasks_dir)
  echo "Using ${#SELECTED_TASKS[@]} tasks from $TASKS_DIR"

  mkdir -p "$LOG_DIR"
  PIDS=()
  LAUNCHED=()

  for agent in "${AGENTS[@]}"; do
    if ! has_creds_for "$agent"; then
      echo "SKIP $agent — no credentials"
      continue
    fi

    model="${AGENT_MODELS[$agent]:-$DEFAULT_MODEL}"
    echo "Launching $agent (model=$model)..."
    uv run bench eval create \
      -t "$TASKS_DIR" \
      -a "$agent" \
      -m "$model" \
      --sandbox daytona \
      -c 30 \
      -o "$JOBS_ROOT/$agent" \
      > "$LOG_DIR/$agent.log" 2>&1 &
    PIDS+=($!)
    LAUNCHED+=("$agent")
  done

  echo ""
  echo "${#LAUNCHED[@]} agents launched in parallel. Waiting..."
  echo ""

  # Wait for all and report as each finishes.
  FAILURES=0
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    agent="${LAUNCHED[$i]}"
    if wait "$pid"; then
      echo "✓ $agent finished — $(tail -1 "$LOG_DIR/$agent.log")"
    else
      echo "✗ $agent failed (exit $?) — see $LOG_DIR/$agent.log"
      FAILURES=$((FAILURES + 1))
    fi
  done

  echo ""
  echo "${#LAUNCHED[@]} agents done, $FAILURES failed."
fi

# ── Check results ───────────────────────────────────────────────────
echo ""
echo "══════ Results ══════"
uv run python tests/integration/check_results.py "$JOBS_ROOT" "${AGENTS[@]}"
