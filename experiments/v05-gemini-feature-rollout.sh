#!/usr/bin/env bash
# v0.5 gemini ACP feature smoke: acp_smoke, terminal-bench, SkillsBench reps.
#
# Usage:
#   GEMINI_API_KEY=... DAYTONA_API_KEY=... experiments/v05-gemini-feature-rollout.sh
#   experiments/v05-gemini-feature-rollout.sh --check-only jobs/codex-feature-rollouts-<run-id>
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"

CONCURRENCY="${BENCHFLOW_INTEGRATION_CONCURRENCY:-64}"
AGENT="${BENCHFLOW_FEATURE_AGENT:-gemini}"
MODEL="${BENCHFLOW_FEATURE_MODEL:-gemini-3.1-flash-lite-preview}"
SANDBOX="${BENCHFLOW_FEATURE_SANDBOX:-daytona}"
IDLE_TIMEOUT="${BENCHFLOW_AGENT_IDLE_TIMEOUT:-600}"

CHECK_ONLY=false
JOBS_ROOT=""

for arg in "$@"; do
  case "$arg" in
    --check-only) CHECK_ONLY=true ;;
    *)
      if [ -z "$JOBS_ROOT" ]; then
        JOBS_ROOT="$arg"
      fi
      ;;
  esac
done

if [ "$CHECK_ONLY" = true ]; then
  if [ -z "$JOBS_ROOT" ]; then
    echo "ERROR: --check-only requires jobs root path" >&2
    exit 1
  fi
  uv run python tests/integration/check_results.py \
    "$JOBS_ROOT" \
    "agent=$AGENT" \
    "model=$MODEL" \
    "environment=$SANDBOX" \
    "concurrency=$CONCURRENCY" \
    "agent_idle_timeout_sec=$IDLE_TIMEOUT"
  exit $?
fi

if [ -z "${GEMINI_API_KEY:-}" ] && [ -z "${GOOGLE_API_KEY:-}" ]; then
  echo "ERROR: GEMINI_API_KEY or GOOGLE_API_KEY required" >&2
  exit 1
fi
if [ -z "${DAYTONA_API_KEY:-}" ]; then
  echo "ERROR: DAYTONA_API_KEY required" >&2
  exit 1
fi

RUN_ID="${BENCHFLOW_FEATURE_RUN_ID:-v0.5-gemini-feature-$(date +%Y%m%d-%H%M%S)}"
TASKSET="$(mktemp -d "${TMPDIR:-/tmp}/benchflow-feature-taskset-$RUN_ID.XXXXXX")"
JOBS_ROOT="${BENCHFLOW_FEATURE_JOBS_ROOT:-jobs/codex-feature-rollouts-$RUN_ID}"

echo "Run ID: $RUN_ID"
echo "Taskset: $TASKSET"
echo "Jobs: $JOBS_ROOT"

SKILLS="$(uv run python -c "
from benchflow._utils.benchmark_repos import resolve_source_with_metadata
print(resolve_source_with_metadata('benchflow-ai/skillsbench', path='tasks', ref='main').path)
" 2>/dev/null)"
cp -R "$SKILLS/pddl-tpp-planning" "$TASKSET/pddl-tpp-planning"
cp -R "$SKILLS/azure-bgp-oscillation-route-leak" "$TASKSET/azure-bgp-oscillation-route-leak"
cp -R "$ROOT/tests/conformance/acp_smoke" "$TASKSET/acp_smoke"
cp -R "$ROOT/tests/examples/terminal-bench-smoke-task" "$TASKSET/terminal-bench-smoke-task"

uv run bench eval create \
  --tasks-dir "$TASKSET" \
  --agent "$AGENT" \
  --model "$MODEL" \
  --sandbox "$SANDBOX" \
  --concurrency "$CONCURRENCY" \
  --agent-idle-timeout "$IDLE_TIMEOUT" \
  --jobs-dir "$JOBS_ROOT"

echo ""
echo "══════ Audit ══════"
uv run python tests/integration/check_results.py \
  "$JOBS_ROOT" \
  "agent=$AGENT" \
  "model=$MODEL" \
  "environment=$SANDBOX" \
  "concurrency=$CONCURRENCY" \
  "agent_idle_timeout_sec=$IDLE_TIMEOUT"
