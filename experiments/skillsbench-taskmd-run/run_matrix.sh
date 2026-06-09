#!/usr/bin/env bash
# Run adapted SkillsBench task.md packages on openhands + DeepSeek v4 across
# skill modes (no-skill / with-skill / self-gen). Smoke first, then full.
#
# Could NOT run in the Claude-on-web container: its network policy returns
# 403 "Host not in allowlist" for api.deepseek.com and app.daytona.io, and no
# docker daemon is available. Run this locally where those hosts are reachable.
#
#   ./run_matrix.sh smoke   # one task, with-skill, 1 concurrency — verify a REAL rollout
#   ./run_matrix.sh full    # all tasks x {no-skill, with-skill, self-gen}
#
# Landmines baked in (see docs/reports/2026-06-08-task-standard-handoff-runbook.md §5):
#  - FRESH --jobs-dir per batch (eval create RESUMES a matching dir, reusing stale results)
#  - DEEPSEEK_BASE_URL is NOT defaulted by the provider — set it
#  - a dead model key surfaces only as an opaque agent error — verify it first
#  - trust a rollout only when n_tool_calls>0, total_tokens>0, reward non-None
#  - use openhands (clean litellm wiring), NOT opencode
#  - Daytona caps each sandbox at 10 GB — keep tasks light
set -euo pipefail
cd "$(dirname "$0")"

: "${DAYTONA_API_KEY:?set DAYTONA_API_KEY}"
: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"

AGENT="${AGENT:-openhands}"
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"   # validated id; override via MODEL=... (confirm via DeepSeek /models)
SANDBOX="${SANDBOX:-daytona}"
TASKS_DIR="${TASKS_DIR:-./adapted}"
CONC="${CONC:-8}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
FIRST_TASK="$(grep -vE '^\s*(#|$)' simple_tasks.txt | head -1)"

echo "verifying DeepSeek key is live (a dead key => opaque agent error later)..."
_models="$(curl -sf "$DEEPSEEK_BASE_URL/models" -H "Authorization: Bearer $DEEPSEEK_API_KEY")" || { echo "  DeepSeek key/endpoint NOT reachable — fix before running"; exit 1; }
if printf %s "$_models" | grep -q "\"${MODEL#deepseek/}\""; then
  echo "  DeepSeek key OK and model $MODEL present"
else
  echo "  model $MODEL not in DeepSeek /models — fix MODEL before running"; exit 1
fi

run () {  # run <skill-mode> <suffix> [extra flags...]
  local mode="$1" suffix="$2"; shift 2
  local jobs="jobs/sb-${suffix}-${TS}"   # fresh per batch
  echo ">>> skill-mode=$mode  agent=$AGENT  model=$MODEL  sandbox=$SANDBOX -> $jobs"
  uv run bench eval create \
    --tasks-dir "$TASKS_DIR" \
    --agent "$AGENT" --model "$MODEL" --sandbox "$SANDBOX" \
    --skill-mode "$mode" --concurrency "$CONC" \
    --jobs-dir "$jobs" "$@"
}

case "${1:-smoke}" in
  smoke)
    run with-skill smoke --include "$FIRST_TASK" --concurrency 1
    echo
    echo "SMOKE done. Inspect jobs/sb-smoke-${TS}/*/result.json —"
    echo "trust it ONLY if n_tool_calls>0, total_tokens>0, and reward is non-None."
    ;;
  full)
    run no-skill   full-noskill
    run with-skill full-withskill
    run self-gen   full-selfgen --self-gen-no-internet
    echo
    echo "FULL done. Compare per-task mean reward across the three modes."
    ;;
  *)
    echo "usage: $0 [smoke|full]"; exit 1 ;;
esac
