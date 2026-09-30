#!/usr/bin/env bash
# The hill-climb demo on SkillsBench's office and spreadsheet tasks. See README.md.
#
#   ./run.sh smoke    # 4 tasks, 2 trials, one forced round, small caps: plumbing only
#   ./run.sh climb    # the demo run
#
# Knobs (environment variables): WORK, SANDBOX, CONCURRENCY, AGENT_MODEL,
# PROPOSER_MODEL, TRIALS, MIN_GAIN, TEST_FRAC, SEED, ROUNDS, MAX_COST_USD,
# MAX_ROLLOUTS, MAX_SANDBOX_HOURS, SKILLSBENCH_SHA.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
STAGE="${1:-climb}"
WORK="${WORK:-$HOME/hillclimb-demo}"
SANDBOX="${SANDBOX:-daytona}"
CONCURRENCY="${CONCURRENCY:-40}"
AGENT_MODEL="${AGENT_MODEL:-claude-haiku-4-5-20251001}"
PROPOSER_MODEL="${PROPOSER_MODEL:-claude-opus-5-5}"
TRIALS="${TRIALS:-5}"
MIN_GAIN="${MIN_GAIN:-0.15}"
TEST_FRAC="${TEST_FRAC:-0.4}"
SEED="${SEED:-7}"
ROUNDS="${ROUNDS:-5}"
MAX_COST_USD="${MAX_COST_USD:-250}"
# Caps that bind when no USD is known (a subscription reports none to
# BenchFlow): rollouts that call a model, and sandbox wall-clock hours.
MAX_ROLLOUTS="${MAX_ROLLOUTS:-610}"
MAX_SANDBOX_HOURS="${MAX_SANDBOX_HOURS:-120}"

# Both agents are Claude Code (claude-agent-acp). Credentials come from the
# environment, never the command line: a Claude subscription token from
# `claude setup-token` (the demo then prices each rollout from Claude Code's
# own session log), or an API key (priced by BenchFlow's model proxy).
if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "set CLAUDE_CODE_OAUTH_TOKEN (claude setup-token) or ANTHROPIC_API_KEY" >&2
  exit 2
fi
if [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  echo "note: both CLAUDE_CODE_OAUTH_TOKEN and ANTHROPIC_API_KEY are set; Claude Code uses the API key" >&2
fi
if [ "$SANDBOX" = daytona ]; then
  : "${DAYTONA_API_KEY:?set DAYTONA_API_KEY, or run with SANDBOX=docker on a large host}"
fi

# SkillsBench at the first commit whose task.md files use this BenchFlow's
# sandbox: key (tag v1.1 still says environment: and fails to parse here).
SKILLSBENCH_SHA="${SKILLSBENCH_SHA:-9a1f4dd5f7659f75707435da3ce854b6e48321d1}"
if [ ! -d "$WORK/skillsbench/tasks" ]; then
  git init -q "$WORK/skillsbench"
  git -C "$WORK/skillsbench" fetch -q --depth 1 \
    https://github.com/benchflow-ai/skillsbench "$SKILLSBENCH_SHA"
  git -C "$WORK/skillsbench" checkout -q FETCH_HEAD
fi

case "$STAGE" in
  smoke)
    tasks=(offer-letter-generator court-form-filling weighted-gdp-calc reserves-at-risk-calc)
    extra=(--trials 2 --rounds 1 --force --min-gain 0.5 --max-cost-usd 10
      --max-rollouts 20 --max-sandbox-seconds 21600)
    ;;
  climb)
    tasks=()
    while IFS= read -r task; do
      [ -n "$task" ] && tasks+=("$task")
    done < "$HERE/tasks.txt"
    extra=(--trials "$TRIALS" --min-gain "$MIN_GAIN" --rounds "$ROUNDS" --max-cost-usd "$MAX_COST_USD"
      --max-rollouts "$MAX_ROLLOUTS" --max-sandbox-seconds "$((MAX_SANDBOX_HOURS * 3600))")
    ;;
  *)
    echo "usage: $0 smoke|climb" >&2
    exit 2
    ;;
esac
out="$WORK/runs/$STAGE-$(date +%Y%m%d-%H%M%S)"

include=()
for task in "${tasks[@]}"; do
  include+=(--include "$task")
done

echo "hillclimb $STAGE: ${#tasks[@]} tasks, agent $AGENT_MODEL, optimizer $PROPOSER_MODEL, sandbox $SANDBOX -> $out"
cd "$REPO"
uv run python "$HERE/hillclimb.py" \
  --tasks-dir "$WORK/skillsbench/tasks" "${include[@]}" \
  --skills "$HERE/office-skills" \
  --out "$out" \
  --agent claude-agent-acp --model "$AGENT_MODEL" \
  --proposer-agent claude-agent-acp --proposer-model "$PROPOSER_MODEL" \
  --sandbox "$SANDBOX" --concurrency "$CONCURRENCY" \
  --test-frac "$TEST_FRAC" --seed "$SEED" \
  "${extra[@]}"
