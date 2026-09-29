#!/usr/bin/env bash
# The bench hillclimb demo: climb an office-files skill on SkillsBench's
# office and spreadsheet tasks, with a held-out test split. See README.md.
#
#   ./run.sh smoke    # 4 tasks, 2 trials, one forced round, $10 cap: plumbing only
#   ./run.sh climb    # the demo run
#
# Knobs (environment variables): WORK, SANDBOX, CONCURRENCY, AGENT_MODEL,
# PROPOSER_MODEL, TRIALS, MIN_GAIN, TEST_FRAC, SEED, ROUNDS, MAX_COST_USD,
# SKILLSBENCH_SHA.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE="${1:-climb}"
WORK="${WORK:-$HOME/hillclimb-demo}"
SANDBOX="${SANDBOX:-daytona}"
CONCURRENCY="${CONCURRENCY:-40}"
AGENT_MODEL="${AGENT_MODEL:-claude-haiku-4-5}"
PROPOSER_MODEL="${PROPOSER_MODEL:-claude-opus-4-8}"
TRIALS="${TRIALS:-5}"
MIN_GAIN="${MIN_GAIN:-0.15}"
TEST_FRAC="${TEST_FRAC:-0.4}"
SEED="${SEED:-7}"
ROUNDS="${ROUNDS:-5}"
MAX_COST_USD="${MAX_COST_USD:-250}"

# Both agents read ANTHROPIC_API_KEY from the environment (or .env), as
# bench eval run does, and reach the provider only through BenchFlow's model
# proxy. Keys are never put on the command line.
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}"
if [ "$SANDBOX" = daytona ]; then
  : "${DAYTONA_API_KEY:?set DAYTONA_API_KEY, or run with SANDBOX=docker on a large host}"
fi

# SkillsBench at the first commit whose task.md files use this BenchFlow's
# `sandbox:` key (tag v1.1 still says `environment:` and targets BenchFlow
# <0.7). Same 87-task roster.
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
    extra=(--trials 2 --rounds 1 --force --min-gain 0.5 --max-cost-usd 10)
    out="$WORK/runs/smoke-$(date +%Y%m%d-%H%M%S)"
    ;;
  climb)
    tasks=()
    while IFS= read -r task; do
      [ -n "$task" ] && tasks+=("$task")
    done < "$HERE/tasks.txt"
    extra=(
      --trials "$TRIALS" --min-gain "$MIN_GAIN" --rounds "$ROUNDS"
      --max-cost-usd "$MAX_COST_USD" --exclude-broken-tasks
    )
    out="$WORK/runs/climb-$(date +%Y%m%d-%H%M%S)"
    ;;
  *)
    echo "usage: $0 smoke|climb" >&2
    exit 2
    ;;
esac

include=()
for task in "${tasks[@]}"; do
  include+=(--include "$task")
done

echo "hillclimb $STAGE: ${#tasks[@]} tasks, agent $AGENT_MODEL, optimizer $PROPOSER_MODEL, sandbox $SANDBOX -> $out"
bench hillclimb \
  --tasks-dir "$WORK/skillsbench/tasks" "${include[@]}" \
  --surface "$HERE/office-skills" \
  --out "$out" \
  --agent claude-agent-acp --model "$AGENT_MODEL" \
  --proposer-agent claude-agent-acp --proposer-model "$PROPOSER_MODEL" \
  --sandbox "$SANDBOX" --concurrency "$CONCURRENCY" \
  --test-frac "$TEST_FRAC" --seed "$SEED" \
  "${extra[@]}"
