#!/usr/bin/env bash
set -euo pipefail
cd /repo

contains() {
  local file="$1"
  shift
  [[ -f "$file" ]] || return 1

  local needle
  for needle in "$@"; do
    grep -Fq "$needle" "$file" || return 1
  done
}

score_document=0
score_strategy_execution=0
score_reward_kit=0
score_agent_judge=0
score_reward_artifacts=0

if contains "src/benchflow/task/verifier_document.py" \
    "class VerifierDocument" \
    "class VerifierStrategy" \
    "selected_strategy" \
    "aggregate_policy" \
  && contains "tests/test_verifier_document.py" \
    "verifier-package dogfood task" \
    "reward-details.json"; then
  score_document=1
fi

if contains "src/benchflow/task/verifier.py" \
    "if strategy.type == \"llm-judge\"" \
    "if strategy.type == \"reward-kit\"" \
    "if strategy.type == \"agent-judge\"" \
  && contains "src/benchflow/task/runtime_capabilities.py" \
    "select script, llm-judge, reward-kit, agent-judge" \
  && contains "tests/test_llm_judge_verifier.py" \
    "verifier.md llm-judge can own rubric, model, input, and context"; then
  score_strategy_execution=1
fi

if contains "src/benchflow/task/verifier.py" \
    "_reward_kit_criteria_policy" \
    "reward-kit strategy with declared criteria must write" \
  && contains "tests/test_llm_judge_verifier.py" \
    "test_dogfood_reward_kit_runner_uses_declared_criteria_policy" \
    "Declared Reward Kit criteria recompute"; then
  score_reward_kit=1
fi

if contains "src/benchflow/task/verifier.py" \
    "agent-judge strategy" \
    "Run a verifier-scoped judge role" \
  && contains "tests/test_llm_judge_verifier.py" \
    "An agent-judge strategy runs in verifier scope"; then
  score_agent_judge=1
fi

if contains "src/benchflow/task/verifier.py" \
    "reward_details_json_path" \
    "apply_aggregate_policy" \
  && contains "tests/test_llm_judge_verifier.py" \
    "verifier package reward-details artifacts" \
    "verifier.outputs.aggregate_policy turns metrics into reward"; then
  score_reward_artifacts=1
fi

points=$((score_document + score_strategy_execution + score_reward_kit + score_agent_judge + score_reward_artifacts))
case "$points" in
  5) reward="1.0" ;;
  4) reward="0.8" ;;
  3) reward="0.6" ;;
  2) reward="0.4" ;;
  1) reward="0.2" ;;
  *) reward="0.0" ;;
esac

mkdir -p /logs/verifier
printf '%s' "$reward" > /logs/verifier/reward.txt
cat > /logs/verifier/reward.json <<JSON
{
  "reward": $reward,
  "items": {
    "verifier_document": $score_document,
    "strategy_execution": $score_strategy_execution,
    "reward_kit": $score_reward_kit,
    "agent_judge": $score_agent_judge,
    "reward_artifacts": $score_reward_artifacts
  },
  "metadata": {"aggregate_policy": "reward"}
}
JSON
cat > /logs/verifier/reward-details.json <<JSON
{
  "criteria": [
    {"id": "verifier_document", "score": $score_document},
    {"id": "strategy_execution", "score": $score_strategy_execution},
    {"id": "reward_kit", "score": $score_reward_kit},
    {"id": "agent_judge", "score": $score_agent_judge},
    {"id": "reward_artifacts", "score": $score_reward_artifacts}
  ]
}
JSON

if [[ "$reward" != "1.0" ]]; then
  cat /logs/verifier/reward.json
fi
