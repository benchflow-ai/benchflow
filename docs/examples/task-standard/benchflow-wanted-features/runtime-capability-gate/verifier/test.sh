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

score_runtime_view=0
score_runtime_package=0
score_fail_closed=0
score_acceptance_live=0
score_compatibility=0

if contains "src/benchflow/task/runtime_view.py" "class TaskRuntimeView"; then
  score_runtime_view=1
fi

if contains "src/benchflow/task/package.py" "class TaskPackage" "runtime_issues" \
  && [[ -f "tests/test_task_package.py" ]]; then
  score_runtime_package=1
fi

if contains \
    "src/benchflow/task/runtime_capabilities.py" \
    "validate_task_runtime_support" \
    "UnsupportedTaskFeature" \
  && [[ -f "tests/test_runtime_capabilities.py" ]]; then
  score_fail_closed=1
fi

if contains \
    "src/benchflow/task/acceptance_live.py" \
    "acceptance-live-report" \
    "RolloutConfig" \
    "pre_agent_hooks" \
    "staged_tree_sha256" \
    "calibration-report" \
    "leaderboard_suitability" \
  && contains \
    "tests/test_tasks.py" \
    "test_acceptance_live_persists_report_and_hash_sidecar" \
    "test_acceptance_live_without_report_does_not_write_artifact" \
    "test_acceptance_live_oracle_case_runs_oracle_before_verify" \
    "test_acceptance_live_generates_cases_from_calibration_report" \
    "test_acceptance_live_leaderboard_suitability_accepts_complete_report"; then
  score_acceptance_live=1
fi

if contains "src/benchflow/task/export.py" "export_task_to_split_layout" \
  && contains "src/benchflow/task/imports.py" "compat" \
  && [[ -f "tests/test_task_export.py" ]]; then
  score_compatibility=1
fi

points=$((score_runtime_view + score_runtime_package + score_fail_closed + score_acceptance_live + score_compatibility))
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
  "runtime_view": $score_runtime_view,
  "runtime_package": $score_runtime_package,
  "fail_closed": $score_fail_closed,
  "acceptance_live": $score_acceptance_live,
  "compatibility": $score_compatibility
}
JSON
cp /logs/verifier/reward.json /logs/verifier/reward-details.json

if [[ "$reward" != "1.0" ]]; then
  printf '{"reward":%s,"runtime_view":%s,"runtime_package":%s,"fail_closed":%s,"acceptance_live":%s,"compatibility":%s}\n' \
    "$reward" \
    "$score_runtime_view" \
    "$score_runtime_package" \
    "$score_fail_closed" \
    "$score_acceptance_live" \
    "$score_compatibility"
fi
