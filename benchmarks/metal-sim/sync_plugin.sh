#!/bin/bash
# Copy the simulator plugin and shared files into every task's build context.
# Docker builds only see the task's environment/ directory, so the plugin is vendored per task.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# plugin/ is the source of truth for the vendored copies in tasks/*/environment/metal-sim
for task in "$here"/tasks/*/; do
  env_dir="$task/environment"
  mkdir -p "$env_dir" "$task/verifier" "$task/oracle"
  rm -rf "$env_dir/metal-sim"
  rsync -a --exclude tests --exclude '*.egg-info' --exclude __pycache__ --exclude .pytest_cache \
    "$here/plugin/" "$env_dir/metal-sim/"
  cp "$here/shared/Dockerfile" "$env_dir/Dockerfile"
  cp "$here/shared/policy_template.py" "$env_dir/policy.py"
  cp "$here/shared/reward.py" "$task/verifier/reward.py"
  cp "$here/shared/oracle_policy.py" "$task/oracle/policy.py"
  echo "synced $(basename "$task")"
done
