#!/usr/bin/env bash
set -euo pipefail
cd /repo

missing=""
for path in \
  src/benchflow/task/runtime_view.py \
  src/benchflow/task/package.py \
  src/benchflow/task/runtime_capabilities.py \
  src/benchflow/task/acceptance_live.py \
  tests/test_task_package.py \
  tests/test_runtime_capabilities.py \
  tests/test_tasks.py; do
  if [[ ! -f "$path" ]]; then
    missing="${missing}${path} "
  fi
done

if [[ -n "$missing" ]]; then
  printf 'missing reference implementation files: %s\n' "$missing" >&2
  exit 1
fi
