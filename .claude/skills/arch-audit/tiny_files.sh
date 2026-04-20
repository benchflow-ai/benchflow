#!/usr/bin/env bash
# Spot-check: list non-test src/benchflow/**/*.py files with non-blank LOC < THRESHOLD (default 100).
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

THRESHOLD="${1:-100}"

while IFS= read -r f; do
  loc=$(awk 'NF > 0' "$f" | wc -l | tr -d ' ')
  if [ "${loc:-0}" -lt "$THRESHOLD" ]; then
    printf "%d\t%s\n" "$loc" "$f"
  fi
done < <(find src/benchflow -name '*.py' -not -name 'conftest.py' -not -path '*/tests/*' | sort)
