#!/usr/bin/env bash
# Rule 3 helper. TSV: ratio<tab>commits<tab>loc<tab>path.
# --follow tracks single-rename chains only; squash-merge repos under-count.
set -euo pipefail

if [ $# -eq 0 ]; then
  echo "usage: churn_ratio.sh <file>..." >&2
  exit 2
fi

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

for f in "$@"; do
  if [ ! -f "$f" ]; then
    echo -e "NA\t0\t0\t$f" >&2
    continue
  fi
  commits=$(git log --since=6.months --follow --format=%H -- "$f" 2>/dev/null | wc -l | tr -d ' ')
  loc=$(awk 'NF > 0' "$f" | wc -l | tr -d ' ')
  loc=${loc:-0}
  if [ "$loc" -eq 0 ]; then
    ratio="inf"
  else
    ratio=$(awk "BEGIN { printf \"%.4f\", $commits / $loc }")
  fi
  printf "%s\t%s\t%s\t%s\n" "$ratio" "$commits" "$loc" "$f"
done
