#!/bin/bash
# Materialise the 50-task parity subset at /tmp/parity-dev-0_3-skillsbench-50/
# from the 173-task corpus at /tmp/parity-full-tasks/ (already populated by
# benchflow/task_download.py via ensure_tasks("skillsbench")).
#
# Idempotent: re-running fixes missing symlinks, leaves existing ones alone.
set -euo pipefail

SRC="${PARITY_FULL_TASKS:-/tmp/parity-full-tasks}"
DST="${PARITY_SUBSET_DIR:-/tmp/parity-dev-0_3-skillsbench-50}"
LIST="$(dirname "$0")/parity-skillsbench-50.txt"

if [ ! -d "$SRC" ]; then
    echo "error: corpus dir $SRC not found; run task_download first" >&2
    exit 1
fi

mkdir -p "$DST"
while IFS= read -r name; do
    [ -z "$name" ] && continue
    if [ ! -e "$SRC/$name" ]; then
        echo "warn: $name not in corpus" >&2
        continue
    fi
    ln -sfn "$SRC/$name" "$DST/$name"
done < "$LIST"

count=$(ls "$DST" | wc -l)
echo "subset ready: $count tasks at $DST"
