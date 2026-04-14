#!/usr/bin/env bash
# Populate .corpora/ with the three benchmark task trees needed by run_matrix.py.
#
# Sources (all public):
#   * skillsbench (77 tasks)        — laude-institute/harbor-datasets
#   * swebench-verified (500 tasks) — laude-institute/harbor-datasets (sparse-checkout)
#   * terminal-bench-2 (~80 tasks)  — harbor-framework/terminal-bench-2
#
# Re-runnable: skips what's already present. Pass --force to re-clone.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
CORPORA="${HERE}/.corpora"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

mkdir -p "${CORPORA}"
cd "${CORPORA}"

# harbor-datasets (sparse — only the two datasets we need)
HD_DIR="${CORPORA}/harbor-datasets"
if [ "${FORCE}" -eq 1 ] && [ -d "${HD_DIR}" ]; then
    rm -rf "${HD_DIR}"
fi

if [ ! -d "${HD_DIR}" ]; then
    echo "[fetch] cloning harbor-datasets (sparse: skillsbench + swebench-verified)"
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/laude-institute/harbor-datasets.git
    (
        cd harbor-datasets
        git sparse-checkout set datasets/skillsbench datasets/swebench-verified
    )
else
    echo "[fetch] harbor-datasets already present, skipping"
fi

# terminal-bench-2 (full clone — repo is small)
TB_DIR="${CORPORA}/terminal-bench-2"
if [ "${FORCE}" -eq 1 ] && [ -d "${TB_DIR}" ]; then
    rm -rf "${TB_DIR}"
fi

if [ ! -d "${TB_DIR}" ]; then
    echo "[fetch] cloning terminal-bench-2 (full)"
    git clone --depth 1 https://github.com/harbor-framework/terminal-bench-2.git
else
    echo "[fetch] terminal-bench-2 already present, skipping"
fi

echo
echo "[done] corpora staged at ${CORPORA}"
du -sh "${HD_DIR}/datasets"/* "${TB_DIR}" 2>&1 | sed 's/^/  /'
