#!/usr/bin/env bash
# Full 663-cell reward-hack-matrix sweep against benchflow 0.2.2.
#
# Reuses the 0.2.0 results from reference_sweep_0.2.0.json (663 cells, still
# valid — 0.2.0 package, exploit scripts, and corpus haven't changed). Only
# runs the 0.2.2 side, which is what's actually new after the Tier 1-4
# sandbox hardening. Expected wall time: ~20 min on Daytona at concurrency
# 64, with each trial capped at 900s via _worker.py's asyncio.wait_for.
#
# Prerequisites:
#   - DAYTONA_API_KEY in env (sourced from /workspace/.env in this repo)
#   - Corpora present at labs/reward-hack-matrix/.corpora/ (run ./fetch_corpora.sh)
#   - Venvs present at .venvs/bf-0.2.0 and .venvs/bf-0.2.2 (run_matrix.py auto-creates)
#
# Usage (from labs/reward-hack-matrix/):
#   bash scripts/full_sweep_0.2.2.sh
#
# Override concurrency or limit for testing:
#   CONCURRENCY=16 bash scripts/full_sweep_0.2.2.sh
#   LIMIT=5 bash scripts/full_sweep_0.2.2.sh  # 5 tasks per benchmark

set -euo pipefail

# Find labs/reward-hack-matrix directory (script may be run from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$LAB_DIR"

CONCURRENCY="${CONCURRENCY:-64}"
LIMIT="${LIMIT:-}"
SUMMARY_PATH="${SUMMARY_PATH:-.jobs/matrix_sweep_0.2.2.json}"
REFERENCE="${REFERENCE:-reference_sweep_0.2.0.json}"

# Seed the summary from the 0.2.0 reference so --resume skips 0.2.0 trials.
# This must happen BEFORE run_matrix.py starts, otherwise the orchestrator
# rebuilds the summary from scratch.
mkdir -p "$(dirname "$SUMMARY_PATH")"
if [ ! -f "$SUMMARY_PATH" ] && [ -f "$REFERENCE" ]; then
    echo "[seed] copying $REFERENCE -> $SUMMARY_PATH (skip 0.2.0 re-runs)"
    cp "$REFERENCE" "$SUMMARY_PATH"
elif [ -f "$SUMMARY_PATH" ]; then
    echo "[seed] $SUMMARY_PATH already exists — resume from it"
fi

# Ensure daytona key is in env
if [ -z "${DAYTONA_API_KEY:-}" ] && [ -f /workspace/.env ]; then
    set -a; . /workspace/.env; set +a
fi
if [ -z "${DAYTONA_API_KEY:-}" ]; then
    echo "ERROR: DAYTONA_API_KEY not set" >&2
    exit 1
fi

ARGS=(--sweep --concurrency "$CONCURRENCY" --resume --summary-path "$SUMMARY_PATH")
if [ -n "$LIMIT" ]; then
    ARGS+=(--limit "$LIMIT")
fi

echo "[launch] python3 run_matrix.py ${ARGS[*]}"
python3 -u run_matrix.py "${ARGS[@]}"
