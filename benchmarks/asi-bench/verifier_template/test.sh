#!/bin/bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HOME="$(mktemp -d)"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

exec /opt/asi-eval/bin/python /verifier/score_entry.py
