#!/usr/bin/env bash
set -euo pipefail

# Source env vars
source /workspace/scripts/agent-env.sh

export ABLATION_MODEL="gemini-3.1-flash-lite-preview"
export ABLATION_AGENT="gemini"
export ABLATION_CONCURRENCY="16"

cd /workspace/repos/benchflow
exec uv run python experiments/ablation_retry.py 2>&1 | tee /tmp/ablation-retry-16.log
