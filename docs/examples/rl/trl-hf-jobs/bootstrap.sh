#!/bin/bash
# Runs inside the Hugging Face Job. Builds two environments with uv, then runs job.py:
#   /opt/bf    torch, transformers, TRL, PEFT, and BenchFlow (training and evaluate.py)
#   /opt/vllm  vLLM, which serves the base and trained models for evaluation
# The two are separate because vLLM and BenchFlow pin different versions of shared
# dependencies. The Job's secrets (HF_TOKEN, DAYTONA_API_KEY) arrive as environment
# variables; nothing here prints them.
set -euo pipefail

BENCHFLOW_SPEC="${BENCHFLOW_SPEC:-benchflow[trl,sandbox-daytona]>=0.8}"
PHASE="all"
prev=""
for arg in "$@"; do
  [ "$prev" = "--phase" ] && PHASE="$arg"
  prev="$arg"
done

export UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache
mkdir -p /work/out
echo "bootstrap: installing the training environment"
uv venv -q -p 3.12 /opt/bf
uv pip install -q -p /opt/bf/bin/python \
  "torch==2.11.0" "transformers==5.6.2" "trl==1.8.0" "peft>=0.17" "accelerate>=1.4" \
  "datasets>=4.7" "matplotlib>=3.8" "huggingface_hub>=1.0" "$BENCHFLOW_SPEC"
if [ "$PHASE" != "train" ]; then
  echo "bootstrap: installing vLLM for evaluation"
  uv venv -q -p 3.12 /opt/vllm
  uv pip install -q -p /opt/vllm/bin/python "vllm==0.23.0"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Run the job in the background so a cancelled Job's SIGTERM reaches it: it then
# stops its rollouts, deletes this run's sandboxes, and uploads what it has.
cd /work
/opt/bf/bin/python -u /inputs/code/job.py --vllm-python /opt/vllm/bin/python "$@" \
  > >(tee -a /work/out/job.log) 2>&1 &
child=$!
trap 'kill -TERM "$child" 2>/dev/null' TERM INT
status=0
wait "$child" || status=$?
if kill -0 "$child" 2>/dev/null; then  # the trap interrupted the first wait
  wait "$child" || status=$?
fi
exit "$status"
