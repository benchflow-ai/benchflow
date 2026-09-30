#!/usr/bin/env bash
# The Baseten cookbook, one step per command. Run from a BenchFlow checkout:
#
#   docs/examples/rl/baseten/run.sh agent-eval   # BenchFlow agents on Baseten Model APIs
#   docs/examples/rl/baseten/run.sh deploy       # serve a base model and a LoRA (Truss)
#   docs/examples/rl/baseten/run.sh evaluate     # evaluate.py on both, same deployment
#   docs/examples/rl/baseten/run.sh cleanup      # deactivate the deployment
#
# Needs BASETEN_API_KEY (and DAYTONA_API_KEY for SANDBOX=daytona) in the
# environment, loaded from mode-600 files; see README.md. After `deploy`, set
# MODEL_ID and DEPLOYMENT_ID from its output.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
HERE=docs/examples/rl/baseten

TASKS=${TASKS:?set TASKS to the task family folder (with train/ and test/)}
OUT=${OUT:-runs/baseten}
SANDBOX=${SANDBOX:-daytona}
CONCURRENCY=${CONCURRENCY:-8}
SAMPLES=${SAMPLES:-2}
AGENT=${AGENT:-opencode}
AGENT_TASKS=${AGENT_TASKS:-sql-900000 log-900001 bugfix-900003}
MODELS=${MODELS:-zai-org/GLM-5.3 moonshotai/Kimi-K3}
BASE_NAME=${BASE_NAME:-Qwen/Qwen3.5-9B}
LORA_NAME=${LORA_NAME:-benchflow-sft}

: "${BASETEN_API_KEY:?load it from a mode-600 file first; see README.md}"

case "${1:-}" in
agent-eval)
  # The model proxy runs beside the sandbox: on Docker that is this machine, so
  # the key stays here. (On Daytona the proxy, and so the key, runs in the sandbox.)
  include=()
  for task in $AGENT_TASKS; do include+=(--include "$task"); done
  for model in $MODELS; do
    uv run bench eval run --tasks-dir "$TASKS/test" "${include[@]}" \
      --agent "$AGENT" --model "baseten/$model" --sandbox docker \
      --concurrency 3 --usage-tracking required \
      --jobs-dir "$OUT/agent-eval/${model//\//-}"
  done
  ;;
deploy)
  uv run --with truss==0.18.32 python "$HERE/baseten_deploy.py" push "$HERE/truss/qwen35-9b-lora"
  echo "Now: export MODEL_ID=... DEPLOYMENT_ID=... from the output above, then:"
  echo "  uv run python $HERE/baseten_deploy.py wait \$MODEL_ID \$DEPLOYMENT_ID"
  ;;
evaluate)
  : "${MODEL_ID:?} ${DEPLOYMENT_ID:?}"
  uv run python "$HERE/baseten_deploy.py" wait "$MODEL_ID" "$DEPLOYMENT_ID"
  url=$(uv run python "$HERE/baseten_deploy.py" url "$MODEL_ID" "$DEPLOYMENT_ID")
  for pair in "base:$BASE_NAME" "lora:$LORA_NAME"; do
    uv run python docs/examples/rl/common/evaluate.py --tasks-dir "$TASKS/test" \
      --base-url "$url" --model "${pair#*:}" --api-key-env BASETEN_API_KEY \
      --sandbox "$SANDBOX" --concurrency "$CONCURRENCY" --samples "$SAMPLES" \
      --out "$OUT/eval-${pair%%:*}" &
  done
  wait
  ;;
cleanup)
  : "${MODEL_ID:?} ${DEPLOYMENT_ID:?}"
  uv run python "$HERE/baseten_deploy.py" deactivate "$MODEL_ID" "$DEPLOYMENT_ID"
  uv run python "$HERE/baseten_deploy.py" status "$MODEL_ID" "$DEPLOYMENT_ID"
  ;;
*)
  sed -n '2,12p' "$0"
  exit 2
  ;;
esac
