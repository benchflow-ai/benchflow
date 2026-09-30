#!/usr/bin/env bash
# The Fireworks cookbook, one step per command. Run from a BenchFlow checkout:
#
#   docs/examples/rl/fireworks/run.sh agent-eval  # BenchFlow agents on a Fireworks model
#   docs/examples/rl/fireworks/run.sh screen      # which train tasks give GRPO a signal
#   docs/examples/rl/fireworks/run.sh train       # serverless RL; promotes the result
#   docs/examples/rl/fireworks/run.sh deploy      # base and tuned on dedicated deployments
#   docs/examples/rl/fireworks/run.sh evaluate    # evaluate.py on both, held-out test split
#   docs/examples/rl/fireworks/run.sh cleanup     # delete both deployments; show their cost
#
# Needs FIREWORKS_API_KEY, FIREWORKS_ACCOUNT_ID and DAYTONA_API_KEY in the
# environment, loaded from mode-600 files; see README.md.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
HERE=docs/examples/rl/fireworks

TASKS=${TASKS:?set TASKS to the task family folder (with train/ and test/)}
OUT=${OUT:-runs/fireworks}
SANDBOX=${SANDBOX:-daytona}
CONCURRENCY=${CONCURRENCY:-16}
BASE_MODEL=${BASE_MODEL:-accounts/fireworks/models/qwen3p8-27b}
MODEL_ID=${MODEL_ID:-bf-rl-qwen38}          # the promoted LoRA, private to the account
SHAPE=${SHAPE:-default}                      # fireworks_deploy.py shapes MODEL lists them
SCREEN_TASKS=${SCREEN_TASKS:-48}
GROUP=${GROUP:-4}
STEPS=${STEPS:-10}
GROUPS_PER_STEP=${GROUPS_PER_STEP:-8}
LR=${LR:-4e-5}
MAX_USD=${MAX_USD:-60}
MAX_TURNS=${MAX_TURNS:-10}                   # evaluate.py's default; training and eval must match
SAMPLES=${SAMPLES:-2}
AGENT_MODEL=${AGENT_MODEL:-accounts/fireworks/models/glm-5p3}
AGENT_TASKS=${AGENT_TASKS:-sql-900000 log-900001 bugfix-900003}

: "${FIREWORKS_API_KEY:?load it from a mode-600 file first; see README.md}"
RL=(uv run --with "fireworks-ai[training-sdk]==1.2.17" --with "transformers==5.6.2" python "$HERE/fireworks_rl.py")
DEPLOY=(uv run --with "fireworks-ai==1.2.17" python "$HERE/fireworks_deploy.py")

case "${1:-}" in
agent-eval)
  # The model proxy runs beside the sandbox: on Docker that is this machine, so
  # the key stays here. (On Daytona the proxy, and so the key, runs in the sandbox.)
  include=()
  for task in $AGENT_TASKS; do include+=(--include "$task"); done
  uv run bench eval run --tasks-dir "$TASKS/test" "${include[@]}" \
    --agent opencode --model "fireworks/$AGENT_MODEL" --sandbox docker \
    --concurrency 3 --usage-tracking required --jobs-dir "$OUT/agent-eval"
  ;;
screen)
  "${RL[@]}" screen --tasks-dir "$TASKS/train" --limit "$SCREEN_TASKS" --group-size "$GROUP" \
    --max-turns "$MAX_TURNS" --sandbox "$SANDBOX" --concurrency "$CONCURRENCY" --out "$OUT/screen"
  ;;
train)
  "${RL[@]}" train --tasks-dir "$TASKS/train" --tasks-file "$OUT/screen/learnable.txt" \
    --steps "$STEPS" --groups-per-step "$GROUPS_PER_STEP" --group-size "$GROUP" \
    --learning-rate "$LR" --max-usd "$MAX_USD" --max-turns "$MAX_TURNS" \
    --sandbox "$SANDBOX" --concurrency "$CONCURRENCY" \
    --output-model-id "$MODEL_ID" --out "$OUT/train"
  ;;
deploy)
  : "${FIREWORKS_ACCOUNT_ID:?}"
  "${DEPLOY[@]}" create bf-rl-base "$BASE_MODEL" --shape "$SHAPE"
  "${DEPLOY[@]}" create bf-rl-tuned "accounts/$FIREWORKS_ACCOUNT_ID/models/$MODEL_ID" --shape "$SHAPE"
  for d in bf-rl-base bf-rl-tuned; do "${DEPLOY[@]}" wait "$d"; done
  "${DEPLOY[@]}" smoke "$BASE_MODEL#accounts/$FIREWORKS_ACCOUNT_ID/deployments/bf-rl-base"
  ;;
evaluate)
  : "${FIREWORKS_ACCOUNT_ID:?}"
  base="$BASE_MODEL#accounts/$FIREWORKS_ACCOUNT_ID/deployments/bf-rl-base"
  tuned="accounts/$FIREWORKS_ACCOUNT_ID/models/$MODEL_ID#accounts/$FIREWORKS_ACCOUNT_ID/deployments/bf-rl-tuned"
  for pair in "base|$base" "tuned|$tuned"; do
    uv run python docs/examples/rl/common/evaluate.py --tasks-dir "$TASKS/test" \
      --base-url https://api.fireworks.ai/inference/v1 --model "${pair#*|}" \
      --api-key-env FIREWORKS_API_KEY --max-turns "$MAX_TURNS" \
      --sandbox "$SANDBOX" --concurrency $((CONCURRENCY / 2)) --samples "$SAMPLES" \
      --out "$OUT/eval-${pair%%|*}" &
  done
  wait
  ;;
cleanup)
  : "${FIREWORKS_ACCOUNT_ID:?}"
  today=$(date -u +%Y-%m-%d)
  for d in bf-rl-base bf-rl-tuned; do
    "${DEPLOY[@]}" delete "$d" || true
    "${DEPLOY[@]}" usage "$d" --since "$today" --usd-per-hour "${USD_PER_HOUR:-8}"
  done
  ;;
*)
  sed -n '2,13p' "$0"
  exit 2
  ;;
esac
