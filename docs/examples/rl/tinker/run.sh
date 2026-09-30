#!/usr/bin/env bash
# RL on BenchFlow tasks with Tinker: LoRA training with tinker-cookbook's RL
# loop on the shared RL task family, and held-out evaluation before and after.
# See README.md.
#
#   ./run.sh smoke      # the plumbing: hello-world, 2 training steps (cents, ~3 min)
#   ./run.sh baseline   # the base model on a stratified slice of train: pick a model
#   ./run.sh train      # held-out baseline, training, held-out eval of the final checkpoint
#
# Knobs (environment variables): WORK, FAMILY, MODEL, RENDERER, STEPS, GROUP_SIZE,
# GROUPS_PER_BATCH, LEARNING_RATE, LORA_RANK, MAX_TOKENS, MAX_SANDBOXES, SANDBOX,
# SAMPLES, EVAL_TEMPERATURE, EVAL_TOP_P, TINKER_OAI_URL.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
STAGE="${1:-train}"
WORK="${WORK:-$HOME/tinker-rl}"
FAMILY="${FAMILY:-$WORK/tasks}"  # a task family folder with train/ and test/
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
RENDERER="${RENDERER:-}"
STEPS="${STEPS:-20}"
GROUP_SIZE="${GROUP_SIZE:-8}"
GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-8}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
LORA_RANK="${LORA_RANK:-32}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_SANDBOXES="${MAX_SANDBOXES:-30}"
SANDBOX="${SANDBOX:-daytona}"
SAMPLES="${SAMPLES:-4}"  # held-out episodes per test task
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.7}"  # the shared evaluator's defaults
EVAL_TOP_P="${EVAL_TOP_P:-0.8}"
# Tinker's OpenAI-compatible endpoint: base models by name, checkpoints by
# their tinker://.../sampler_weights/... path.
TINKER_OAI_URL="${TINKER_OAI_URL:-https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1}"

# Credentials come from the environment, never the command line.
: "${TINKER_API_KEY:?set TINKER_API_KEY (Tinker console)}"
if [ "$SANDBOX" = daytona ]; then
  : "${DAYTONA_API_KEY:?set DAYTONA_API_KEY, or run with SANDBOX=docker}"
  # Labels the sandboxes: `bench sandbox list` / `bench sandbox cleanup --all`.
  export BENCHFLOW_DAYTONA_OWNER="${BENCHFLOW_DAYTONA_OWNER:-tinker-rl-${USER:-me}}"
fi

cd "$REPO"
PY=(uv run --no-sync python)  # tinker-cookbook is installed into this venv; see README
if ! "${PY[@]}" -c "import tinker_cookbook" 2>/dev/null; then
  echo "tinker-cookbook is not installed in this venv; see README.md, Prerequisites" >&2
  exit 2
fi
renderer=()
if [ -n "$RENDERER" ]; then renderer=(--renderer "$RENDERER"); fi

# The shared task family, generated from its seeds when absent.
family() {
  if [ ! -d "$FAMILY/train" ] || [ ! -d "$FAMILY/test" ]; then
    for split in train test; do
      "${PY[@]}" docs/examples/rl/tasks/generate.py --split "$split" --out "$FAMILY/$split"
    done
  fi
}

# One held-out evaluation with the shared evaluator, through Tinker's
# OpenAI-compatible endpoint: evaluate LABEL MODEL_OR_CHECKPOINT
evaluate() {
  "${PY[@]}" docs/examples/rl/common/evaluate.py \
    --tasks-dir "$FAMILY/test" --base-url "$TINKER_OAI_URL" --model "$2" \
    --api-key-env TINKER_API_KEY --sandbox "$SANDBOX" --concurrency "$MAX_SANDBOXES" \
    --samples "$SAMPLES" --max-tokens "$MAX_TOKENS" \
    --temperature "$EVAL_TEMPERATURE" --top-p "$EVAL_TOP_P" \
    ${BENCHFLOW_DAYTONA_OWNER:+--owner "$BENCHFLOW_DAYTONA_OWNER"} \
    --out "$OUT/eval-$1"
}

OUT="$WORK/runs/$STAGE-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
echo "tinker $STAGE: $MODEL, sandbox $SANDBOX (at most $MAX_SANDBOXES) -> $OUT"

case "$STAGE" in
  smoke)
    # The bundled hello-world task works in /app, not the family's /workdir.
    "${PY[@]}" "$HERE/tinker_train.py" --tasks-dir src/benchflow/demo_task \
      --model "$MODEL" ${renderer[@]+"${renderer[@]}"} --sandbox "$SANDBOX" \
      --harness-message "" --submit-path /app/answer.txt \
      --steps 2 --group-size 2 --groups-per-batch 1 --max-sandboxes 4 --save-every 0 \
      --log-path "$OUT/train"
    ;;
  baseline)
    family
    # Two tasks of each kind and level, two episodes each, at the training temperature.
    include=()
    while IFS= read -r task; do include+=(--include "$task"); done < <(
      "${PY[@]}" - "$FAMILY/train/manifest.jsonl" <<'EOF'
import collections, json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
strata = collections.defaultdict(list)
for row in sorted(rows, key=lambda r: r["task"]):
    strata[(row["kind"], row["level"])].append(row["task"])
for key in sorted(strata):
    print("\n".join(strata[key][:2]))
EOF
    )
    "${PY[@]}" "$HERE/tinker_eval.py" --tasks-dir "$FAMILY/train" "${include[@]}" \
      --policy base --model "$MODEL" ${renderer[@]+"${renderer[@]}"} --sandbox "$SANDBOX" \
      --samples 2 --max-tokens "$MAX_TOKENS" --max-sandboxes "$MAX_SANDBOXES" \
      --out "$OUT/eval.json"
    ;;
  train)
    family
    evaluate base "$MODEL"
    "${PY[@]}" "$HERE/tinker_train.py" --tasks-dir "$FAMILY/train" \
      --model "$MODEL" ${renderer[@]+"${renderer[@]}"} --sandbox "$SANDBOX" \
      --steps "$STEPS" --group-size "$GROUP_SIZE" --groups-per-batch "$GROUPS_PER_BATCH" \
      --learning-rate "$LEARNING_RATE" --lora-rank "$LORA_RANK" --max-tokens "$MAX_TOKENS" \
      --max-sandboxes "$MAX_SANDBOXES" --log-path "$OUT/train"
    final=$("${PY[@]}" - "$OUT/train/checkpoints.jsonl" <<'EOF'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
print([r for r in rows if r.get("sampler_path")][-1]["sampler_path"])
EOF
    )
    echo "final checkpoint: $final"
    evaluate trained "$final"
    "${PY[@]}" "$HERE/tinker_stats.py" "$OUT/eval-base" "$OUT/eval-trained" \
      | tee "$OUT/comparison.txt"
    ;;
  *)
    echo "usage: $0 smoke|baseline|train" >&2
    exit 2
    ;;
esac

if [ "$SANDBOX" = daytona ]; then
  uv run --no-sync bench sandbox list | tail -1  # expect 0 left
fi
