#!/usr/bin/env bash
# Runs ON the GPU host: one step of the Miles cookbook, inside the `miles` container
# that setup_pod.sh started. Logs go to /work/logs.
#
#   job.sh download HF_REPO...           models into /work/models/<name>
#   job.sh serve-policy MODEL_DIR        SGLang on 127.0.0.1:30100, for the evaluator
#   job.sh stop-policy
#   job.sh eval NAME SPLIT [ARGS...]     the shared evaluator against it -> /work/eval/NAME   (Daytona key)
#   job.sh serve-env NAME [ARGS...]      the BenchFlow environment server on 127.0.0.1:12100    (Daytona key)
#   job.sh stop-env
#   job.sh train MODEL_NAME [ARGS...]    examples/experimental/benchflow/run.py
#   job.sh convert CKPT_DIR OUT_DIR ORIGIN_HF_DIR
#
# Start the steps marked "Daytona key" with run_on_pod.sh, which streams the key
# over ssh's stdin into this script's environment. This script hands it, by name,
# to the BenchFlow process that needs it and to nothing else: never to Miles,
# never to a file, never on a command line.
set -euo pipefail
step=$1; shift
owner=${BENCHFLOW_DAYTONA_OWNER:-}
docker() { if id -nG | grep -qw docker; then command docker "$@"; else sudo docker "$@"; fi; }
with_key() {
  [ -n "${DAYTONA_API_KEY:-}" ] || { echo "no DAYTONA_API_KEY: start this step with run_on_pod.sh" >&2; exit 2; }
  docker exec -e DAYTONA_API_KEY -e BENCHFLOW_DAYTONA_OWNER="$owner" miles bash -c "$1"
}
without_key() { docker exec miles bash -c "$1"; }
q() { printf '%q ' "$@"; }
mkdir -p /work/logs

case "$step" in
  download)
    for repo in "$@"; do
      without_key "hf download $(q "$repo") --local-dir /work/models/$(q "${repo##*/}") > /dev/null && echo $(q "$repo") downloaded"
    done
    ;;
  serve-policy)
    without_key "nohup python -m sglang.launch_server --model-path $(q "$1") --served-model-name policy \
      --tool-call-parser qwen25 --reasoning-parser qwen3 --host 127.0.0.1 --port 30100 \
      --mem-fraction-static 0.85 > /work/logs/sglang-policy.log 2>&1 &
      for i in \$(seq 1 120); do curl -sf 127.0.0.1:30100/health >/dev/null && echo up && exit 0; sleep 5; done
      echo 'SGLang did not come up' >&2; exit 1"
    ;;
  stop-policy)
    without_key "pkill -f '[s]glang.launch_server' || true; sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader"
    ;;
  eval)
    name=$1 split=$2; shift 2
    with_key "cd /work/benchflow && SGLANG_KEY=unused .venv/bin/python docs/examples/rl/common/evaluate.py \
      --tasks-dir /work/tasks/$split --base-url http://127.0.0.1:30100/v1 --model policy --api-key-env SGLANG_KEY \
      --sandbox daytona --extra-body '{\"chat_template_kwargs\": {\"enable_thinking\": false}}' \
      --out /work/eval/$name $(q "$@") > /work/logs/eval-$name.log 2>&1; tail -3 /work/logs/eval-$name.log"
    ;;
  serve-env)
    name=$1; shift
    with_key "cd /work/benchflow && nohup .venv/bin/python -m benchflow.integrations.miles serve \
      --tasks-dir /work/tasks/train --sandbox daytona --jobs-dir /work/jobs --job-name $(q "$name") $(q "$@") \
      > /work/logs/env-$name.log 2>&1 &
      for i in \$(seq 1 60); do curl -sf 127.0.0.1:12100/health && echo && exit 0; sleep 2; done
      echo 'the environment server did not come up' >&2; exit 1"
    ;;
  stop-env)
    # SIGTERM: the server cancels its episodes and releases their sandboxes.
    without_key "pkill -TERM -f '[b]enchflow.integrations.miles serve' || true; sleep 15; curl -sf 127.0.0.1:12100/health || echo stopped"
    ;;
  train)
    model=$1; shift
    unset DAYTONA_API_KEY
    without_key "cd /root/miles && python examples/experimental/benchflow/run.py --model-name $(q "$model") \
      --model-dir /work/models --prompt-data /work/train.jsonl --save-dir /work/ckpt $(q "$@") \
      > /work/logs/train.log 2>&1"
    ;;
  convert)
    without_key "cd /root/miles && python tools/convert_fsdp_to_hf.py --input-dir $(q "$1") --output-dir $(q "$2") \
      --origin-hf-dir $(q "$3") > /work/logs/convert.log 2>&1; tail -3 /work/logs/convert.log"
    ;;
  *) echo "unknown step $step" >&2; exit 2 ;;
esac
