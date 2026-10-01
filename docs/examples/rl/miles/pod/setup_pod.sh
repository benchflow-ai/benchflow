#!/usr/bin/env bash
# Runs ON the GPU host (a rented pod, as its login user with sudo): firewall it,
# start the Miles container, and install BenchFlow, the example and the prompt data.
#
# Expects /work/benchflow (a `git archive` of this BenchFlow branch) and
# /work/tasks/{train,test} (task folders without oracle/). No secret is read or
# written here: the Daytona key reaches only the jobs that need it (job.sh, started
# by run_on_pod.sh from the machine that holds the key).
#
#   MILES_COMMIT=077fb59 bash setup_pod.sh
set -euo pipefail
log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# Miles' session servers, SGLang, Ray and its dashboard listen on the node's
# address. A pod with a public address takes SSH only.
if command -v ufw >/dev/null; then
  sudo ufw default deny incoming >/dev/null
  sudo ufw default allow outgoing >/dev/null
  sudo ufw allow 22/tcp >/dev/null
  sudo ufw --force enable >/dev/null
  log "firewall: inbound SSH only"
else
  log "WARNING: no ufw; allow only SSH inbound some other way before training"
fi
id -nG | grep -qw docker || sudo usermod -aG docker "$USER"
docker() { sudo docker "$@"; }

driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# The CUDA 13 build needs driver 580 or newer. Pin a dated tag, not latest.
default_image=radixark/miles:dev-202609302131
[ "${driver%%.*}" -ge 580 ] || default_image=radixark/miles:dev-cu12-202609302230
image=${MILES_IMAGE:-$default_image}
log "pulling $image"
docker pull -q "$image"
docker image inspect "$image" --format '{{index .RepoDigests 0}}'
docker rm -f miles >/dev/null 2>&1 || true
# SYS_PTRACE: colocated FSDP hands its weights to SGLang as CUDA IPC tensors, which
# PyTorch shares through pidfd_getfd under expandable_segments; Docker's default seccomp
# profile allows that call only with this capability ("pidfd_getfd: Operation not
# permitted" at the first weight update otherwise).
docker run -d --name miles --gpus all --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add SYS_PTRACE --network=host -v /work:/work "$image" sleep infinity >/dev/null

log "Miles at ${MILES_COMMIT:-the commit the image cloned}; BenchFlow venv, example and prompt data"
docker exec -e MILES_COMMIT="${MILES_COMMIT:-}" miles bash -lc '
  set -euo pipefail
  if [ -n "$MILES_COMMIT" ]; then
    git -C /root/miles fetch -q origin && git -C /root/miles checkout -q "$MILES_COMMIT"
  fi
  # dev-202609302131 pairs opentelemetry-api 1.45.0 with the 1.44 SDK; the Ray dashboard
  # agent then fails to import, the raylet exits with it, and `ray start` times out.
  sdk=$(pip show opentelemetry-sdk 2>/dev/null | sed -n "s/^Version: //p")
  api=$(pip show opentelemetry-api 2>/dev/null | sed -n "s/^Version: //p")
  if [ -n "$sdk" ] && [ "$sdk" != "$api" ]; then
    echo "aligning opentelemetry-api $api with opentelemetry-sdk $sdk"
    pip install -q "opentelemetry-api==$sdk"
  fi
  command -v uv >/dev/null || pip install -q uv
  cd /work/benchflow
  uv venv -q --allow-existing .venv --python 3.12 && uv sync -q --locked --extra sandbox-daytona
  rm -rf /root/miles/examples/experimental/benchflow
  cp -r docs/examples/rl/miles/upstream/examples/experimental/benchflow /root/miles/examples/experimental/
  for split in train test; do
    .venv/bin/python -m benchflow.integrations.miles prepare --tasks-dir /work/tasks/$split \
      --out /work/$split.jsonl --split $split
  done
  echo "miles $(git -C /root/miles rev-parse --short HEAD), sglang $(python -c "import sglang; print(sglang.__version__)"), torch $(python -c "import torch; print(torch.__version__, torch.version.cuda)")"
'
log "ready"
