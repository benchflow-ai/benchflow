#!/usr/bin/env bash
# Runs ON the pod: install prime-rl (pinned), a BenchFlow venv, and benchflow-taskset.
#
# Expects the BenchFlow checkout at $WORK/benchflow (the caller streams it in with
# `git archive`), and the task folders at $WORK/tasks. No secret is read or written
# here: the Daytona key reaches the pod only as the environment of the processes that
# need it (see run_on_pod.sh).
#
#   WORK=~/bf-rl PRIME_RL_REF=<commit> bash setup_pod.sh
set -euo pipefail

WORK="${WORK:-$HOME/bf-rl}"
PRIME_RL_REF="${PRIME_RL_REF:-d5f29c072a6731a17897f5aed661e0b0e6053d8c}"
mkdir -p "$WORK"
cd "$WORK"

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# prime-rl's submodules use SSH URLs; a pod has no GitHub key.
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0="url.https://github.com/.insteadOf"
export GIT_CONFIG_VALUE_0="git@github.com:"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  # Leave shell profiles alone: some images ship a root-owned ~/.config, and the
  # installer's profile edit then fails the whole setup.
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi

log "prime-rl at $PRIME_RL_REF"
if [ ! -d prime-rl/.git ]; then
  git clone -q https://github.com/PrimeIntellect-ai/prime-rl.git
fi
git -C prime-rl fetch -q origin
git -C prime-rl checkout -q "$PRIME_RL_REF"
git -C prime-rl submodule update --init --recursive -q

log "prime-rl venv (uv sync --all-extras)"
(cd prime-rl && uv sync --all-extras)

log "benchflow-taskset into prime-rl's venv (no deps: prime-rl pins verifiers)"
uv pip install --python prime-rl/.venv/bin/python --no-deps -e "$WORK/benchflow/docs/examples/rl/prime/benchflow_taskset"

log "BenchFlow venv (the bridge runs here)"
(cd "$WORK/benchflow" && uv sync --extra sandbox-daytona)

log "checks"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
prime-rl/.venv/bin/python - <<'EOF'
import importlib.metadata as md
import torch
import verifiers.v1 as vf
import benchflow_taskset
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), torch.cuda.device_count())
print("verifiers", md.version("verifiers"), "mcp", md.version("mcp"), "vllm", md.version("vllm"))
print("env for benchflow-taskset:", vf.environment_class("benchflow-taskset").__name__)
EOF
"$WORK/benchflow/.venv/bin/python" -c 'import importlib.metadata as md, benchflow.integrations.rewards; print("benchflow", md.version("benchflow"), "rewards ok")'
BENCHFLOW_PYTHON="$WORK/benchflow/.venv/bin/python" prime-rl/.venv/bin/python - <<EOF
import benchflow_taskset as bt
tasks = list(bt.BenchFlowTaskset(bt.BenchFlowConfig(id="benchflow-taskset", tasks_dir="$WORK/tasks/train")))
print(len(tasks), "train tasks; first:", tasks[0].data.name)
EOF
log "setup done"
