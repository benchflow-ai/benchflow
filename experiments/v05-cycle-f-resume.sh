#!/usr/bin/env bash
# Resume Cycle F from batch 2 after skillsbench-no-skills completed.
set -euo pipefail
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
source /tmp/benchflow-cycle-c.env

RUN_ID="${1:-cycle-f-20260522-115820}"
JOBS_ROOT="jobs/$RUN_ID"
CONCURRENCY=100
AGENT=gemini
MODEL=gemini-3.1-flash-lite-preview
IDLE=600
SELF_GEN_IDLE=1800

CONFIG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/benchflow-cycle-f-resume.XXXXXX")"
export BENCHFLOW_CYCLE_F_SCOPE=subset BENCHFLOW_CYCLE_F_ADAPTER_TASKS=50 BENCHFLOW_CYCLE_F_TB2_TASKS=15

# Regenerate configs (same as large-validation script)
uv run python - "$CONFIG_DIR" subset 50 15 <<'PY'
import hashlib, json, sys
from pathlib import Path
from benchflow._utils.benchmark_repos import resolve_source_with_metadata
config_dir = Path(sys.argv[1])
RELEASE_SUBSET = [
    "jax-computing-basics", "python-scala-translation", "jpg-ocr-stat",
    "grid-dispatch-operator", "threejs-to-obj", "data-to-d3",
    "lake-warming-attribution", "weighted-gdp-calc", "shock-analysis-supply",
]
TB2_REPRESENTATIVE = [
    "adaptive-rejection-sampler", "bn-fit-modify", "cancel-async-tasks",
    "cobol-modernization", "code-from-image", "distribution-search",
    "filter-js-from-html", "git-multibranch", "hf-model-inference",
    "kv-store-grpc", "log-summary-date-ranges", "mailman",
    "modernize-scientific-stack", "nginx-request-logging", "path-tracing",
]
import yaml
common = {"agent": "gemini", "model": "gemini-3.1-flash-lite-preview", "environment": "daytona", "concurrency": 100, "max_retries": 0}
skills_root = resolve_source_with_metadata("benchflow-ai/skillsbench", path="tasks", ref="main").path
skills_include = RELEASE_SUBSET
harvey_root = resolve_source_with_metadata("benchflow-ai/benchmarks", path="datasets/harvey-lab/tasks", ref="main").path
pb_root = resolve_source_with_metadata("benchflow-ai/benchmarks", path="datasets/programbench/tasks", ref="main").path
tb2_root = resolve_source_with_metadata("harbor-framework/terminal-bench-2").path
harvey_tasks = sorted(d.name for d in harvey_root.iterdir() if d.is_dir() and (d/"task.toml").exists())
pb_tasks = sorted(d.name for d in pb_root.iterdir() if d.is_dir() and (d/"task.toml").exists())
tb2_tasks = sorted(d.name for d in tb2_root.iterdir() if d.is_dir() and (d/"task.toml").exists())
def sample(tasks, n):
    scored = sorted(tasks, key=lambda t: hashlib.sha256(t.encode()).hexdigest())
    step = max(1, len(scored)//n)
    return [scored[i*step] for i in range(n)][:n]
harvey_sample = sample(harvey_tasks, 25)
pb_sample = sample(pb_tasks, 25)
tb2_include = [t for t in TB2_REPRESENTATIVE if t in tb2_tasks][:15]
def w(name, body):
    (config_dir/name).write_text(yaml.safe_dump(body, sort_keys=False))
w("skillsbench-deployed.yaml", {**common, "source": {"repo": "benchflow-ai/skillsbench", "path": "tasks", "ref": "main"}, "include": skills_include, "skills_dir": "auto"})
w("skillsbench-self-gen.yaml", {**common, "source": {"repo": "benchflow-ai/skillsbench", "path": "tasks", "ref": "main"}, "include": skills_include, "skill_mode": "self-gen", "self_gen_no_internet": True})
w("tb2-representative.yaml", {**common, "tasks_dir": str(tb2_root), "include": tb2_include})
w("adapters-harvey.yaml", {**common, "source": {"repo": "benchflow-ai/benchmarks", "path": "datasets/harvey-lab/tasks", "ref": "main"}, "include": harvey_sample})
w("adapters-programbench.yaml", {**common, "source": {"repo": "benchflow-ai/benchmarks", "path": "datasets/programbench/tasks", "ref": "main"}, "include": pb_sample})
print(config_dir)
PY

batch() {
  local name="$1" cfg="$2" idle="$3"
  echo "══════ $name ══════"
  uv run bench eval create --config "$cfg" --concurrency "$CONCURRENCY" --agent-idle-timeout "$idle" --jobs-dir "$JOBS_ROOT/$name" || true
  uv run python tests/integration/check_results.py "$JOBS_ROOT/$name" agent=$AGENT model=$MODEL environment=daytona concurrency=$CONCURRENCY agent_idle_timeout_sec=$IDLE || true
}

batch skillsbench-deployed "$CONFIG_DIR/skillsbench-deployed.yaml" "$IDLE"
batch skillsbench-self-gen "$CONFIG_DIR/skillsbench-self-gen.yaml" "$SELF_GEN_IDLE"
batch tb2-representative "$CONFIG_DIR/tb2-representative.yaml" "$IDLE"
batch adapters-harvey "$CONFIG_DIR/adapters-harvey.yaml" "$IDLE"
batch adapters-programbench "$CONFIG_DIR/adapters-programbench.yaml" "$IDLE"

skills_result="$(find "$JOBS_ROOT/skillsbench-deployed" -name result.json 2>/dev/null | head -1)"
uv run python tests/integration/check_adapter_evidence.py --skillsbench-result "$skills_result" || true
echo "Resume complete: $JOBS_ROOT HEAD=$(git rev-parse --short HEAD)"
