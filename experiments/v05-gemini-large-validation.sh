#!/usr/bin/env bash
# v0.5 Cycle F — gemini ACP large validation (c=100, Daytona).
#
# Batches (audited after each):
#   1. SkillsBench — no-skills / deployed / self-gen (subset or full)
#   2. Terminal-Bench 2.0 representative tasks
#   3. Adapter smokes — Harvey LAB + ProgramBench (50 tasks by default)
#
# Usage:
#   GEMINI_API_KEY=... DAYTONA_API_KEY=... experiments/v05-gemini-large-validation.sh
#   experiments/v05-gemini-large-validation.sh --check-only jobs/cycle-f-<run-id>
#   BENCHFLOW_CYCLE_F_SCOPE=full ...   # all 94 SkillsBench tasks × 3 modes
#
# Scope doc written to jobs/cycle-f-<run-id>/scope.json after each batch.
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"

CONCURRENCY="${BENCHFLOW_INTEGRATION_CONCURRENCY:-100}"
AGENT="${BENCHFLOW_CYCLE_F_AGENT:-gemini}"
MODEL="${BENCHFLOW_CYCLE_F_MODEL:-gemini-3.1-flash-lite-preview}"
SANDBOX="${BENCHFLOW_CYCLE_F_SANDBOX:-daytona}"
IDLE_TIMEOUT="${BENCHFLOW_AGENT_IDLE_TIMEOUT:-600}"
SELF_GEN_IDLE_TIMEOUT="${BENCHFLOW_CYCLE_F_SELF_GEN_IDLE_TIMEOUT:-1800}"
SCOPE="${BENCHFLOW_CYCLE_F_SCOPE:-subset}"
ADAPTER_TASK_BUDGET="${BENCHFLOW_CYCLE_F_ADAPTER_TASKS:-50}"
TB2_TASK_BUDGET="${BENCHFLOW_CYCLE_F_TB2_TASKS:-15}"

CHECK_ONLY=false
JOBS_ROOT=""

for arg in "$@"; do
  case "$arg" in
    --check-only) CHECK_ONLY=true ;;
    *)
      if [ -z "$JOBS_ROOT" ]; then
        JOBS_ROOT="$arg"
      fi
      ;;
  esac
done

AUDIT_ARGS=(
  "agent=$AGENT"
  "model=$MODEL"
  "environment=$SANDBOX"
  "concurrency=$CONCURRENCY"
  "agent_idle_timeout_sec=$IDLE_TIMEOUT"
)

run_audit() {
  local label="$1"
  local root="$2"
  shift 2
  echo ""
  echo "══════ Audit: $label ══════"
  uv run python tests/integration/check_results.py "$root" "${AUDIT_ARGS[@]}" "$@" || return 1
}

audit_idle_for_batch() {
  local batch="$1"
  if [ "$batch" = "skillsbench-self-gen" ]; then
    echo "$SELF_GEN_IDLE_TIMEOUT"
  else
    echo "$IDLE_TIMEOUT"
  fi
}

run_adapter_audit() {
  local skillsbench_result="$1"
  echo ""
  echo "══════ Adapter evidence audit ══════"
  if [ -z "$skillsbench_result" ] || [ ! -f "$skillsbench_result" ]; then
    echo "WARN: no SkillsBench result.json for adapter evidence — skipping"
    return 0
  fi
  uv run python tests/integration/check_adapter_evidence.py \
    --skillsbench-result "$skillsbench_result" || return 1
}

write_scope() {
  local root="$1"
  uv run python - "$root" "$SCOPE" "$CONCURRENCY" <<'PY'
import json, sys
from pathlib import Path
root, scope, concurrency = sys.argv[1:4]
doc = {
    "cycle": "F",
    "scope": scope,
    "concurrency": int(concurrency),
    "agent": "gemini",
    "model": "gemini-3.1-flash-lite-preview",
    "sandbox": "daytona",
    "batches": {},
}
for batch_dir in sorted(Path(root).glob("*")):
    if not batch_dir.is_dir() or batch_dir.name.startswith("."):
        continue
    summary = batch_dir / "summary.json"
    if summary.exists():
        doc["batches"][batch_dir.name] = {
            "summary": str(summary),
            "tasks": json.loads(summary.read_text()).get("total"),
        }
Path(root, "scope.json").write_text(json.dumps(doc, indent=2) + "\n")
print(f"Wrote scope doc: {root}/scope.json")
PY
}

if [ "$CHECK_ONLY" = true ]; then
  if [ -z "$JOBS_ROOT" ]; then
    echo "ERROR: --check-only requires jobs root path" >&2
    exit 1
  fi
  audit_fail=0
  for batch in "$JOBS_ROOT"/*/; do
    [ -d "$batch" ] || continue
    name="$(basename "$batch")"
    run_audit "$name" "$batch" "agent_idle_timeout_sec=$(audit_idle_for_batch "$name")" || audit_fail=1
  done
  skills_result="$(find "$JOBS_ROOT/skillsbench-deployed" -name result.json 2>/dev/null | head -1)"
  if [ -z "$skills_result" ]; then
    skills_result="$(find "$JOBS_ROOT/skillsbench-no-skills" -name result.json 2>/dev/null | head -1)"
  fi
  run_adapter_audit "$skills_result" || audit_fail=1
  exit "$audit_fail"
fi

if [ -z "${GEMINI_API_KEY:-}" ] && [ -z "${GOOGLE_API_KEY:-}" ]; then
  echo "ERROR: GEMINI_API_KEY or GOOGLE_API_KEY required" >&2
  exit 1
fi
if [ -z "${DAYTONA_API_KEY:-}" ]; then
  echo "ERROR: DAYTONA_API_KEY required" >&2
  exit 1
fi

RUN_ID="${BENCHFLOW_CYCLE_F_RUN_ID:-cycle-f-$(date +%Y%m%d-%H%M%S)}"
JOBS_ROOT="${BENCHFLOW_CYCLE_F_JOBS_ROOT:-jobs/$RUN_ID}"
CONFIG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/benchflow-cycle-f-config-$RUN_ID.XXXXXX")"

echo "Run ID: $RUN_ID"
echo "Jobs: $JOBS_ROOT"
echo "Scope: $SCOPE (adapter_tasks=$ADAPTER_TASK_BUDGET tb2_tasks=$TB2_TASK_BUDGET)"
echo "Config dir: $CONFIG_DIR"

mkdir -p "$JOBS_ROOT"

# Generate YAML configs + task lists
uv run python - "$CONFIG_DIR" "$SCOPE" "$ADAPTER_TASK_BUDGET" "$TB2_TASK_BUDGET" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from benchflow._utils.benchmark_repos import resolve_source_with_metadata

config_dir = Path(sys.argv[1])
scope = sys.argv[2]
adapter_budget = int(sys.argv[3])
tb2_budget = int(sys.argv[4])

RELEASE_SUBSET = [
    "jax-computing-basics",
    "python-scala-translation",
    "jpg-ocr-stat",
    "grid-dispatch-operator",
    "threejs-to-obj",
    "data-to-d3",
    "lake-warming-attribution",
    "weighted-gdp-calc",
    "shock-analysis-supply",
]

TB2_REPRESENTATIVE = [
    "adaptive-rejection-sampler",
    "bn-fit-modify",
    "cancel-async-tasks",
    "cobol-modernization",
    "code-from-image",
    "distribution-search",
    "filter-js-from-html",
    "git-multibranch",
    "hf-model-inference",
    "kv-store-grpc",
    "log-summary-date-ranges",
    "mailman",
    "modernize-scientific-stack",
    "nginx-request-logging",
    "path-tracing",
]

skills_root = resolve_source_with_metadata(
    "benchflow-ai/skillsbench", path="tasks", ref="main"
).path
all_skills = sorted(
    d.name
    for d in skills_root.iterdir()
    if d.is_dir() and (d / "task.toml").exists()
)
skills_include = all_skills if scope == "full" else RELEASE_SUBSET

harvey_root = resolve_source_with_metadata(
    "benchflow-ai/benchmarks", path="datasets/harvey-lab/tasks", ref="main"
).path
pb_root = resolve_source_with_metadata(
    "benchflow-ai/benchmarks", path="datasets/programbench/tasks", ref="main"
).path
tb2_root = resolve_source_with_metadata("harbor-framework/terminal-bench-2").path

harvey_tasks = sorted(
    d.name for d in harvey_root.iterdir() if d.is_dir() and (d / "task.toml").exists()
)
pb_tasks = sorted(
    d.name for d in pb_root.iterdir() if d.is_dir() and (d / "task.toml").exists()
)
tb2_tasks = sorted(
    d.name for d in tb2_root.iterdir() if d.is_dir() and (d / "task.toml").exists()
)

def sample(tasks: list[str], n: int) -> list[str]:
    if n >= len(tasks):
        return tasks
    scored = sorted(tasks, key=lambda t: hashlib.sha256(t.encode()).hexdigest())
    step = max(1, len(scored) // n)
    picked = [scored[i * step] for i in range(n)]
    return picked[:n]

harvey_n = adapter_budget // 2
pb_n = adapter_budget - harvey_n
harvey_sample = sample(harvey_tasks, harvey_n)
pb_sample = sample(pb_tasks, pb_n)
tb2_include = [t for t in TB2_REPRESENTATIVE if t in tb2_tasks][:tb2_budget]
if len(tb2_include) < tb2_budget:
    tb2_include.extend(sample([t for t in tb2_tasks if t not in tb2_include], tb2_budget - len(tb2_include)))

common = {
    "agent": "gemini",
    "model": "gemini-3.1-flash-lite-preview",
    "environment": "daytona",
    "concurrency": 100,
    "max_retries": 0,
}

def write_yaml(name: str, body: dict) -> None:
    import yaml

    path = config_dir / name
    path.write_text(yaml.safe_dump(body, sort_keys=False))
    print(f"config {path.name}: {len(body.get('include', body.get('tasks', [])))} tasks")

write_yaml(
    "skillsbench-no-skills.yaml",
    {
        **common,
        "source": {"repo": "benchflow-ai/skillsbench", "path": "tasks", "ref": "main"},
        "include": skills_include,
        "skill_mode": "default",
    },
)
write_yaml(
    "skillsbench-deployed.yaml",
    {
        **common,
        "source": {"repo": "benchflow-ai/skillsbench", "path": "tasks", "ref": "main"},
        "include": skills_include,
        "skills_dir": "auto",
        "skill_mode": "default",
    },
)
write_yaml(
    "skillsbench-self-gen.yaml",
    {
        **common,
        "source": {"repo": "benchflow-ai/skillsbench", "path": "tasks", "ref": "main"},
        "include": skills_include,
        "skill_mode": "self-gen",
        "self_gen_no_internet": True,
    },
)
write_yaml(
    "tb2-representative.yaml",
    {
        **common,
        "tasks_dir": str(tb2_root),
        "include": tb2_include,
    },
)
write_yaml(
    "adapters-harvey.yaml",
    {
        **common,
        "source": {
            "repo": "benchflow-ai/benchmarks",
            "path": "datasets/harvey-lab/tasks",
            "ref": "main",
        },
        "include": harvey_sample,
    },
)
write_yaml(
    "adapters-programbench.yaml",
    {
        **common,
        "source": {
            "repo": "benchflow-ai/benchmarks",
            "path": "datasets/programbench/tasks",
            "ref": "main",
        },
        "include": pb_sample,
    },
)

manifest = {
    "scope": scope,
    "skillsbench_tasks": len(skills_include),
    "skillsbench_modes": ["no-skills", "deployed", "self-gen"],
    "skillsbench_total_trials": len(skills_include) * 3,
    "tb2_tasks": tb2_include,
    "adapter_harvey": harvey_sample,
    "adapter_programbench": pb_sample,
    "adapter_total": len(harvey_sample) + len(pb_sample),
    "remaining_for_full_run": {
        "skillsbench_all_tasks": len(all_skills) - len(skills_include) if scope != "full" else 0,
        "harvey_remaining": len(harvey_tasks) - len(harvey_sample),
        "programbench_remaining": len(pb_tasks) - len(pb_sample),
        "tb2_remaining": len(tb2_tasks) - len(tb2_include),
    },
}
(config_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

cp "$CONFIG_DIR/manifest.json" "$JOBS_ROOT/manifest.json"

batch_eval() {
  local batch="$1"
  local config="$2"
  local idle="$3"
  local jobs="$JOBS_ROOT/$batch"
  echo ""
  echo "══════ Batch: $batch ══════"
  eval_status=0
  uv run bench eval create \
    --config "$config" \
    --concurrency "$CONCURRENCY" \
    --agent-idle-timeout "$idle" \
    --jobs-dir "$jobs" || eval_status=$?
  run_audit "$batch" "$jobs" "agent_idle_timeout_sec=$idle" || return 1
  write_scope "$JOBS_ROOT"
  return "$eval_status"
}

overall=0

batch_eval "skillsbench-no-skills" "$CONFIG_DIR/skillsbench-no-skills.yaml" "$IDLE_TIMEOUT" || overall=1
batch_eval "skillsbench-deployed" "$CONFIG_DIR/skillsbench-deployed.yaml" "$IDLE_TIMEOUT" || overall=1
batch_eval "skillsbench-self-gen" "$CONFIG_DIR/skillsbench-self-gen.yaml" "$SELF_GEN_IDLE_TIMEOUT" || overall=1
batch_eval "tb2-representative" "$CONFIG_DIR/tb2-representative.yaml" "$IDLE_TIMEOUT" || overall=1
batch_eval "adapters-harvey" "$CONFIG_DIR/adapters-harvey.yaml" "$IDLE_TIMEOUT" || overall=1
batch_eval "adapters-programbench" "$CONFIG_DIR/adapters-programbench.yaml" "$IDLE_TIMEOUT" || overall=1

skills_result="$(find "$JOBS_ROOT/skillsbench-deployed" -name result.json 2>/dev/null | head -1)"
run_adapter_audit "$skills_result" || overall=1

echo ""
echo "Cycle F complete — jobs root: $JOBS_ROOT"
echo "HEAD: $(git rev-parse --short HEAD)"
exit "$overall"
