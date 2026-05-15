# Running Adapted Benchmarks

How to run benchmarks that have been converted to BenchFlow format.

BenchFlow ships with adapted benchmarks under `benchmarks/<name>/`. Each benchmark
includes a converter, parity tests, metadata, and one or more YAML job configs.
This guide covers how to run them — from a single task to a full evaluation sweep.

---

## Available benchmarks

| Benchmark | Tasks | Verification | Config |
|-----------|-------|--------------|--------|
| [Harvey LAB](https://github.com/harveyai/harvey-labs) | 1,251 | LLM-as-judge (per-criterion) | `benchmarks/harvey-lab/` |
| [SkillsBench](https://github.com/benchflow-ai/skillsbench) | 94+ | Unit tests | `benchmarks/skillsbench-*.yaml` |
| [Terminal-Bench 2](https://github.com/harbor-framework/terminal-bench-2) | 91 | Script | `benchmarks/tb2-*.yaml` |

Each adapted benchmark includes:
- **`benchflow.py`** — converter: raw benchmark → BenchFlow task format
- **`benchmark.yaml`** — metadata descriptor (task count, categories, verification method, parity results)
- **`<name>-*.yaml`** — job configs for different agents/models
- **`parity_test.py`** — parity validation suite
- **`parity_experiment.json`** — recorded parity results

---

## Quick start

### Option 1: YAML config (`bench eval create -f`)

The simplest path. Point at a YAML config that specifies the source repo:

```bash
# Harvey LAB with Gemini
GEMINI_API_KEY=... bench eval create -f benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml

# SkillsBench with Claude
bench eval create -f benchmarks/skillsbench-claude-glm51.yaml
```

The config handles everything — downloads the source repo, resolves the task path,
and runs the evaluation.

### Option 2: CLI with `--source-repo`

Use the CLI flags for ad-hoc runs without a config file:

```bash
# Harvey LAB — all converted tasks
bench eval create \
  --source-repo harveyai/harvey-labs \
  --source-path tasks \
  -a gemini \
  -m gemini-3.1-flash-lite-preview \
  -e docker \
  -c 4

# SkillsBench — single task
bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks/edit-pdf \
  -a gemini \
  -m gemini-3.1-flash-lite-preview \
  -e docker

# Or with Claude Code on Daytona
bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks \
  -a claude-agent-acp \
  -m anthropic/claude-sonnet-4-6 \
  -e daytona \
  -c 32
```

### Option 3: Python API

For programmatic use, custom pipelines, or integration with other tools:

```python
import asyncio
from benchflow.job import Job

async def main():
    # Run from YAML config (auto-downloads source repo)
    job = Job.from_yaml("benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml")
    result = await job.run()
    print(f"Score: {result.passed}/{result.total} ({result.score:.1%})")

asyncio.run(main())
```

For single-task runs:

```python
import benchflow as bf
from benchflow.trial import TrialConfig, Scene
from benchflow.task_download import resolve_source

task_path = resolve_source("benchflow-ai/skillsbench", path="tasks/edit-pdf")

config = TrialConfig(
    task_path=task_path,
    scenes=[Scene.single(agent="gemini", model="gemini-3.1-flash-lite-preview")],
    environment="docker",
)
result = await bf.run(config)
print(result.rewards)
```

---

## Running a subset of tasks

### Using `--split` (at conversion time)

The Harvey LAB converter supports named splits for generating subsets:

```bash
# Parity slice: first 50 tasks alphabetically
python benchmarks/harvey-lab/benchflow.py \
  --output-dir .cache/harvey-lab-parity \
  --harvey-root .cache/datasets/harveyai/harvey-labs \
  --split parity

# XLSX slice: first 25 tasks with .xlsx deliverables
python benchmarks/harvey-lab/benchflow.py \
  --output-dir .cache/harvey-lab-xlsx \
  --harvey-root .cache/datasets/harveyai/harvey-labs \
  --split xlsx

# Single practice area
python benchmarks/harvey-lab/benchflow.py \
  --output-dir .cache/harvey-lab-re \
  --harvey-root .cache/datasets/harveyai/harvey-labs \
  --split real-estate
```

Then point `bench eval create -t` at the generated directory.

### Using `--limit` and `--task-ids`

```bash
# First 10 tasks
python benchmarks/harvey-lab/benchflow.py \
  --output-dir .cache/harvey-lab-small \
  --harvey-root .cache/datasets/harveyai/harvey-labs \
  --limit 10

# Specific tasks
python benchmarks/harvey-lab/benchflow.py \
  --output-dir .cache/harvey-lab-pick \
  --harvey-root .cache/datasets/harveyai/harvey-labs \
  --task-ids "corporate-ma/analyze-cim-deal-teaser/scenario-01,real-estate/draft-construction-contract"
```

### Using `bench eval create` for a single task

```bash
bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks/edit-pdf \
  -a gemini \
  -m gemini-3.1-flash-lite-preview \
  -e docker
```

---

## Choosing an agent

Any registered BenchFlow agent works with adapted benchmarks. List them:

```bash
bench agent list
```

Common choices:

| Agent | Key | Auth |
|-------|-----|------|
| Gemini | `gemini` | `GEMINI_API_KEY` or host login |
| Claude Code | `claude-agent-acp` (alias: `claude`) | `ANTHROPIC_API_KEY` or host login |
| Codex | `codex-acp` (alias: `codex`) | `OPENAI_API_KEY` or host login |
| OpenHands | `openhands` (alias: `oh`) | `LLM_API_KEY` |
| Harvey LAB harness | `harvey-lab-harness` (alias: `harvey-lab`) | Provider key matching model |

The **Harvey LAB harness** agent is special — it runs Harvey LAB's own agent loop
(6 tools, system prompt) inside BenchFlow's sandbox. Use it for parity testing
(same agent on both original and converted tasks).

---

## Choosing a backend

| Backend | Flag | Best for |
|---------|------|----------|
| Docker | `-e docker` | Local development, small runs (≤10 tasks) |
| Daytona | `-e daytona` | Cloud runs with concurrency (needs `DAYTONA_API_KEY`) |
| Modal | `-e modal` | Serverless, high concurrency (needs Modal auth) |

For large-scale runs (100+ tasks), use Daytona or Modal with high concurrency:

```bash
bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks \
  -a gemini -m gemini-3.1-flash-lite-preview \
  -e daytona -c 64
```

---

## Reading results

Results land under `jobs/<job-name>/<trial-name>/`:

```
jobs/
└── harvey-lab-gemini-2026-05-06/
    ├── corporate-ma-review-data-room-red-flag-review/
    │   ├── result.json          # verifier output (reward, passed criteria)
    │   └── trajectory/
    │       └── acp_trajectory.jsonl  # full agent trace
    ├── real-estate-extract-psa-key-terms-scenario-01/
    │   ├── result.json
    │   └── trajectory/
    └── ...
```

The `result.json` contains:
```json
{
  "rewards": {"reward": 0.48},
  "passed": true,
  "verifier_output": "..."
}
```

List evaluations:
```bash
bench eval list jobs/
```

---

## Running parity validation

Before trusting results from an adapted benchmark, you can re-validate parity
(verify the conversion preserves benchmark semantics):

```bash
# Structural parity — checks all generated tasks have correct files/metadata
python benchmarks/harvey-lab/parity_test.py --mode full \
  --harvey-root .cache/datasets/harveyai/harvey-labs

# Eval parity — runs the verifier on synthetic output
GEMINI_API_KEY=... python benchmarks/harvey-lab/parity_test.py --mode eval-parity

# Side-by-side parity — compares original vs adapted prompts through same judge
GEMINI_API_KEY=... python benchmarks/harvey-lab/parity_test.py --mode side-by-side
```

Recorded parity results are in `parity_experiment.json` and `benchmark.yaml`.

---

## Running the SkillsBench E2E matrix

BenchFlow's own ENG-6 E2E check is file-driven through the normal eval CLI:

```bash
bench eval create -f tasks/skillsbench-e2e/e2e.yaml --dry-run
```

The dry run validates the configured 9-task × all-agent matrix and writes a
planned output bundle without creating sandboxes.

The live run is intentionally gated and should not run on every commit:

```bash
export BENCHFLOW_RUN_SKILLSBENCH_E2E=1
export DAYTONA_API_KEY=...
export GEMINI_API_KEY=...
bench eval create -f tasks/skillsbench-e2e/e2e.yaml
```

It runs the selected SkillsBench tasks across all registered BenchFlow agents
with `gemini-3.1-flash-lite-preview`, Daytona backend, concurrency 30, and no
skills. Agents that cannot run that Gemini model are recorded as findings rather
than hidden.

Each run writes:

- `matrix_config.json`
- `matrix_summary.json`
- `artifact_audit.json`
- `parity_report.json`
- `audit_findings.json`
- `findings.md`
- `audit_agent_prompt.md`

To re-run deterministic audits on an existing output directory:

```bash
python benchmarks/scripts/skillsbench_e2e_audit.py jobs/skillsbench-e2e/<run-id>
```

The config also supports an optional post-processing audit agent. Set
`audit.audit_agent.enabled: true` in `tasks/skillsbench-e2e/e2e.yaml` to create
an internal audit task from `audit/trajectory-result-auditor.md`; BenchFlow will
run the configured audit agent after deterministic scripts finish and write
`audit_agent_result.json`.

---

## YAML config reference

Job configs use the two-field `source` pattern to reference remote benchmark repos:

```yaml
# benchmarks/skillsbench-claude-glm51.yaml — direct from remote repo
source:
  repo: benchflow-ai/skillsbench   # GitHub repo (org/repo)
  path: tasks                      # subpath within the repo
  ref: main                        # branch/tag (optional)
agent: claude-agent-acp            # agent from registry
model: zai/glm-5.1                 # model ID
environment: daytona               # backend
concurrency: 8                     # parallel tasks
```

For benchmarks that require conversion (like Harvey LAB), use `tasks_dir` pointing
at the converted output:

```yaml
# benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml
# (conversion handled by run_harvey_lab.py)
agent: gemini
model: gemini/gemini-3.1-flash-lite-preview
environment: docker
concurrency: 4
```

You can also use `tasks_dir:` for local paths:

```yaml
tasks_dir: ./my-local-tasks        # local path (no download)
agent: gemini
model: gemini/gemini-3.1-flash-lite-preview
```

All fields from [CLI reference](./reference/cli.md#yaml-config-format) apply:
`source`, `tasks_dir`, `agent`, `model`, `environment`, `concurrency`,
`sandbox_setup_timeout`, `skills_dir`, `agent_env`, `max_retries`.

---

## Adding a new benchmark

See the [Benchmark Conversion Guide](../benchmarks/CONVERT.md) for the 9-step
process to convert a new benchmark into BenchFlow format. Harvey LAB
(`benchmarks/harvey-lab/`) is the reference implementation.
