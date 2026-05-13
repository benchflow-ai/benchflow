# Running Adapted Benchmarks

How to run benchmarks that have been converted to BenchFlow format.

BenchFlow ships with adapted benchmarks under `benchmarks/<name>/`. Each benchmark
includes a converter, parity tests, metadata, and one or more YAML job configs.
This guide covers how to run them — from a single task to a full evaluation sweep.

---

## Available benchmarks

| Benchmark | Tasks | Verification | Source |
|-----------|-------|--------------|--------|
| [Harvey LAB](https://github.com/harveyai/harvey-labs) | 1,251 | LLM-as-judge (per-criterion) | `benchmarks/harvey-lab/` |
| [SkillsBench](https://github.com/benchflow-ai/skillsbench) | — | Unit tests | `benchmarks/run_skillsbench.py` |
| [Terminal-Bench 2](https://github.com/harbor-framework/terminal-bench-2) | — | Script | `benchmarks/run_tb2.py` |

Each adapted benchmark includes:
- **`benchflow.py`** — converter: raw benchmark → BenchFlow task format
- **`benchmark.yaml`** — metadata descriptor (task count, categories, verification method, parity results)
- **`run_<name>.py`** — one-command runner (downloads, converts, runs)
- **`<name>-*.yaml`** — job configs for different agents/models
- **`parity_test.py`** — parity validation suite
- **`parity_experiment.json`** — recorded parity results

---

## Quick start

### Option 1: One-command runner

The simplest path. Downloads the benchmark, converts tasks, and runs the evaluation:

```bash
# Harvey LAB with Gemini (default config)
GEMINI_API_KEY=... python benchmarks/harvey-lab/run_harvey_lab.py

# Harvey LAB with a custom config
python benchmarks/harvey-lab/run_harvey_lab.py benchmarks/harvey-lab/harvey-lab-harness-parity.yaml

# SkillsBench
python benchmarks/run_skillsbench.py
```

The runner script:
1. Clones the source benchmark repo into `.ref/<name>/`
2. Runs the converter to produce BenchFlow task directories in `.ref/<name>-benchflow/`
3. Loads the YAML job config and runs via `Job.from_yaml()`
4. Prints the aggregate score

### Option 2: CLI (`bench eval create`)

Use the CLI for more control over agent, model, backend, and concurrency:

```bash
# Step 1: Download + convert tasks (one-time)
python benchmarks/harvey-lab/run_harvey_lab.py  # creates .ref/harvey-lab-benchflow/

# Step 2: Run with any agent/model/backend
bench eval create \
  -t .ref/harvey-lab-benchflow \
  -a gemini \
  -m gemini-3.1-flash-lite-preview \
  -e docker \
  -c 4

# Or with Claude Code
bench eval create \
  -t .ref/harvey-lab-benchflow \
  -a claude-agent-acp \
  -m anthropic/claude-sonnet-4-6 \
  -e daytona \
  -c 32
```

### Option 3: YAML config (`bench eval create -f`)

Write a YAML job config and run it:

```yaml
# my-harvey-lab-run.yaml
tasks_dir: .ref/harvey-lab-benchflow
agent: gemini
model: gemini/gemini-3.1-flash-lite-preview
environment: docker
concurrency: 4
```

```bash
bench eval create -f my-harvey-lab-run.yaml
```

### Option 4: Python API

For programmatic use, custom pipelines, or integration with other tools:

```python
import asyncio
from benchflow.job import Job
from benchflow.task_download import ensure_tasks

async def main():
    # Download raw tasks (cached after first run)
    ensure_tasks("harvey-lab")

    # Run from YAML config
    job = Job.from_yaml("benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml")
    result = await job.run()
    print(f"Score: {result.passed}/{result.total} ({result.score:.1%})")

asyncio.run(main())
```

For single-task runs:

```python
import benchflow as bf
from benchflow.trial import TrialConfig, Scene
from pathlib import Path

config = TrialConfig(
    task_path=Path(".ref/harvey-lab-benchflow/corporate-ma-review-data-room-red-flag-review"),
    scenes=[Scene.single(agent="gemini", model="gemini-3.1-flash-lite-preview")],
    environment="docker",
)
result = await bf.run(config)
print(result.rewards)
```

---

## Running a subset of tasks

### Using `--split` (at conversion time)

The converter supports named splits for generating subsets:

```bash
# Parity slice: first 50 tasks alphabetically
python benchmarks/harvey-lab/benchflow.py \
  --output-dir .ref/harvey-lab-parity \
  --harvey-root .ref/harvey-lab \
  --split parity

# XLSX slice: first 25 tasks with .xlsx deliverables
python benchmarks/harvey-lab/benchflow.py \
  --output-dir .ref/harvey-lab-xlsx \
  --harvey-root .ref/harvey-lab \
  --split xlsx

# Single practice area
python benchmarks/harvey-lab/benchflow.py \
  --output-dir .ref/harvey-lab-re \
  --harvey-root .ref/harvey-lab \
  --split real-estate
```

Then point `bench eval create -t` at the generated directory.

### Using `--limit` and `--task-ids`

```bash
# First 10 tasks
python benchmarks/harvey-lab/benchflow.py \
  --output-dir .ref/harvey-lab-small \
  --harvey-root .ref/harvey-lab \
  --limit 10

# Specific tasks
python benchmarks/harvey-lab/benchflow.py \
  --output-dir .ref/harvey-lab-pick \
  --harvey-root .ref/harvey-lab \
  --task-ids "corporate-ma/analyze-cim-deal-teaser/scenario-01,real-estate/draft-construction-contract"
```

### Using `bench run` for a single task

```bash
GEMINI_API_KEY=... bench run .ref/harvey-lab-benchflow/corporate-ma-review-data-room-red-flag-review \
  --agent gemini \
  --model gemini-3.1-flash-lite-preview \
  --backend docker
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
bench eval create -t .ref/harvey-lab-benchflow \
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
  --harvey-root .ref/harvey-lab

# Eval parity — runs the verifier on synthetic output
GEMINI_API_KEY=... python benchmarks/harvey-lab/parity_test.py --mode eval-parity

# Side-by-side parity — compares original vs adapted prompts through same judge
GEMINI_API_KEY=... python benchmarks/harvey-lab/parity_test.py --mode side-by-side
```

Recorded parity results are in `parity_experiment.json` and `benchmark.yaml`.

---

## YAML config reference

Job configs live alongside each benchmark:

```yaml
# benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml
tasks_dir: .ref/harvey-lab-benchflow       # converted task directory
agent: gemini                              # agent from registry
model: gemini/gemini-3.1-flash-lite-preview  # model ID
environment: docker                        # backend
concurrency: 4                             # parallel tasks
```

All fields from [CLI reference](./reference/cli.md#yaml-config-format) apply:
`tasks_dir`, `agent`, `model`, `environment`, `concurrency`, `sandbox_setup_timeout`,
`skills_dir`, `agent_env`, `max_retries`.

---

## Adding a new benchmark

See the [Benchmark Conversion Guide](../benchmarks/CONVERT.md) for the 9-step
process to convert a new benchmark into BenchFlow format. Harvey LAB
(`benchmarks/harvey-lab/`) is the reference implementation.
