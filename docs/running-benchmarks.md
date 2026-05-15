# Running Adapted Benchmarks

How to run benchmarks that have been converted to BenchFlow format.

BenchFlow ships with adapted benchmarks under `benchmarks/<name>/`. Each benchmark
includes a converter, parity tests, metadata, and one or more YAML job configs.
This guide covers how to run them — from a single task to a full evaluation sweep.

> **Working inside the benchflow repo?** Use `uv run bench` instead of `bench`
> to run the CLI from your local editable install.

---

## Available benchmarks

| Benchmark | Tasks | Verification | Config |
|-----------|-------|--------------|--------|
| [Harvey LAB](https://github.com/harveyai/harvey-labs) | 1,251 | LLM-as-judge (per-criterion) | `benchmarks/harvey-lab/` |
| [ProgramBench](https://programbench.com) | 201 | Deterministic unit tests | `benchmarks/programbench/` |
| [SkillsBench](https://github.com/benchflow-ai/skillsbench) | 94+ | Unit tests | `benchmarks/skillsbench-*.yaml` |

Each adapted benchmark includes:
- **`benchflow.py`** — converter: raw benchmark → BenchFlow task format
- **`benchmark.yaml`** — metadata descriptor (task count, categories, verification method, parity results)
- **`<name>-*.yaml`** — job configs for different agents/models
- **`parity_test.py`** — parity validation suite
- **`parity_experiment.json`** — recorded parity results

---

## Quick start

### Option 1: YAML config (`bench eval create -f`)

The simplest path. Point at a YAML config that specifies the benchmark source,
agent, and model:

```bash
GEMINI_API_KEY=... bench eval create -f benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml
GEMINI_API_KEY=... bench eval create -f benchmarks/programbench/programbench-gemini-flash-lite.yaml
bench eval create -f benchmarks/skillsbench-claude-glm51.yaml
```

The config handles everything — downloads/generates tasks, resolves the task path,
and runs the evaluation.

### Option 2: CLI flags

Use CLI flags for ad-hoc runs without a config file:

```bash
bench eval create --source-repo harveyai/harvey-labs --source-path tasks -a gemini -m gemini-3.1-flash-lite-preview -e docker -c 4
bench run --source-repo benchflow-ai/skillsbench --source-path tasks/edit-pdf -a gemini -m gemini-3.1-flash-lite-preview
bench run benchmarks/programbench/tasks/abishekvashok__cmatrix.5c082c6 -a gemini -m gemini-3.1-flash-lite-preview -b docker
bench eval create --source-repo benchflow-ai/skillsbench --source-path tasks -a claude-agent-acp -m anthropic/claude-sonnet-4-6 -e daytona -c 32
```

### Option 3: Python API

For programmatic use, custom pipelines, or integration with other tools:

```python
import asyncio
from benchflow.job import Job

async def main():
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

### Single task

```bash
bench run --source-repo benchflow-ai/skillsbench --source-path tasks/edit-pdf -a gemini -m gemini-3.1-flash-lite-preview
bench run benchmarks/programbench/tasks/abishekvashok__cmatrix.5c082c6 -a gemini -m gemini-3.1-flash-lite-preview -b docker
bench run .cache/harvey-lab-tasks/corporate-ma-review-data-room-red-flag-review -a gemini -m gemini-3.1-flash-lite-preview -b docker
```

### Batch with a tasks directory

Point `bench eval create -t` at a directory containing only the tasks you want:

```bash
bench eval create -t benchmarks/programbench/tasks -a gemini -m gemini-3.1-flash-lite-preview -e docker -c 4
```

### Using `--source-path` for remote subsets

```bash
bench eval create --source-repo benchflow-ai/skillsbench --source-path tasks/edit-pdf -a gemini -m gemini-3.1-flash-lite-preview -e docker
```

---

## Running ProgramBench

201 program-reconstruction tasks across 7 languages (C, Rust, Go, C++, Java, Haskell, Bash).
Tasks are **generated** at runtime from the ProgramBench repo's metadata.

### Prerequisites

- Docker (images are linux/amd64 only — use a Linux x86_64 machine)
- ~20GB disk for Docker images
- Internet access for HuggingFace test blob downloads during verification

### Run all tasks

```bash
bench eval create -f benchmarks/programbench/programbench-gemini-flash-lite.yaml
```

### Run a single task

```bash
bench run benchmarks/programbench/tasks/abishekvashok__cmatrix.5c082c6 -a gemini -m gemini-3.1-flash-lite-preview -b docker
```

### Oracle verification

Verify a task is solvable using the gold solution (original source at commit):

```bash
bench run benchmarks/programbench/tasks/abishekvashok__cmatrix.5c082c6 -a oracle -b docker
```

### Validate a task directory

```bash
bench tasks check benchmarks/programbench/tasks/abishekvashok__cmatrix.5c082c6
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
bench eval create --source-repo benchflow-ai/skillsbench --source-path tasks -a gemini -m gemini-3.1-flash-lite-preview -e daytona -c 64
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

Parity validation is a **developer/maintainer workflow** for verifying that an
adapter preserves benchmark semantics. These scripts live under each benchmark's
directory:

```bash
uv run python benchmarks/harvey-lab/parity_test.py --mode full --harvey-root .cache/datasets/harveyai/harvey-labs
GEMINI_API_KEY=... uv run python benchmarks/harvey-lab/parity_test.py --mode eval-parity
GEMINI_API_KEY=... uv run python benchmarks/harvey-lab/parity_test.py --mode side-by-side
```

Recorded parity results are in `parity_experiment.json` and `benchmark.yaml`.

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
# (conversion handled automatically)
agent: gemini
model: gemini/gemini-3.1-flash-lite-preview
environment: docker
concurrency: 4
```

For generated benchmarks (like ProgramBench), use `tasks_dir` pointing at the
generated output:

```yaml
# benchmarks/programbench/programbench-gemini-flash-lite.yaml
tasks_dir: benchmarks/programbench/tasks
agent: gemini
model: gemini-3.1-flash-lite-preview
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
(`benchmarks/harvey-lab/`) and ProgramBench (`benchmarks/programbench/`) are
reference implementations.
