# Running Benchmarks

How to run existing benchmark suites with BenchFlow.

---

## Available benchmarks

| Benchmark | Tasks | Description | Source |
|-----------|-------|-------------|--------|
| `terminal-bench-2` | ~50 | Terminal-based programming tasks | Cloned from git |
| `skillsbench` | — | Skill evaluation tasks | Cloned from git |
| `programbench` | 200 | Program reconstruction from compiled binaries | Generated from [ProgramBench](https://programbench.com) |

---

## Quick start

Every benchmark follows the same pattern: download/generate tasks, then run via CLI or YAML config.

### 1. CLI — single task

```bash
# Terminal-Bench 2
bench run .ref/terminal-bench-2/regex-log \
  --agent gemini --model gemini-3.1-flash-lite-preview --backend docker

# ProgramBench (generate first, then run one task)
python -m benchmarks.programbench.main \
  --task-ids abishekvashok__cmatrix.5c082c6
bench run benchmarks/programbench/tasks/abishekvashok__cmatrix.5c082c6 \
  --agent gemini --model gemini-3.1-flash-lite-preview --backend docker
```

### 2. YAML config — batch run

Create a YAML config file:

```yaml
# my-benchmark-run.yaml
tasks_dir: benchmarks/programbench/tasks   # directory of task subdirectories
jobs_dir: jobs/my-run                       # where results land
agent: gemini
model: gemini-3.1-flash-lite-preview
environment: docker                         # or daytona for cloud
concurrency: 4                              # parallel tasks
max_retries: 1
```

Then run it:

```bash
python benchmarks/run_programbench.py my-benchmark-run.yaml
```

Or use the Python API directly:

```python
import asyncio
from benchflow.job import Job
from benchflow.task_download import ensure_tasks

async def main():
    ensure_tasks("programbench")  # downloads/generates tasks if needed
    job = Job.from_yaml("my-benchmark-run.yaml")
    result = await job.run()
    print(f"Score: {result.passed}/{result.total} ({result.score:.1%})")

asyncio.run(main())
```

### 3. Python API — single task

```python
import benchflow as bf
from benchflow.trial import TrialConfig, Scene
from pathlib import Path

config = TrialConfig(
    task_path=Path("benchmarks/programbench/tasks/abishekvashok__cmatrix.5c082c6"),
    scenes=[Scene.single(agent="gemini", model="gemini-3.1-flash-lite-preview")],
    environment="docker",
)
result = await bf.run(config)
print(result.rewards)  # {'reward': 0.37}
```

---

## Running Terminal-Bench 2

Pre-built tasks cloned from git. No generation step needed.

```bash
# Download tasks (one-time)
python -c "from benchflow.task_download import ensure_tasks; ensure_tasks('terminal-bench-2')"

# Run with provided config
python benchmarks/run_tb2.py benchmarks/tb2-gemini-baseline.yaml
```

**Config:** `benchmarks/tb2-gemini-baseline.yaml`

---

## Running ProgramBench

200 program-reconstruction tasks. Tasks are **generated** at runtime from the ProgramBench repo's metadata.

### Prerequisites

- Docker (images are linux/amd64 only — use a Linux x86_64 machine)
- ~20GB disk for Docker images
- Internet access for HuggingFace test blob downloads during verification

### Generate tasks

```bash
# Generate all 200 tasks
python -m benchmarks.programbench.main \
  --output-dir benchmarks/programbench/tasks

# Generate a subset
python -m benchmarks.programbench.main \
  --output-dir benchmarks/programbench/tasks \
  --task-ids abishekvashok__cmatrix.5c082c6 ajeetdsouza__zoxide.67ca1bc

# Or let ensure_tasks() handle it automatically (from a dev checkout, prefix with `uv run`)
python -c "from benchflow.task_download import ensure_tasks; ensure_tasks('programbench')"
```

Tasks are written to `benchmarks/programbench/tasks/` (gitignored, cached across runs).

### Run

```bash
# Single task
bench run benchmarks/programbench/tasks/abishekvashok__cmatrix.5c082c6 \
  --agent gemini --model gemini-3.1-flash-lite-preview --backend docker

# Batch run with config
python benchmarks/run_programbench.py benchmarks/programbench-gemini-flash-lite.yaml
```

### Oracle verification

Verify a task is solvable using the gold solution:

```bash
bench run benchmarks/programbench/tasks/abishekvashok__cmatrix.5c082c6 \
  --agent oracle --backend docker
```

The oracle clones the original source at the specified commit and runs `compile.sh`.

---

## Config reference

All YAML configs support these fields:

| Field | Required | Description |
|-------|----------|-------------|
| `tasks_dir` | yes | Path to directory containing task subdirectories |
| `jobs_dir` | yes | Output directory for results |
| `agent` | yes | Agent name (`gemini`, `claude-agent-acp`, `codex-acp`, `oracle`) |
| `model` | yes | Model identifier (e.g. `gemini-3.1-flash-lite-preview`) |
| `environment` | yes | `docker` (local) or `daytona` (cloud) |
| `concurrency` | no | Parallel task count (default 1) |
| `max_retries` | no | Retry failed tasks (default 0) |
| `sandbox_user` | no | Override sandbox user (default: auto) |

---

## Results

Results land under `<jobs_dir>/<trial>/`:

| File | Contents |
|------|----------|
| `result.json` | Verifier output: reward, metadata, errors |
| `trajectory/acp_trajectory.jsonl` | Full agent trace: prompts, tool calls, responses |

---

## Adding a new benchmark

See [`task-authoring.md`](./task-authoring.md) for creating individual tasks. For converting an existing benchmark suite:

1. **Understand the original benchmark**: identify instructions, environments, tests, and scoring
2. **Write a generator** (`benchmarks/<name>/benchflow.py`) that reads original metadata and produces BenchFlow task directories
3. **Register in `task_download.py`** (if tasks should auto-generate on first use)
4. **Validate**: `bench tasks check benchmarks/<name>/tasks/<task_id>/` for every generated task
5. **Run oracle parity**: verify `--agent oracle` produces expected scores
6. **Run agent parity**: run the same agent on both original and adapted benchmark, compare rewards
7. **Document**: add a README with format comparison tables and parity results

See `benchmarks/programbench/` for a complete example of this workflow.
