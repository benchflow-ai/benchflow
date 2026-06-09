# OpaqueToolsBench — BFCL Adapter

BenchFlow adapter for the **BFCL (Berkeley Function Calling Leaderboard)**
environment from [OpaqueToolsBench](https://github.com/shallinan1/OpaqueToolsBench).

## What is OpaqueToolsBench?

OpaqueToolsBench ([paper](https://arxiv.org/abs/2602.15197)) studies whether
LLM agents can recover the meaning of *opacified* tools — tools whose names,
descriptions, and parameters have been deliberately obscured. It includes three
environments: BFCL, Chess, and BrowseCompPlus.

This adapter covers the **BFCL environment** only.

## BFCL Environment

BFCL tests general function-calling ability: given a natural-language query and
a set of tool (function) definitions, the agent must produce the correct
function call(s).

| Category                       | Tasks | Description                    |
|--------------------------------|-------|--------------------------------|
| `executable_simple`            |   100 | Single function call           |
| `executable_multiple_function` |    50 | Multiple function calls        |
| **Total**                      | **150** |                              |

### Evaluation

Evaluation is **deterministic** — no LLM judge required. The verifier parses
the agent's function-call JSON output and compares it against ground-truth
function-call strings using AST-based matching:

- Function name must match exactly.
- All ground-truth parameters must be present with matching values.
- Float comparison uses 5% relative tolerance.
- Reward is binary: **1.0** (all calls correct) or **0.0**.

## Quick Start

### 1. Generate Tasks

```bash
# Clone OpaqueToolsBench if you haven't already
git clone https://github.com/shallinan1/OpaqueToolsBench.git /tmp/OpaqueToolsBench

# Generate all 150 tasks
python benchmarks/opaquetoolsbench/benchflow.py \
    --opaquetoolsbench-dir /tmp/OpaqueToolsBench \
    --output-dir /tmp/opaquetoolsbench-tasks

# Or generate a subset
python benchmarks/opaquetoolsbench/benchflow.py \
    --opaquetoolsbench-dir /tmp/OpaqueToolsBench \
    --output-dir /tmp/opaquetoolsbench-tasks \
    --limit 10

# Or generate native task.md packages
python benchmarks/opaquetoolsbench/benchflow.py \
    --opaquetoolsbench-dir /tmp/OpaqueToolsBench \
    --output-dir /tmp/opaquetoolsbench-task-md \
    --task-format task-md \
    --limit 10
```

### 2. Validate Structure

```bash
python benchmarks/opaquetoolsbench/parity_test.py \
    --tasks-dir /tmp/opaquetoolsbench-tasks

python benchmarks/opaquetoolsbench/parity_test.py \
    --tasks-dir /tmp/opaquetoolsbench-task-md \
    --task-format task-md
```

### 3. Run Benchmark

```bash
python benchmarks/opaquetoolsbench/run_opaquetoolsbench.py

# Or pass an explicit config
python benchmarks/opaquetoolsbench/run_opaquetoolsbench.py \
    benchmarks/opaquetoolsbench/opaquetoolsbench-gemini-flash-lite.yaml

# Run from native task.md conversions
python benchmarks/opaquetoolsbench/run_opaquetoolsbench.py \
    --task-format task-md
```

## Generated Task Structure

By default each generated task directory uses the legacy BenchFlow split layout:

```
<task-id>/
├── task.toml                  # BenchFlow task metadata
├── instruction.md             # NL query + function descriptions for the agent
├── environment/
│   └── Dockerfile             # Python 3.13 slim environment
└── tests/
    ├── test.sh                # Runs evaluate.py, writes reward
    ├── evaluate.py            # AST-based function-call comparison
    └── ground_truth.json      # Expected function call(s)
```

With `--task-format task-md`, each generated task directory uses the native
authoring layout:

```
<task-id>/
├── task.md                    # Unified task config and prompt
├── environment/
│   └── Dockerfile             # Python 3.13 slim environment
├── verifier/
│   ├── verifier.md            # Verifier package contract
│   ├── rubrics/verifier.md    # Deterministic scoring rubric
│   ├── test.sh                # Runs evaluate.py, writes reward artifacts
│   ├── evaluate.py            # AST-based function-call comparison
│   └── ground_truth.json      # Expected function call(s)
└── oracle/
    └── solve.sh               # Writes the ground-truth response JSON
```

## Agent Response Format

The agent must write a JSON array of function-call objects to
`/app/output/response.json`:

```json
[
  {
    "function": "calc_binomial_probability",
    "args": {
      "n": 20,
      "k": 5,
      "p": 0.6
    }
  }
]
```

## CLI Flags

| Flag                     | Description                                |
|--------------------------|--------------------------------------------|
| `--opaquetoolsbench-dir` | Path to OpaqueToolsBench repo checkout     |
| `--output-dir`           | Where to write generated task directories  |
| `--limit N`              | Cap number of tasks to generate            |
| `--overwrite`            | Regenerate existing task directories       |
| `--task-ids ID1,ID2,...` | Comma-separated list of specific task IDs  |
| `--task-format FORMAT`   | `legacy` or `task-md` output layout        |
