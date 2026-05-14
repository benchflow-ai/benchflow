# Benchmark Conversion Guide

How to convert a new benchmark into BenchFlow format.

## Overview

Converting a benchmark produces:
- **`benchmarks/<name>/`** in the benchflow repo — converter code, parity tests, metadata
- **`datasets/<name>/`** in the benchmarks repo — ready-to-run task directories
- **HuggingFace upload** — parity experiments + task metadata

ProgramBench (`benchmarks/programbench/`) is the reference implementation.

## Steps

### 1. Understand the benchmark

Clone the benchmark repo and study its structure:
- What format are tasks in? (JSON, YAML, directories, etc.)
- What does each task contain? (instructions, documents, test cases, rubrics)
- How is evaluation done? (unit tests, script, LLM-as-judge, human eval)
- Are there oracle solutions?
- How many tasks are there? What categories/tags exist?

### 2. Write the converter (`benchflow.py`)

Create `benchmarks/<name>/benchflow.py` that maps the source format to BenchFlow task format.

Each generated task directory must contain:
```
<task-id>/
├── task.toml          # metadata: name, author, difficulty, category, tags, timeouts
├── instruction.md     # what the agent should do
├── environment/
│   └── Dockerfile     # container setup + input files
└── tests/
    ├── test.sh        # verifier entry point
    └── evaluate.py    # evaluation logic (if LLM-as-judge)
```

Optional: `solution/solve.sh` (oracle solution, if available).

The converter must accept CLI flags:
```
--output-dir DIR    # where to write generated tasks
--limit N           # cap number of tasks
--overwrite         # regenerate existing tasks
--task-ids IDS      # comma-separated specific task IDs
```

Key conventions:
- Use `string.Template.safe_substitute()` for any prompt templates (prevents injection)
- Sanitize task IDs to lowercase-hyphenated form
- Copy source documents into `environment/` for Docker COPY
- Set appropriate timeouts based on task complexity

### 3. Run structural parity

Verify every generated task has:
- All required files present
- Valid `task.toml` with correct metadata
- `instruction.md` non-empty
- `Dockerfile` builds
- Criteria count matches source (for rubric-based benchmarks)

```bash
python benchmarks/<name>/parity_test.py --mode full
```

### 4. Run eval parity

For LLM-as-judge benchmarks: run the evaluation pipeline on synthetic/dummy output
to confirm the judge produces valid verdicts (pass/fail with reasoning).

For unit-test benchmarks: run the tests on a known-good solution to confirm they pass.

```bash
python benchmarks/<name>/parity_test.py --mode eval-parity
```

### 5. Run side-by-side parity

The core validation: run the **original** evaluation prompt/script AND the
**converted** BenchFlow evaluation on identical agent output. Compare per-criterion
verdicts.

For LLM-as-judge: both prompts go through the same judge model on the same output.
For script-based: both scripts run on the same solution files.

Target: 100% agreement on a representative sample (≥5 tasks across categories).

```bash
python benchmarks/<name>/parity_test.py --mode side-by-side
```

### 6. Record results

Save parity experiment results to `parity_experiment.json`:
```json
{
  "experiment": "side-by-side-parity",
  "judge_model": "...",
  "tasks": [
    {
      "task_id": "...",
      "n_criteria": 5,
      "criteria_results": [
        {
          "criterion_id": "C-001",
          "criterion_title": "...",
          "original_verdict": "pass",
          "adapted_verdict": "pass",
          "agreement": true
        }
      ]
    }
  ]
}
```

### 7. Generate `benchmark.yaml`

Standard descriptor:
```yaml
name: <name>
description: "..."
url: <source-repo-url>
author: BenchFlow

tasks:
  count: <N>
  categories: <N>
  tags: [...]

conversion:
  script: benchflow.py
  source_format: <format>
  has_oracle_solutions: <true/false>

verification:
  method: <llm-as-judge|unit-test|script>
  judge_model: <model>  # if LLM-as-judge
  reward: <proportional|binary>

parity:
  structural:
    tasks_tested: <N>
    passed: <N>
  eval_pipeline:
    tasks_tested: <N>
    passed: <N>
  side_by_side:
    criteria_compared: <N>
    agreed: <N>
    agreement_rate: <float>
```

### 8. Create runner (`run_<name>.py`)

Script that:
1. Downloads/clones the source benchmark
2. Runs the converter
3. Runs the benchmark via BenchFlow `Job`

### 9. Publish

1. **benchflow repo**: PR with `benchmarks/<name>/` (converter + parity + metadata)
2. **benchmarks repo**: PR with `datasets/<name>/tasks/` (converted task dirs) + `datasets/<name>/parity/`
3. **HuggingFace**: Upload parity experiments + task metadata to `benchflow/benchmarks`

## File Checklist

```
benchmarks/<name>/
├── benchflow.py              # converter (required)
├── parity_test.py            # parity validation (required)
├── parity_experiment.json    # side-by-side results (required)
├── benchmark.yaml            # standard descriptor (required)
├── run_<name>.py             # runner (required)
├── <name>.yaml               # BenchFlow job config (optional)
└── README.md                 # documentation (required)
```
