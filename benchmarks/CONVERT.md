# Benchmark Conversion Guide

How to convert an external benchmark into BenchFlow tasks.

## Overview

BenchFlow's native authoring format is now `task.md`: one Markdown document with
YAML frontmatter for task config and a `## prompt` section for the agent-facing
instruction. The native task package vocabulary is:

```text
<task-id>/
├── task.md
├── environment/
│   └── Dockerfile
├── verifier/
│   ├── test.sh
│   └── verifier.md        # optional, for richer verifier strategies
└── oracle/
    └── solve.sh           # optional, if an oracle solution exists
```

BenchFlow still supports the legacy Harbor-compatible split layout:
`task.toml`, `instruction.md`, `tests/`, and `solution/`. Treat that layout as a
compatibility target or explicit `legacy` export, not as the default for new
converters. Use
[task.md adapter capabilities](../docs/task-md-adapter-capabilities.md) to
classify the conversion as lossless, native verifier package, protocol adapter,
wrapper-only Harbor export, or fail-closed.

Converting a benchmark produces:

- **`benchmarks/<name>/`** in this repo: converter code, parity tests, metadata.
- **`datasets/<name>/`** in the benchmarks repo: ready-to-run task directories.
- **HuggingFace upload**: parity experiments and task metadata.

ProgramBench (`benchmarks/programbench/`) is the current legacy reference
implementation. New converters should be task.md-first.

## Steps

### 1. Understand the benchmark

Clone the benchmark repo and study its structure:

- What format are tasks in? JSON, YAML, directories, database rows, traces?
- What does each task contain? Instructions, documents, test cases, rubrics?
- How is evaluation done? Unit tests, scripts, LLM-as-judge, human eval?
- Are oracle solutions available?
- How many tasks are there, and what categories/tags exist?
- Does the benchmark require a special runtime such as browser, desktop,
  multi-container services, simulator users, or hidden assets?

### 2. Write the converter

Create `benchmarks/<name>/benchflow.py` that maps the source benchmark into
BenchFlow task packages.

The converter should accept:

```bash
--output-dir DIR
--limit N
--overwrite
--task-ids IDS
--task-format task-md|legacy
```

If the converter is called from Python instead of the CLI, use the same
vocabulary: `task_format: Literal["task-md", "legacy"] = "task-md"`.
New converters should default to `task-md`. Existing migrated converters may
keep a `legacy` default while release lanes and downstream jobs still consume
the split layout, but `benchmark.yaml` must make that compatibility default
explicit with `default_task_format: legacy` and `task_formats: [legacy,
task-md]`.

For `task-md`, each generated task directory should contain:

```text
<task-id>/
├── task.md
├── environment/
│   └── Dockerfile
├── verifier/
│   ├── test.sh
│   ├── evaluate.py        # if needed
│   └── verifier.md        # optional but preferred for publication
└── oracle/
    └── solve.sh           # if available
```

For `legacy`, emit:

```text
<task-id>/
├── task.toml
├── instruction.md
├── environment/
├── tests/
└── solution/
```

Key conventions:

- Use structured writers (`tomli_w`, YAML, JSON) instead of ad hoc string
  interpolation for config data.
- Use `string.Template.safe_substitute()` for prompt/script templates that need
  interpolation.
- Sanitize task IDs to lowercase hyphenated path segments.
- Keep foreign benchmark fields under `source`, `metadata`, or a
  benchmark-owned namespace instead of adding root task config keys.
- Copy source documents into `environment/` for Docker build context.
- Set timeouts based on task complexity, not a single benchmark-wide constant.
- Verifier scripts should avoid hardcoded `/tests` paths. Prefer:

```bash
VERIFIER_DIR="${BENCHFLOW_VERIFIER_DIR:-/verifier}"
if [ ! -d "$VERIFIER_DIR" ] && [ -d /tests ]; then
  VERIFIER_DIR=/tests
fi
python3 "$VERIFIER_DIR/evaluate.py"
```

This makes the same verifier work for native `verifier/` tasks and legacy
`tests/` exports.

### 3. Run structural parity

Verify every generated task has:

- A valid entrypoint: `task.md` for native tasks, or `task.toml` plus
  `instruction.md` for legacy tasks.
- A non-empty prompt.
- A buildable `environment/Dockerfile`.
- A runnable verifier entrypoint.
- Correct native/legacy directory vocabulary for the selected format.
- Criteria/test counts matching the source benchmark.

Prefer parsing through BenchFlow task helpers (`Task`, `TaskPaths`,
`TaskDocument`) instead of hardcoding filenames in parity tests. If a parity
test is intentionally checking an export format, name that explicitly.

```bash
python benchmarks/<name>/parity_test.py --mode full
```

### 4. Run eval parity

For LLM-as-judge benchmarks, run the adapted evaluation pipeline on synthetic
or dummy output and assert the judge returns valid structured verdicts.

For unit-test or script-based benchmarks, run the verifier on a known-good
oracle solution and confirm it passes.

```bash
python benchmarks/<name>/parity_test.py --mode eval-parity
```

### 5. Run side-by-side parity

The core validation is unchanged by the task format: run the original
evaluation prompt/script and the adapted BenchFlow verifier on identical agent
output, then compare verdicts.

For LLM-as-judge, both prompts should use the same judge model and same output.
For script-based benchmarks, both scripts should run against the same solution
files.

Target: 100% agreement on a representative sample of at least five tasks across
categories.

```bash
python benchmarks/<name>/parity_test.py --mode side-by-side
```

### 6. Record results

Save side-by-side results to `parity_experiment.json`:

```json
{
  "experiment": "side-by-side-parity",
  "task_format": "task-md",
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
  default_task_format: task-md
  task_formats: [task-md, legacy]
  has_oracle_solutions: <true|false>

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

### 8. Create runner

Add `run_<name>.py` that:

1. Downloads or clones the source benchmark.
2. Runs the converter in `task-md` mode by default.
3. Optionally exports legacy layout for external compatibility.
4. Runs the benchmark via BenchFlow `Job`.

### 9. Publish

1. **benchflow repo**: PR with `benchmarks/<name>/` converter, parity tests, and
   metadata.
2. **benchmarks repo**: PR with `datasets/<name>/tasks/` and
   `datasets/<name>/parity/`.
3. **HuggingFace**: upload parity experiments and task metadata to
   `benchflow/benchmarks`.

## File Checklist

```text
benchmarks/<name>/
├── benchflow.py              # converter
├── parity_test.py            # structural/eval/side-by-side parity
├── parity_experiment.json    # side-by-side results
├── benchmark.yaml            # descriptor
├── run_<name>.py             # runner
├── <name>.yaml               # BenchFlow job config, optional
└── README.md                 # benchmark-specific docs
```

## Migration Checklist

For an existing legacy converter:

1. Add `output_format` or `--task-format`.
2. Render `task.md` by moving the current TOML config into YAML frontmatter and
   the current `instruction.md` body into `## prompt`.
3. Emit `verifier/` instead of `tests/` and `oracle/` instead of `solution/`
   when `output_format == "task-md"`.
4. Make verifier scripts path-portable between `/verifier` and `/tests`.
5. Update parity tests to use `TaskPaths` or to explicitly test both formats.
6. Keep legacy export until downstream Harbor/Pier users no longer need it.
