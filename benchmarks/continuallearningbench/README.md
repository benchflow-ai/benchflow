# ContinualLearningBench Adapter

BenchFlow adapter for [ContinualLearningBench](https://github.com/pgasawa/continual-learning-bench).

ContinualLearningBench evaluates how well AI agents learn from past environment interactions across sequential task instances. Unlike most benchmarks that test single-shot performance, ContinualLearningBench measures **continual learning** — whether agents improve over time.

## Tasks

| Task | Instances | Difficulty | Description |
|------|-----------|------------|-------------|
| `exploitable_poker` | 120 (default schedule) | medium | Play poker against exploitable opponents, learn their strategies |
| `database_exploration` | 30 | medium | Answer questions about unknown SQLite databases, reduce queries over time |
| `cohort_studies` | 18 | hard | Medical cohort survival estimation across sequential clinical studies |

## Design

Each ContinualLearningBench task becomes **one BenchFlow task**. The agent runs the full sequential evaluation inside a Docker container using the ContinualLearningBench harness. This preserves the continual learning aspect — instance order matters and agents are expected to improve.

### Key Metrics

- **Reward**: Per-instance score (0-1), task-specific
- **Gain**: Reward minus stateless baseline — measures actual learning improvement
- **Score**: Mean reward across all instances

## Usage

### Generate task directories

```bash
python benchmarks/continuallearningbench/benchflow.py \
    --continuallearningbench-dir /path/to/continual-learning-bench \
    --output-dir /tmp/continuallearningbench-tasks
```

Native `task.md` output is available with the same source data:

```bash
python benchmarks/continuallearningbench/benchflow.py \
    --continuallearningbench-dir /path/to/continual-learning-bench \
    --output-dir /tmp/continuallearningbench-tasks-task-md \
    --task-format task-md
```

### Run parity tests

```bash
# Structural checks
python benchmarks/continuallearningbench/parity_test.py --output-dir /tmp/continuallearningbench-tasks --mode structural

# Eval parity (tests evaluate.py with synthetic results)
python benchmarks/continuallearningbench/parity_test.py --output-dir /tmp/continuallearningbench-tasks --mode eval

# Both
python benchmarks/continuallearningbench/parity_test.py --output-dir /tmp/continuallearningbench-tasks --mode all

# Native task.md output
python benchmarks/continuallearningbench/parity_test.py \
    --output-dir /tmp/continuallearningbench-tasks-task-md \
    --mode all \
    --task-format task-md
```

### Quick start (clone + generate + test)

```bash
python benchmarks/continuallearningbench/run_continuallearningbench.py
```

## CLI Flags

### benchflow.py

| Flag | Description |
|------|-------------|
| `--continuallearningbench-dir DIR` | Path to cloned ContinualLearningBench repo (required) |
| `--output-dir DIR` | Where to write generated tasks (required) |
| `--limit N` | Cap number of tasks |
| `--overwrite` | Regenerate existing tasks |
| `--task-ids IDS` | Comma-separated: `exploitable_poker,database_exploration,cohort_studies` |
| `--task-format FORMAT` | `legacy` for split layout or `task-md` for native packages |

### parity_test.py

| Flag | Description |
|------|-------------|
| `--output-dir DIR` | Directory with generated tasks (required) |
| `--mode MODE` | `structural`, `eval`, `live`, `e2e`, or `all` |
| `--task-format FORMAT` | Validate `legacy` or native `task-md` layout |

## task.md Layout

Native output writes each task as:

```text
continuallearningbench-<task>/
├── task.md
├── environment/
│   ├── Dockerfile
│   ├── run_task.py
│   └── schedule.json
├── verifier/
│   ├── test.sh
│   ├── evaluate.py
│   ├── verifier.md
│   └── rubrics/verifier.md
└── oracle/
    └── README.md
```

There is no static `oracle/solve.sh` because the benchmark's ground truth is the
original sequential environment and reward function. `oracle/README.md`
records that evidence, and the verifier reads `/opt/results.json` produced by
the ContinualLearningBench driver.

## Differences from Other Adapters

1. **Few tasks**: Only 3 BenchFlow task directories (one per ContinualLearningBench task type), unlike ProgramBench (many per-instance tasks)
2. **Interactive**: The agent interacts with the ContinualLearningBench harness inside the container, not just producing a single output
3. **Sequential**: Instance order matters — the benchmark measures learning over time
4. **Heavier containers**: Each task needs the full ContinualLearningBench installation + task-specific dependencies

## Source

- **Repo**: https://github.com/pgasawa/continual-learning-bench
- **Website**: https://continual-learning-bench.com
- **Authors**: Parth Asawa, Chris Glaze, Gabe Orlanski et al.
