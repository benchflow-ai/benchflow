# CLBench — Continual Learning Bench Adapter

BenchFlow adapter for [Continual Learning Bench](https://github.com/pgasawa/continual-learning-bench) (CLBench).

CLBench evaluates how well AI agents learn from past environment interactions across sequential task instances. Unlike most benchmarks that test single-shot performance, CLBench measures **continual learning** — whether agents improve over time.

## Tasks

| Task | Instances | Difficulty | Description |
|------|-----------|------------|-------------|
| `exploitable_poker` | 120 (default schedule) | medium | Play poker against exploitable opponents, learn their strategies |
| `database_exploration` | 30 | medium | Answer questions about unknown SQLite databases, reduce queries over time |
| `cohort_studies` | 18 | hard | Medical cohort survival estimation across sequential clinical studies |

## Design

Each CLBench task becomes **one BenchFlow task**. The agent runs the full sequential evaluation inside a Docker container using the CLBench harness. This preserves the continual learning aspect — instance order matters and agents are expected to improve.

### Key Metrics

- **Reward**: Per-instance score (0-1), task-specific
- **Gain**: Reward minus stateless baseline — measures actual learning improvement
- **Score**: Mean reward across all instances

## Usage

### Generate task directories

```bash
python benchmarks/clbench/benchflow.py \
    --clbench-dir /path/to/continual-learning-bench \
    --output-dir /tmp/clbench-tasks
```

### Run parity tests

```bash
# Structural checks
python benchmarks/clbench/parity_test.py --output-dir /tmp/clbench-tasks --mode structural

# Eval parity (tests evaluate.py with synthetic results)
python benchmarks/clbench/parity_test.py --output-dir /tmp/clbench-tasks --mode eval

# Both
python benchmarks/clbench/parity_test.py --output-dir /tmp/clbench-tasks --mode all
```

### Quick start (clone + generate + test)

```bash
python benchmarks/clbench/run_clbench.py
```

## CLI Flags

### benchflow.py

| Flag | Description |
|------|-------------|
| `--clbench-dir DIR` | Path to cloned CLBench repo (required) |
| `--output-dir DIR` | Where to write generated tasks (required) |
| `--limit N` | Cap number of tasks |
| `--overwrite` | Regenerate existing tasks |
| `--task-ids IDS` | Comma-separated: `exploitable_poker,database_exploration,cohort_studies` |

### parity_test.py

| Flag | Description |
|------|-------------|
| `--output-dir DIR` | Directory with generated tasks (required) |
| `--mode MODE` | `structural`, `eval`, or `all` |

## Differences from Other Adapters

1. **Few tasks**: Only 3 BenchFlow task directories (one per CLBench task type), unlike ProgramBench (many per-instance tasks)
2. **Interactive**: The agent interacts with the CLBench harness inside the container, not just producing a single output
3. **Sequential**: Instance order matters — the benchmark measures learning over time
4. **Heavier containers**: Each task needs the full CLBench installation + task-specific dependencies

## Source

- **Repo**: https://github.com/pgasawa/continual-learning-bench
- **Website**: https://continual-learning-bench.com
- **Authors**: Parth Asawa, Chris Glaze, Gabe Orlanski et al.
