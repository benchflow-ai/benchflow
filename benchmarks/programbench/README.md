# BenchFlow × ProgramBench

Generate BenchFlow task directories from
[ProgramBench](https://programbench.com) instances.

ProgramBench evaluates whether AI agents can reverse-engineer black-box
software: given only a compiled binary and its documentation, agents must
architect and implement a complete codebase that reproduces the original
program's behavior. The benchmark includes 200 tasks spanning Rust, Go,
C, C++, Haskell, and Java — from small CLI tools like `jq` and `ripgrep`
to massive projects like FFmpeg, SQLite, and the PHP interpreter.

## Quick start

```bash
# Generate all 200 tasks (requires programbench repo)
python -m benchmarks.programbench.main \
    --programbench-dir ~/programbench \
    --output-dir .ref/programbench/tasks

# Or generate a subset
python -m benchmarks.programbench.main \
    --programbench-dir ~/programbench \
    --output-dir .ref/programbench/tasks \
    --task-ids jqlang__jq.b33a763 burntsushi__ripgrep.3b7fd44

# Run via BenchFlow
python benchmarks/run_programbench.py benchmarks/programbench-gemini-flash-lite.yaml
```

## How it works

1. **Task generation** reads ProgramBench's `task.yaml` and `tests.json`
   per instance and emits a standard BenchFlow task directory:

   ```
   <instance_id>/
   ├── task.toml           # timeouts, metadata, resources
   ├── instruction.md      # agent-facing instructions
   ├── environment/
   │   └── Dockerfile      # FROM programbench/<image>:task_cleanroom
   └── tests/
       ├── test.sh          # verifier entry point
       ├── verify.py        # downloads test blobs, runs pytest, writes reward
       └── tests.json       # per-branch test manifest
   ```

2. **At runtime** the agent works inside the ProgramBench cleanroom
   Docker image, which contains the compiled binary and docs but no
   source code. The agent must produce `compile.sh` that builds a new
   `executable`.

3. **Verification** (`verify.py`) compiles the submission, removes files
   that match the original binary's hash (anti-cheat), downloads
   behavioral test archives from HuggingFace, runs each test branch's
   pytest suite, and writes a partial-credit reward based on the fraction
   of tests passed.

## Configuration

See `benchmarks/programbench-gemini-flash-lite.yaml` for the default
config. Key fields:

| Field | Description |
|-------|-------------|
| `tasks_dir` | Path to generated tasks (`.ref/programbench/tasks`) |
| `agent` | Agent name (e.g. `gemini`, `claude-agent-acp`) |
| `model` | Model identifier |
| `environment` | `docker` or `daytona` |
| `concurrency` | Parallel task count |

## Notes

- All ProgramBench Docker images are **linux/amd64 only**. Use a Linux
  x86_64 machine.
- The agent **must not** have internet access during inference (enforced
  via `allow_internet = false` in `task.toml`).
- Test blob archives are downloaded on demand from HuggingFace during
  verification.
