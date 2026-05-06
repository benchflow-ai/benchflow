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
    --output-dir benchmarks/programbench/tasks

# Or generate a subset
python -m benchmarks.programbench.main \
    --programbench-dir ~/programbench \
    --output-dir benchmarks/programbench/tasks \
    --task-ids jqlang__jq.b33a763 burntsushi__ripgrep.3b7fd44

# Run via BenchFlow
python benchmarks/run_programbench.py benchmarks/programbench-gemini-flash-lite.yaml
```

## Format comparison: ProgramBench → BenchFlow

### Directory structure

| ProgramBench (original) | BenchFlow (generated) |
|---|---|
| `task.yaml` (repository, commit, language, difficulty, eval_clean_hashes) | `task.toml` (name, metadata, timeouts, resources) |
| `tests.json` (per-branch test lists, ignored flags) | `tests/tests.json` (copied verbatim) |
| Docker image `programbench/<repo>:task_cleanroom` (pre-built on DockerHub) | `environment/Dockerfile` (FROM the same cleanroom image) |
| Agent produces `submission.tar.gz` (entire codebase) | Agent works directly in `/workspace` inside the container |
| `programbench eval <run_dir>` (CLI evaluation) | `tests/test.sh` → `tests/verify.py` (self-contained verifier) |
| Original source code at commit (gold answer) | `solution/solve.sh` (clones original repo at commit) |
| No instruction file — agent receives Docker image | `instruction.md` (explicit agent-facing instructions) |

### Evaluation pipeline

| Step | ProgramBench | BenchFlow |
|---|---|---|
| **Environment** | Agent uses `programbench/<repo>:task_cleanroom` Docker image directly | Dockerfile wraps the same cleanroom image, adds verifier dependencies |
| **Agent submission** | Agent produces `submission.tar.gz` extracted into container | Agent writes files directly in `/workspace` |
| **Compilation** | `compile.sh` → `./executable` | Same — `compile.sh` → `./executable` |
| **Anti-cheat** | Remove files matching `eval_clean_hashes` SHA-256 | Same — `verify.py` matches identical hashes |
| **Test blobs** | Downloaded from HuggingFace by `programbench eval` CLI | Downloaded from HuggingFace by `verify.py` inside container |
| **Test execution** | Per-branch: extract tar, patch timeout method, run `eval/run.sh`, parse JUnit XML | Same sequence in `verify.py` |
| **Scoring** | `EvaluationResult` JSON with per-test pass/fail | `reward.txt` with float 0.0–1.0 (passed / total) |
| **Oracle** | Original source at commit (`:task` image tag) | `solution/solve.sh` clones repo at commit, runs `compile.sh` |

### Key fields mapping

| ProgramBench `task.yaml` | BenchFlow `task.toml` |
|---|---|
| `repository` | Used in `instruction.md` and `solution/solve.sh` |
| `commit` | Used in `solution/solve.sh` for oracle checkout |
| `language` | `[metadata] tags` |
| `difficulty` | `[metadata] difficulty` + determines `[agent] timeout_sec` and `[verifier] timeout_sec` |
| `eval_clean_hashes` | Embedded in `verify.py` as `CLEAN_HASHES` |

### What changes vs. what stays the same

| Aspect | Changed? | Details |
|---|---|---|
| Docker base image | No | Same `programbench/<repo>:task_cleanroom` images from DockerHub |
| Test archives | No | Same HuggingFace blobs (`programbench/ProgramBench-Tests`) |
| Test execution | No | Same pytest + JUnit XML parsing + timeout patching |
| Anti-cheat hashing | No | Same SHA-256 hash removal |
| Scoring formula | No | Same passed/total partial credit |
| Agent instructions | Yes | Explicit `instruction.md` instead of implicit Docker image |
| Evaluation trigger | Yes | BenchFlow verifier (test.sh) instead of `programbench eval` CLI |
| Result format | Yes | `reward.txt` float instead of `EvaluationResult` JSON |
| Oracle format | Yes | `solve.sh` script instead of `:task` Docker image tag |

## How it works

1. **Task generation** reads ProgramBench's `task.yaml` and `tests.json`
   per instance and emits a standard BenchFlow task directory:

   ```
   <instance_id>/
   ├── task.toml           # timeouts, metadata, resources
   ├── instruction.md      # agent-facing instructions
   ├── environment/
   │   └── Dockerfile      # FROM programbench/<image>:task_cleanroom
   ├── solution/
   │   └── solve.sh        # oracle: clones original source at commit
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
| `tasks_dir` | Path to generated tasks (`benchmarks/programbench/tasks`) |
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
