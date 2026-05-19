# BenchFlow × HILBench

Generate BenchFlow task directories from
[HILBench](https://github.com/scale-ai/hil-bench) (Human-in-the-Loop
Benchmark) SWE baseline instances.

HILBench evaluates AI coding agents on software engineering tasks that
may require clarification from humans.  The benchmark includes 200
tasks total (100 SWE + 100 SQL) across 3 modes (baseline, full-info,
ask-human).  This adapter converts the **SWE baseline** subset — 100
tasks across 4 repositories.

Paper: https://arxiv.org/abs/2604.09408

## Quick start

```bash
# Generate all 100 SWE baseline tasks
python benchmarks/hilbench/benchflow.py \
    --output-dir benchmarks/hilbench/tasks

# Generate a subset
python benchmarks/hilbench/benchflow.py \
    --output-dir benchmarks/hilbench/tasks \
    --limit 10

# Generate specific tasks
python benchmarks/hilbench/benchflow.py \
    --output-dir benchmarks/hilbench/tasks \
    --task-ids public_swe_0,public_swe_1

# Run structural parity tests
python benchmarks/hilbench/parity_test.py \
    --tasks-dir benchmarks/hilbench/tasks

# Run via BenchFlow (requires a config YAML)
python benchmarks/hilbench/run_hilbench.py path/to/config.yaml
```

## Dataset

The tasks are loaded from HuggingFace: [`ScaleAI/hil-bench`](https://huggingface.co/datasets/ScaleAI/hil-bench).

| Field | Description |
|-------|-------------|
| `task_id` | Unique identifier (e.g., `public_swe_0`) |
| `task_type` | `swe` or `sql` |
| `repo_or_db_name` | Repository name (e.g., `ansible/ansible`) |
| `repo_or_db_download_link` | HuggingFace bucket link to Docker image tarball |
| `problem` | Problem statement for the agent |
| `test_patch` | Diff that adds/modifies test files |
| `tests_to_pass` | List of pytest test IDs the solution must pass |
| `test_files` | Test files referenced by the patch |
| `ground_truth_answer` | Gold patch (the correct solution) |
| `blocker_registry` | List of ambiguities requiring human clarification |

## Repositories

| Repository | Tasks |
|------------|-------|
| `ansible/ansible` | 40 |
| `protonmail/webclients` | 29 |
| `navidrome/navidrome` | 16 |
| `flipt-io/flipt` | 15 |

## Format comparison: HILBench → BenchFlow

### Directory structure

| HILBench (HuggingFace) | BenchFlow (generated) |
|---|---|
| `task_id` | `task.toml` name field `hilbench/<sanitized-id>` |
| `problem` | `instruction.md` |
| `repo_or_db_download_link` (Docker image tarball) | `environment/Dockerfile` (uses pre-built image as `BASE_IMAGE`) |
| `test_patch` + `tests_to_pass` | `tests/test_patch.diff` + `tests/verify.py` |
| `ground_truth_answer` (gold patch) | Not used for verification (could be oracle) |

### Evaluation pipeline

| Step | HILBench | BenchFlow |
|---|---|---|
| **Environment** | Docker image loaded from HF bucket tarball | Dockerfile uses pre-built image via `BASE_IMAGE` build arg |
| **Agent submission** | Agent modifies code in the repo | Agent works in `/workspace/` |
| **Test execution** | Apply test_patch, run tests_to_pass | Same — verify.py applies patch, runs pytest per test |
| **Scoring** | Count tests_to_pass that pass | Same — reward = passed / total |

### Modes

HILBench defines three modes per task:

| Mode | Description | Supported |
|------|-------------|-----------|
| **baseline** | Agent gets only the problem statement | Yes |
| **full_info** | Agent gets problem + all blocker resolutions | Not yet |
| **ask_human** | Agent can ask clarifying questions | Not yet |

This adapter implements **baseline mode only**.

## Docker image setup

Each HILBench SWE task ships a pre-built Docker image tarball on
HuggingFace bucket `ScaleAI/hil-bench-swe-images`.  These images contain
the repository at the correct commit plus the SWEAP test harness.  The
task metadata dataset is `ScaleAI/hil-bench`; image links use the bucket
URI form `hf://buckets/ScaleAI/hil-bench-swe-images/images/<uid>.tar.zst`.

The runner resolves bucket URIs to
`https://huggingface.co/buckets/ScaleAI/hil-bench-swe-images/resolve/images/<uid>.tar.zst`
and downloads them directly.  If HuggingFace later gates a bucket or object,
set `HF_TOKEN` or `HUGGINGFACE_TOKEN`; public objects do not require a token.

The runner (`run_hilbench.py`) handles downloading and loading images
automatically.  Each task's `Dockerfile` references a predictable
`hilbench-base:<task_id>` image tag, and the runner retags loaded images
to that name before the task image is built.  The HILBench base images keep
the repository under `/app` and expose `/testbed` as a symlink, so generated
Dockerfiles also map `/workspace` to the repository path expected by BenchFlow
instructions and verifiers.

## Notes

- The `blocker_registry` field contains structured information about
  ambiguities in the task — these are used in `full_info` and `ask_human`
  modes but ignored in baseline mode.
- Eval parity and side-by-side parity have not yet been performed.
  These require running agents against the actual HILBench Docker images.
