# Onramp · ProgramBench

Adapter that converts the [ProgramBench](https://programbench.com/) benchmark into BenchFlow tasks.

## What ProgramBench is

200 program-rebuild tasks: each gives the agent a **compiled binary** (execute-only) and **documentation**, and asks it to produce a from-scratch implementation that reproduces the binary's externally-observable behaviour. Tests are AI-generated behavioural test suites (~248 K cases total) executed against the agent's `./executable`. Languages span Rust (107), Go (46), C (33), C++ (12), Haskell (1), Java (1).

The upstream repo is [`facebookresearch/ProgramBench`](https://github.com/facebookresearch/ProgramBench); per-instance docker images live at `programbench/<id>:task_cleanroom`; per-branch test blobs live in the [`programbench/ProgramBench-Tests`](https://huggingface.co/datasets/programbench/ProgramBench-Tests) HuggingFace dataset.

## What this adapter does

For each upstream instance it emits a BenchFlow task directory:

```
.ref/programbench-bf/<instance_id>/
├── task.toml              # name + metadata + per-difficulty resources
├── instruction.md         # rebuild-from-binary spec, /workspace/executable + /workspace/docs
├── environment/Dockerfile # FROM programbench/<sanitized_id>:task_cleanroom + verifier deps
└── tests/
    ├── test.sh            # task-agnostic verifier (BF_INSTANCE_ID is set in the Dockerfile)
    └── tests.json         # sidecar copy of the upstream tests.json
```

The verifier mirrors `programbench eval`'s pipeline inside one BenchFlow sandbox:
1. Snapshot pristine `/workspace`.
2. Wipe and stage the agent's `/app/` into `/workspace/`.
3. Seed a deterministic git repo, then run `compile.sh` -> `/workspace/executable`.
4. Stash the executable and record its sha256.
5. For each active test branch from `tests.json`: restore `/workspace`, restore the executable (re-checking the hash), unpack the branch tarball from HuggingFace, run `eval/run.sh`, parse `eval/results.xml`.
6. `reward = total_passed / total_expected`.

Branches and tests with `ignored: true` upstream are excluded, matching `programbench info`'s scoring.

## Generate the converted dataset

```bash
# From the repo root
uv run python -m onramp.programbench.main --output-dir .ref/programbench-bf
```

The first run clones `facebookresearch/ProgramBench` into `.ref/programbench/` to read the per-instance `task.yaml` + `tests.json`. Subsequent runs reuse the cache.

Useful flags:

```bash
# Smoke-test a handful of tasks
python -m onramp.programbench.main --output-dir .ref/programbench-bf --limit 5

# Convert specific instances
python -m onramp.programbench.main --output-dir .ref/programbench-bf \
    --task-ids abishekvashok__cmatrix.5c082c6 jq__jq.cff5336

# Regenerate after touching the templates
python -m onramp.programbench.main --output-dir .ref/programbench-bf --overwrite
```

Validate the generated set:

```bash
for t in .ref/programbench-bf/*/; do bench tasks check "$t" || exit 1; done
```

## Run the converted dataset

The default config in [`run_programbench.yaml`](./run_programbench.yaml) targets Gemini 3.1 Flash Lite and a local Docker backend. For a real run:

```bash
uv run python -m benchflow.job onramp/programbench/run_programbench.yaml \
    --override agent=claude-agent-acp model=claude-haiku-4-5-20251001 environment=daytona
```

The cleanroom images are large (gigabytes per task) — Daytona is the practical choice for full-set runs.

## Parity check

[`parity.py`](./parity.py) re-runs the same submission archive through both pipelines and reports `passed/total` deltas:

```bash
# Drives the fixture instance shipped with ProgramBench (no HF blobs needed).
python -m onramp.programbench.parity \
    --upstream-repo .ref/programbench \
    --limit 1
```

For a real instance, supply your own submission tarball (the artifact your agent produced):

```bash
python -m onramp.programbench.parity \
    --upstream-repo .ref/programbench \
    --instance-id abishekvashok__cmatrix.5c082c6 \
    --submission /path/to/submission.tar.gz
```

Live, model-driven parity (the case the user cares about) — generate submissions with Gemini through BenchFlow, then score those same submissions through the upstream evaluator:

```bash
export GEMINI_API_KEY=...
uv run python -m benchflow.job onramp/programbench/run_programbench.yaml \
    --override jobs_dir=../jobs/programbench-parity-subset \
    --tasks-glob '.ref/programbench-bf/abishekvashok__cmatrix.5c082c6'

# Pull each submission out of the BenchFlow trial and feed it back through the upstream eval.
for inst in $(ls ../jobs/programbench-parity-subset); do
    python -m onramp.programbench.parity \
        --upstream-repo .ref/programbench \
        --instance-id "$inst" \
        --submission "../jobs/programbench-parity-subset/$inst/submission.tar.gz"
done
```

Results land in [`parity_experiment.json`](./parity_experiment.json).

## Resource sizing

Per-task limits in `task.toml` are picked from the upstream `difficulty` field:

| Difficulty | CPUs | Memory | Storage | Agent timeout | Verifier timeout |
|---|---:|---:|---:|---:|---:|
| easy     | 2 | 4 GB  | 20 GB | 30 min  | 1 hr |
| medium   | 4 | 8 GB  | 40 GB | 1 hr    | 1 hr |
| hard     | 4 | 16 GB | 80 GB | 2 hr    | 1 hr |
| unrated  | 4 | 8 GB  | 40 GB | 1 hr    | 1 hr |

Override per-task by editing the generated `task.toml`, or change defaults in [`adapter.py`](./adapter.py).

## Notes

- Cleanroom Docker images are `linux/amd64` only — runs on Apple Silicon need QEMU and will be slow.
- The verifier streams test blobs from HuggingFace at run time. Set `BF_BLOB_CACHE` to a host-mounted path if you want to share blobs across runs.
- When the upstream `tests.json` marks a branch or test `ignored: true`, the verifier excludes it from both numerator and denominator — same as `programbench info`'s headline score.
