# Running Adapted Benchmarks

How to run benchmarks that have been converted into Harbor-format tasks for BenchFlow.

BenchFlow ships with adapted benchmarks under `benchmarks/<name>/`. Each benchmark
includes a converter, parity tests, metadata, and one or more YAML job configs.
This guide covers how to run them — from a single task to a full evaluation sweep.

> [!NOTE]
> BenchFlow is providing first-party support for PrimeIntellect Verifiers and OpenReward Standard.

> **Working inside the benchflow repo?** Use `uv run bench` instead of `bench`
> to run the CLI from your local editable install.

---

## Available benchmarks

| Benchmark | Tasks | Verification | Config |
|-----------|-------|--------------|--------|
| [Harvey LAB](https://github.com/harveyai/harvey-labs) | 1,251 | LLM-as-judge (per-criterion) | `benchmarks/harvey-lab/` |
| [ProgramBench](https://programbench.com) | 201 | Deterministic unit tests | `benchmarks/programbench/` |
| [SkillsBench](https://github.com/benchflow-ai/skillsbench) | 94+ | Unit tests | `--source-repo benchflow-ai/skillsbench --source-path tasks` |

Each adapted benchmark includes:
- **`benchflow.py`** — converter for the raw benchmark source
- **`benchmark.yaml`** — metadata descriptor (task count, categories, verification method, parity results)
- **`<name>-*.yaml`** — job configs for different agents/models
- **`parity_test.py`** — parity validation suite
- **`parity_experiment.json`** — recorded parity results

### Environment-plane benchmarks

Stateful, multi-service benchmarks integrate differently: instead of a
converter they ship an `environment.toml` **manifest** and run on the
[Environment plane](./environment-plane.md). Two are onboarded:

| Benchmark | Topology | Manifest |
|-----------|----------|----------|
| **ClawsBench** | mock Gmail/Slack/Calendar/Docs/Drive, framework-started | `benchmarks/clawsbench/environment.toml` |
| **chi-bench** | ~25k-LOC healthcare simulator, image-owned lifecycle | `benchmarks/chi-bench/environment.toml` |

See [Running a benchmark with an Environment manifest](#running-a-benchmark-with-an-environment-manifest) below.

---

## Quick start

### Option 1: YAML config (`bench eval create --config`)

The simplest path. Point at a YAML config that specifies the benchmark source,
agent, and model:

```bash
GEMINI_API_KEY=... bench eval create --config benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml
GEMINI_API_KEY=... bench eval create --config benchmarks/programbench/programbench-gemini-flash-lite.yaml
bench eval create --config benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml
```

The config handles everything — downloads/generates tasks, resolves the task path,
and runs the evaluation.

### Option 2: CLI flags

Use CLI flags for ad-hoc runs without a config file:

```bash
# Harvey LAB — single pre-converted task
bench eval create \
  --source-repo benchflow-ai/benchmarks \
  --source-path datasets/harvey-lab/tasks/corporate-ma-analyze-cim-deal-teaser-scenario-01 \
  --agent gemini --model gemini-3.1-flash-lite-preview --sandbox docker

# Harvey LAB harness adapter smoke test.
# Requires GEMINI_API_KEY for the agent and ANTHROPIC_API_KEY for the verifier.
uv run bench eval create \
  --source-repo benchflow-ai/benchmarks \
  --source-path datasets/harvey-lab/tasks/corporate-ma-analyze-cim-deal-teaser-scenario-01 \
  --agent harvey-lab-harness \
  --model gemini-3.1-flash-lite-preview \
  --sandbox docker \
  --concurrency 1 \
  --jobs-dir jobs/smoke-test/harvey-harness

# Harvey LAB — all pre-converted tasks
bench eval create \
  --source-repo benchflow-ai/benchmarks \
  --source-path datasets/harvey-lab/tasks \
  --agent gemini --model gemini-3.1-flash-lite-preview --sandbox docker --concurrency 4

# SkillsBench
bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks/edit-pdf \
  --agent gemini --model gemini-3.1-flash-lite-preview

# ProgramBench — single task (tasks are generated at runtime by the converter;
# see "Running ProgramBench" below for the generation step)
bench eval create \
  --tasks-dir benchmarks/programbench/tasks/abishekvashok__cmatrix.5c082c6 \
  --agent gemini --model gemini-3.1-flash-lite-preview --sandbox docker

# Claude Code on Daytona
bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks \
  --agent claude-agent-acp --model anthropic/claude-sonnet-4-6 --sandbox daytona --concurrency 32
```

> **Note:** Harvey LAB task names in `benchflow-ai/benchmarks` are flattened with
> hyphens (e.g. `corporate-ma-analyze-cim-deal-teaser-scenario-01`), not nested
> paths like the original repo (`corporate-ma/analyze-cim-deal-teaser/scenario-01`).

### Option 3: Python API

For programmatic use, custom pipelines, or integration with other tools:

```python
import asyncio
from benchflow.evaluation import Evaluation

async def main():
    job = Evaluation.from_yaml("benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml")
    result = await job.run()
    print(f"Score: {result.passed}/{result.total} ({result.score:.1%})")

asyncio.run(main())
```

For single-task runs:

```python
import benchflow as bf
from benchflow import RolloutConfig, Scene
from benchflow._utils.benchmark_repos import resolve_source

task_path = resolve_source("benchflow-ai/skillsbench", path="tasks/edit-pdf")

config = RolloutConfig(
    task_path=task_path,
    scenes=[Scene.single(agent="gemini", model="gemini-3.1-flash-lite-preview")],
    environment="docker",
)
result = await bf.run(config)
print(result.rewards)
```

---

## Versioned dataset runs (`--dataset`)

For runs whose results should be attributable to a published, immutable
dataset version (leaderboards, papers, release evidence), resolve the task
set from a dataset registry instead of pointing at a directory or branch:

```bash
# Resolve skillsbench@1.1 from the registry, verify every task's content
# digest against the pinned snapshot, then run.
bench eval create -d skillsbench@1.1 \
  --agent claude-agent-acp --model claude-haiku-4-5-20251001

# Versions are immutable, so the version is always explicit — there is no
# floating "latest". --include/--exclude filter the registry roster.
bench eval create -d skillsbench@1.1 --include xlsx-recover-data ...
```

A registry (`registry.json` at a dataset repo's root — see skillsbench's
[`docs/dataset-versioning.md`](https://github.com/benchflow-ai/skillsbench/blob/main/docs/dataset-versioning.md))
pins each dataset version to an exact `git_commit_id` and per-task sha256
content digests. Resolution clones the pinned commit into
`.cache/datasets`, materializes an immutable per-commit snapshot,
recomputes every task's digest, and **fails before running anything** on
any mismatch. Snapshot directories that are not part of the registry entry
are excluded from the run. The entry's `bench_version` range is checked
against the installed benchflow; running outside the range the dataset was
validated against prints a warning — results may not be comparable with
published runs.

The registry is fetched from the skillsbench repo by default; point
`--registry` at another URL or a local `registry.json` to override.

Every `result.json`/`config.json` is stamped with `dataset_name`,
`dataset_version`, and the per-task `task_digest` (`summary.json` carries
the name/version), so downstream tooling can group results by
`dataset@version`. `--tasks-dir` stays as the visibly distinct dev mode:
its artifacts carry no dataset fields — but they still stamp a
live-computed `task_digest`, so even dev trajectories remain attributable
to exact task content. `bench tasks digest <task-dir>` prints the same
digest for any task directory.

---

## Running a subset of tasks

### Single task

```bash
bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks/edit-pdf \
  --agent gemini --model gemini-3.1-flash-lite-preview

bench eval create \
  --tasks-dir benchmarks/programbench/tasks/abishekvashok__cmatrix.5c082c6 \
  --agent gemini --model gemini-3.1-flash-lite-preview --sandbox docker

bench eval create \
  --tasks-dir .cache/harvey-lab-tasks/corporate-ma-review-data-room-red-flag-review \
  --agent gemini --model gemini-3.1-flash-lite-preview --sandbox docker
```

### Batch with a tasks directory

Point `bench eval create --tasks-dir` at a directory containing only the tasks you want:

```bash
bench eval create --tasks-dir benchmarks/programbench/tasks \
  --agent gemini --model gemini-3.1-flash-lite-preview --sandbox docker --concurrency 4
```

### Using `--source-path` for remote subsets

```bash
# SkillsBench single task
bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks/edit-pdf \
  --agent gemini --model gemini-3.1-flash-lite-preview --sandbox docker

# Harvey LAB single task (pre-converted)
bench eval create \
  --source-repo benchflow-ai/benchmarks \
  --source-path datasets/harvey-lab/tasks/corporate-ma-analyze-cim-deal-teaser-scenario-01 \
  --agent gemini --model gemini-3.1-flash-lite-preview --sandbox docker
```

---

## Running ProgramBench

201 program-reconstruction tasks across 7 languages (C, Rust, Go, C++, Java, Haskell, Bash).
Tasks are **generated** at runtime from the ProgramBench repo's metadata —
`benchmarks/programbench/tasks/` is not checked into this repo and must be
produced first.

### Prerequisites

- Docker (images are linux/amd64 only — use a Linux x86_64 machine)
- ~20GB disk for Docker images
- Internet access for HuggingFace test blob downloads during verification
- A local clone of [`programbench`](https://programbench.com) (passed via
  `--programbench-dir` to the generator)

### Generate the tasks

```bash
# All 200 tasks
python -m benchmarks.programbench.main \
    --programbench-dir ~/programbench \
    --output-dir benchmarks/programbench/tasks

# Or a single task
python -m benchmarks.programbench.main \
    --programbench-dir ~/programbench \
    --output-dir benchmarks/programbench/tasks \
    --task-ids abishekvashok__cmatrix.5c082c6
```

### Run all tasks

```bash
bench eval create --config benchmarks/programbench/programbench-gemini-flash-lite.yaml
```

### Run a single task (after generation)

```bash
bench eval create \
  --tasks-dir benchmarks/programbench/tasks/abishekvashok__cmatrix.5c082c6 \
  --agent gemini --model gemini-3.1-flash-lite-preview --sandbox docker
```

### Oracle verification

Verify a task is solvable using the gold solution (original source at commit):

```bash
bench eval create \
  --tasks-dir benchmarks/programbench/tasks/abishekvashok__cmatrix.5c082c6 \
  --agent oracle --sandbox docker
```

### Validate a task directory

```bash
bench tasks check benchmarks/programbench/tasks/abishekvashok__cmatrix.5c082c6
```

---

## Choosing an agent

Any registered BenchFlow agent works with adapted benchmarks. List them:

```bash
bench agent list
```

Common choices:

| Agent | Key | Auth |
|-------|-----|------|
| Gemini | `gemini` | `GEMINI_API_KEY` or host login |
| Claude Code | `claude-agent-acp` (alias: `claude`) | `ANTHROPIC_API_KEY` or host login |
| Codex | `codex-acp` (alias: `codex`) | `OPENAI_API_KEY`, `CODEX_API_KEY`, `CODEX_ACCESS_TOKEN`, or host login |
| OpenHands | `openhands` (alias: `oh`) | `LLM_API_KEY` |
| Harvey LAB harness | `harvey-lab-harness` (alias: `harvey-lab`) | Provider key matching model |

The auth column shows each agent's native/default credentials. Provider-prefixed
models can use provider-specific credentials instead; for example, Azure
Foundry models use `AZURE_API_KEY` plus `AZURE_API_ENDPOINT` with prefixes such
as `azure-foundry-openai/gpt-5.5` or
`azure-foundry-anthropic/claude-opus-4-5`.

Any agent can also be run via [ACPX](https://acpx.sh/) by prefixing with `acpx/`:

```bash
bench eval create --tasks-dir tasks/edit-pdf --agent acpx/gemini --model gemini-3.1-flash-lite-preview --sandbox daytona
```

ACPX is a headless ACP client that adds persistent sessions and crash recovery.
The underlying agent's install, env vars, credentials, and skill paths are all preserved.

The **Harvey LAB harness** agent is special — it runs Harvey LAB's own agent loop
(6 tools, system prompt) inside BenchFlow's sandbox. Use it for parity testing
(same agent on both original and converted tasks).

---

## Choosing a sandbox

| Sandbox | Flag | Best for |
|---------|------|----------|
| Docker | `--sandbox docker` | Local development, small runs (≤10 tasks) |
| Daytona | `--sandbox daytona` | Cloud runs with concurrency (needs `DAYTONA_API_KEY`) |
| Modal | `--sandbox modal` | Serverless, high concurrency (needs Modal auth) |

For large-scale runs (100+ tasks), use Daytona or Modal with high concurrency:

```bash
bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks \
  --agent gemini --model gemini-3.1-flash-lite-preview --sandbox daytona --concurrency 64
```

> **Daytona has a 10 GB-per-sandbox hard cap.** Tasks with heavy images (large
> HuggingFace model snapshots, Playwright, LaTeX/marker — e.g.
> `latex-formula-extraction`) overflow during bootstrap (`No space left on
> device`) or hang at "Sandbox user agent ready" with no trajectory. Run those on
> `--sandbox docker` (host disk, no cap); keep Daytona for lighter tasks.

---

## SkillsBench skill-toggle matrix (Opus-4.8 + Gemini) on Daytona

A self-contained recipe for the four-cell matrix of
**{Opus-4.8 via Bedrock, Gemini-3.5-flash} × {with-skills, without-skills}**,
agent `openhands`, sandbox `daytona`. Each cell produces a complete trajectory
(`trajectory/{acp,llm}_trajectory.jsonl`) plus a verifier reward — but treat a
cell as done only after the audit in [Verifying the batch](#verifying-the-batch)
passes.

### Setup (once per shell)

```bash
# 1. Run the CLI from a benchflow checkout. BenchFlow starts LiteLLM as the
#    provider gateway for Bedrock/Gemini/Azure/etc. Inside the repo, use `uv run bench`.
cd /path/to/benchflow

# 2. A local skillsbench clone, so --skills-dir can point at a task's bundled skills
git clone https://github.com/benchflow-ai/skillsbench
export SKILLSBENCH=$PWD/skillsbench

# 3. Credentials — verify each LIVE first. A dead key shows up only as
#    `openhands ACP error -32603` on the first model call, never a clean auth error:
#      Bedrock: a `converse` call with `Authorization: Bearer $AWS_BEARER_TOKEN_BEDROCK` -> 200
#      Gemini:  `.../v1beta/models/<model>:generateContent?key=$GEMINI_API_KEY`           -> 200
export AWS_BEARER_TOKEN_BEDROCK=...
export AWS_REGION=us-west-2 AWS_DEFAULT_REGION=us-west-2
export GEMINI_API_KEY=...

# 4. MAX thinking for Opus-4.8 (opt-in). WITHOUT this the run uses the agent's
#    default effort = adaptive-thinking `high`, NOT max. LiteLLM receives this
#    env var on both Daytona and Docker.
export BENCHFLOW_BEDROCK_THINKING_EFFORT=max

# 5. Strip stale external gateway vars; BenchFlow will generate its own LiteLLM config:
unset LLM_BASE_URL LLM_API_KEY OPENAI_BASE_URL BENCHFLOW_PROVIDER_BASE_URL \
      BENCHFLOW_PROVIDER_API_KEY LITELLM_BASE_URL LITELLM_API_KEY
```

> Pick a **light** task — Daytona caps each sandbox at 10 GB (see the note above).
> `citation-check` is a good default; heavy tasks need `--sandbox docker`. Note that
> MAX effort makes each Opus turn much slower (deep server-side reasoning — a
> `citation-check` cell took ~10–15 min at `max` vs ~3 min at the default effort).

### Run the four cells

```bash
TASK=citation-check
COMMON="--tasks-dir $SKILLSBENCH/tasks --include $TASK --agent openhands \
  --sandbox daytona --concurrency 1 --usage-tracking required --agent-idle-timeout none"

# (1) Opus-4.8 (MAX) — with skills
bench eval create $COMMON --model aws-bedrock/us.anthropic.claude-opus-4-8 \
  --skill-mode with-skill --jobs-dir jobs/opus-skill

# (2) Opus-4.8 (MAX) — without skills
bench eval create $COMMON --model aws-bedrock/us.anthropic.claude-opus-4-8 \
  --skill-mode no-skill --jobs-dir jobs/opus-noskill

# (3) Gemini-3.5-flash — with skills
bench eval create $COMMON --model gemini-3.5-flash --agent-env LLM_CACHING_PROMPT=false \
  --skill-mode with-skill --jobs-dir jobs/gemini-skill

# (4) Gemini-3.5-flash — without skills
bench eval create $COMMON --model gemini-3.5-flash --agent-env LLM_CACHING_PROMPT=false \
  --skill-mode no-skill --jobs-dir jobs/gemini-noskill
```

`BENCHFLOW_BEDROCK_THINKING_EFFORT=max` is what makes the two Opus cells actually
run at MAX. LiteLLM writes the provider call metadata to
`trajectory/llm_trajectory.jsonl`; confirm the adaptive thinking effort there.

| Model (`--model`) | Skills | Cell-specific flags |
|-------------------|--------|---------------------|
| `aws-bedrock/us.anthropic.claude-opus-4-8` | with | `--skill-mode with-skill` |
| `aws-bedrock/us.anthropic.claude-opus-4-8` | without | `--skill-mode no-skill` |
| `gemini-3.5-flash` | with | `--agent-env LLM_CACHING_PROMPT=false --skill-mode with-skill` |
| `gemini-3.5-flash` | without | `--agent-env LLM_CACHING_PROMPT=false --skill-mode no-skill` |

### Verifying the batch

A finished command is **not** a healthy trial. After each batch, audit the
trajectories with the **`benchflow-experiment-review`** skill (repo copy at
`.claude/skills/benchflow-experiment-review`; see the Experiment-guidance notes in
`AGENTS.md`). A trial counts as healthy only when **every** check passes: complete
trajectory + metadata (timing, token usage, tool usage), correct
pass/fail/timeout status, verifier isolation (verifier starts after the agent
exits), no reward hacking, and the right skill posture — with-skills cells must
show the task skill loaded (`task_skills_loading: 1`), without-skills cells must
not (`task_skills_loading: 0`, and the task skill absent from the trajectory;
generic openhands built-ins such as `.agents/skills` / `invoke_skill` appear in
*every* run and are **not** leakage).

Quick smoke checks before the full audit (per jobs-dir):

```bash
J=jobs/opus-skill
find $J -name rewards.jsonl -exec tail -1 {} \;                                  # reward
grep -ho '"usage_source": "[a-z_]*"' $(find $J -name result.json)                # expect provider_response
grep -ho '"effort": "[a-z]*"' $(find $J -name llm_trajectory.jsonl) | sort -u    # Opus MAX -> "max"
python3 .claude/skills/benchflow-experiment-review/scripts/extract_harness_skills.py \
  "$(find $J -name llm_trajectory.jsonl | head -1)" --task-skill <task-skill-name>
```

Notes:
- `--usage-tracking required` records provider-reported token usage into each trajectory.
- `--agent-idle-timeout none` disables the idle watchdog (the task wall-clock still applies).
- Opus-4.8 on Bedrock needs the adaptive-thinking shim (`opus-4.8 bedrock thinking shim ACTIVE` in the install log); see `AGENTS.md`.
- For heavy tasks, replace `--sandbox daytona` with `--sandbox docker` — same flags otherwise.

---

## Running a benchmark with an Environment manifest

A **stateful** benchmark — one with mock services, databases, or accounts the
agent acts on — declares its world in an `environment.toml` manifest and runs
on the [Environment plane](./environment-plane.md). Use
`bench eval create --tasks-dir ...` for both single-task and batch
manifest-backed evaluations; `--environment-manifest` applies the manifest to
every rollout in the Job pipeline.

```bash
# single task
bench eval create --tasks-dir benchmarks/clawsbench/tasks/<task> \
  --environment-manifest benchmarks/clawsbench/environment.toml \
  --agent claude-agent-acp --model claude-haiku-4-5

bench eval create --tasks-dir benchmarks/chi-bench/tasks/<task> \
  --environment-manifest benchmarks/chi-bench/environment.toml \
  --agent claude-agent-acp --model claude-haiku-4-5

# batch via the Job API
bench eval create --tasks-dir benchmarks/clawsbench/tasks \
  --environment-manifest benchmarks/clawsbench/environment.toml \
  --agent claude-agent-acp --model claude-haiku-4-5
```

YAML configs may declare the same seam with ``environment_manifest:
<path>`` at the top level so the batch run is reproducible from disk.

`--environment-manifest` is distinct from `--sandbox`: the sandbox is *where*
the rollout runs; the environment manifest is *the world* the agent acts in.
BenchFlow provisions the environment, gates on its readiness before the agent
runs, and tears it down afterward. See [the Environment plane](./environment-plane.md)
for the full manifest schema, both onboarded benchmarks, and the
`snapshot`/`restore` roll-back contract.

---

## Running foreign benchmarks (inbound adapters)

BenchFlow runs benchmarks authored in other formats without converting them
first. An **inbound adapter** translates a foreign task directory into
BenchFlow-native shape; the rollout then runs natively. Two adapters ship:

| Source format | Signature file | Adapter |
|---------------|----------------|---------|
| Harbor | `task.toml` | `HarborAdapter` |
| Terminal-Bench | `task.yaml` | `TerminalBenchAdapter` |

`benchflow.adapters.inbound.detect_adapter()` sniffs a task directory and
picks the adapter whose format it matches (`task.toml` is checked first, so a
directory carrying both is treated as Harbor — the native superset). Each
adapter is a pure `Path -> InboundTask` translation: it reads a directory and
returns an in-memory native task, building no sandboxes and running nothing.
Terminal-Bench tasks are backward-compatible this way — old terminal-style
tasks keep running on BenchFlow unchanged.

---

## Continual learning (`sequential-shared` job mode)

By default a job runs its rollouts concurrently and isolated
(`parallel-independent`). A **continual-learning** job instead runs them
strictly in order over one persistent, versioned store of memory + skills —
set `job_mode: sequential-shared` in the YAML config:

```yaml
source:
  repo: benchflow-ai/skillsbench
  path: tasks
agent: claude-agent-acp
model: claude-haiku-4-5
job_mode: sequential-shared
```

In this mode each rollout reads the current `LearnerStore` state and, after
it scores, offers its reward as a learning-curve metric: an improvement
stamps a new generation, a regression is reverted to the best generation so
far. Concurrency is ignored — a shared mutable store cannot be written by
overlapping rollouts. See the [architecture doc](./architecture.md#the-eight-capabilities--how-each-fits),
capability 5, for the full design.

---

## Reading results

Results land under `jobs/<job-name>/<rollout-name>/`:

```
jobs/
└── harvey-lab-gemini-2026-05-06/
    ├── corporate-ma-review-data-room-red-flag-review/
    │   ├── result.json          # verifier output (reward, passed criteria)
    │   └── trajectory/
    │       └── acp_trajectory.jsonl  # full agent trace
    ├── real-estate-extract-psa-key-terms-scenario-01/
    │   ├── result.json
    │   └── trajectory/
    └── ...
```

The `result.json` contains:
```json
{
  "rewards": {"reward": 0.48},
  "n_tool_calls": 12,
  "n_skill_invocations": 2,
  "passed": true,
  "verifier_output": "..."
}
```

`n_skill_invocations` is derived from structured ACP trajectory events: BenchFlow
counts only `tool_call` events whose `kind` is `skill`. Job `summary.json`
also includes `total_skill_invocations` and `avg_skill_invocations` across the
rollouts in the run.

List evaluations:
```bash
bench eval list jobs/
```

---

## Running parity validation

Parity validation is a **developer/maintainer workflow** for verifying that an
adapter preserves benchmark semantics. These scripts live under each benchmark's
directory:

```bash
uv run python benchmarks/harvey-lab/parity_test.py \
  --mode full \
  --harvey-root .cache/datasets/harveyai/harvey-labs

ANTHROPIC_API_KEY=... uv run python benchmarks/harvey-lab/parity_test.py \
  --mode eval-parity

GEMINI_API_KEY=... uv run python benchmarks/harvey-lab/parity_test.py \
  --mode side-by-side
```

Recorded parity results are in `parity_experiment.json` and `benchmark.yaml`.

---

## YAML config reference

Job configs use the two-field `source` pattern to reference remote benchmark repos:

```yaml
# Example: SkillsBench config — direct from remote repo
source:
  repo: benchflow-ai/skillsbench   # GitHub repo (org/repo)
  path: tasks                      # subpath within the repo
  ref: main                        # branch/tag (optional)
agent: claude-agent-acp            # agent from registry
model: zai/glm-5.1                 # model ID
environment: daytona               # sandbox
concurrency: 8                     # parallel tasks
```

All adapted benchmarks use the same `source` pattern, pointing at the
[benchmarks dataset repo](https://github.com/benchflow-ai/benchmarks):

```yaml
# benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml
source:
  repo: benchflow-ai/benchmarks
  path: datasets/harvey-lab/tasks
agent: gemini
model: gemini/gemini-3.1-flash-lite-preview
environment: docker
concurrency: 4
```

```yaml
# benchmarks/programbench/programbench-gemini-flash-lite.yaml
source:
  repo: benchflow-ai/benchmarks
  path: datasets/programbench/tasks
agent: gemini
model: gemini-3.1-flash-lite-preview
environment: docker
concurrency: 4
```

You can also use `tasks_dir:` for local paths:

```yaml
tasks_dir: ./my-local-tasks
agent: gemini
model: gemini/gemini-3.1-flash-lite-preview
```

All fields from [CLI reference](./reference/cli.md#yaml-config-format) apply:
`source`, `tasks_dir`, `agent`, `model`, `environment`, `concurrency`,
`sandbox_setup_timeout`, `skills_dir`, `agent_env`, `max_retries`.

---

## Adding a new benchmark

See the [Benchmark Conversion Guide](../benchmarks/CONVERT.md) for the 9-step
process to convert a new benchmark into Harbor-format tasks for BenchFlow. Harvey LAB
(`benchmarks/harvey-lab/`) and ProgramBench (`benchmarks/programbench/`) are
reference implementations.
