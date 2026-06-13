# CLI reference
BenchFlow uses a resource-verb pattern: `bench <resource> <verb>`.

```bash
bench --version
```

---

## bench agent

> **Two different nouns share this group.** `bench agent list` and `bench agent
> show` operate on **registered AI agents** (Claude Code, Gemini CLI, Codex,
> OpenHands, …) — the programs that solve tasks. `bench agent create`, `bench
> agent run`, and `bench agent verify` operate on **benchmark adoptions** —
> scaffolding, driving, and parity-gating a `benchmarks/<name>/` adoption of a
> third-party benchmark. They are unrelated despite living under the same
> command group: `list`/`show` inspect solver agents, while
> `create`/`run`/`verify` are the benchmark-onboarding workflow.

### bench agent list

List all registered agents with their protocol and native/default auth
requirements. Provider-prefixed models may use provider-specific credentials;
Azure Foundry models use `AZURE_API_KEY` plus `AZURE_API_ENDPOINT`.

```bash
bench agent list
```

### bench agent show

Show details for a specific agent, including native/default auth and a note
about provider-specific credentials.

```bash
bench agent show gemini
```

### bench agent create

Scaffold `benchmarks/<name>/` for a new benchmark adoption. The layout mirrors
the reference benchmark `benchmarks/programbench/` and the contract in
[`benchmarks/CONVERT.md`](../../benchmarks/CONVERT.md): it writes
`benchflow.py` (converter), `main.py`, `parity_test.py`, `run_<name>.py`,
`<name>.yaml`, `benchmark.yaml`, `parity_experiment.json` (status `template`),
`README.md`, and `__init__.py`. It is fail-closed: the slug is validated
(lowercase, leading letter, single internal hyphens, max 64 chars) and the
command refuses to overwrite an existing benchmark directory.

```bash
bench agent create my-bench
bench agent create my-bench --benchmarks-dir ./benchmarks
```

| Flag | Default | Description |
|------|---------|-------------|
| `--benchmarks-dir` | repo `benchmarks/` | Target benchmarks/ directory |

### bench agent run

Drive the `CONVERT.md` adoption workflow by launching the host `codex` CLI.
The command assembles the adoption context (the source, the target
`benchmarks/<name>/` path, the adoption skills, and the embedded
`benchmarks/CONVERT.md` guide) and runs `codex exec` against the repo root to
drive the conversion toward a `benchmarks/<name>/` pull request. It is
fail-closed on credentials: `codex` needs `OPENAI_API_KEY` (or `CODEX_API_KEY`)
in the environment, or a `~/.codex/auth.json` from `codex login`, otherwise the
command exits before assembling any context. Use `--dry-run` to print the exact
launch command without running it (no credentials required). When `--name` is
omitted the slug is derived from the source basename.

```bash
# Print the codex launch command without running it
bench agent run https://github.com/org/some-benchmark --dry-run

# Launch the host codex driver against a local source
bench agent run ./vendor/some-benchmark --name my-bench --model o3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | derived from source | Benchmark slug (default: from source basename) |
| `--model` | codex default | Model for the codex driver |
| `--dry-run` | `false` | Print the launch command, do not run |
| `--codex-bin` | `codex` | Host codex binary |

### bench agent verify

Run the parity gate for an adopted benchmark and emit a confidence verdict. It
reads `benchmarks/<name>/parity_experiment.json` and scores two layers: a
deterministic conversion-faithfulness floor (every compared criterion's
converted verdict must match the original's verdict on identical inputs) and a
statistical reward-distribution layer (every legacy-vs-converted reward delta
must sit within `--tolerance`). The gate is parity-only — a faithful conversion
reproduces the original's behavior, including any reward-hackability the source
has; it never "improves" or sanitizes the source. The verdict is one of
`parity-confirmed`, `parity-divergent`, or `insufficient-evidence` (no recorded
comparisons). On any non-confirmed verdict the command exits non-zero and emits
a draft GitHub issue body for human support — printed to stdout, or written to
`--issue-out`. The draft is never filed automatically. Pass `--roundtrip-task`
to also run the structural round-trip conformance check on a concrete task
directory.

```bash
bench agent verify my-bench
bench agent verify my-bench --tolerance 0.05 --issue-out divergence.md
bench agent verify my-bench --roundtrip-task benchmarks/my-bench/tasks/example
bench agent verify my-bench --json
bench agent verify my-bench --require-adoption-report --loop-report-out loop-report.json --json
```

Use `--json` when an adapter-adoption loop needs a parseable parity verdict.
Confirmed records emit `{"status": "parity-confirmed", "passed": true, ...}`;
divergent records emit conversion disagreements, reward deltas, and either
`issue_out` or an inline `issue_draft`; insufficient records emit
`{"status": "insufficient-evidence", "passed": false, ...}`.
If `benchmarks/<name>/adoption_report.json` exists, `--json` includes it under
`adoption_report` with the sidecar path and scrubbed architecture/artifact
manifest payload. Use `--require-adoption-report` for 0.7 environment-adapter
adoptions: it fails unless `adoption_report.json` proves the architecture
planes, reward/parity agreement, trace and screenshot artifacts, eval summary,
timing, and cleanup criterion, and `loop_state.json` proves the resumable
controller state is at least `review-ready`. `--loop-report-out` writes the
same machine-readable verdict to disk for a controller or reviewer role.

| Flag | Default | Description |
|------|---------|-------------|
| `--benchmarks-dir` | repo `benchmarks/` | Target benchmarks/ directory |
| `--tolerance` | `0.02` | Max abs reward delta (statistical layer) |
| `--issue-out` | — | Write the divergence issue draft to this path instead of stdout |
| `--roundtrip-task` | — | Also run the structural round-trip check on this task dir |
| `--json` | `false` | Emit a machine-readable parity verdict or error |
| `--require-adoption-report` | `false` | Fail unless `adoption_report.json` passes the 0.7 adapter-adoption loop gate |
| `--loop-report-out` | — | Write the parseable verify/adoption-loop verdict to this path |

## bench eval

### bench eval create

Create and run an evaluation. Use it for YAML configs and batch runs; it also
accepts a single task directory.

```bash
# From YAML config
bench eval create --config benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml

# From remote repo (fast Daytona batch; token usage may be unavailable)
bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks \
  --agent gemini \
  --model gemini-3.1-flash-lite-preview \
  --sandbox daytona \
  --concurrency 64 \
  --sandbox-setup-timeout 300

# From remote repo with required token usage telemetry
bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks \
  --agent gemini \
  --model gemini-3.1-flash-lite-preview \
  --sandbox daytona \
  --usage-tracking required \
  --concurrency 16 \
  --sandbox-setup-timeout 300

# From local directory
bench eval create --tasks-dir ./tasks --agent gemini --model gemini-3.1-flash-lite-preview

# From local directory with a machine-readable run report
bench eval create --tasks-dir ./tasks --agent gemini --sandbox docker --json

# From a hosted PrimeIntellect / Verifiers environment
bench eval create \
  --source-env primeintellect/general-agent \
  --source-env-version 0.1.1 \
  --source-env-arg task=calendar_scheduling_t0 \
  --agent gemini \
  --model google/gemini-2.5-flash-lite

# Single task with mounted skills and the recommended skill nudge
bench eval create \
  --tasks-dir tasks/pdf-fix \
  --agent gemini \
  --model gemini-3.1-flash-lite-preview \
  --sandbox daytona \
  --skill-mode with-skill \
  --agent-env BENCHFLOW_SKILL_NUDGE=name
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | — | YAML config file |
| `--tasks-dir` | — | Local task dir (native task, supported foreign task descriptor such as `browser-use-task.json` / `stagehand-task.json`, or parent of many) |
| `--source-repo` | — | Remote repo as `org/repo` (e.g. `benchflow-ai/skillsbench`) |
| `--source-path` | — | Subpath within the repo (e.g. `tasks`) |
| `--source-ref` | — | Branch or tag to clone (e.g. `main`) |
| `--source-env` | — | Hosted environment source (e.g. `primeintellect/general-agent`) |
| `--source-env-version` | — | Hosted environment version |
| `--source-env-arg` | — | Hosted environment argument as `KEY=VALUE`; repeatable |
| `--source-env-num-examples` | `1` | Number of hosted environment examples |
| `--source-env-rollouts-per-example` | `1` | Rollouts per hosted environment example |
| `--source-env-max-tokens` | `1024` | Max tokens for hosted environment model calls |
| `--source-env-temperature` | `0.0` | Temperature for hosted environment model calls |
| `--source-env-sampling-arg` | — | Verifiers sampling argument as `KEY=VALUE`; repeatable (for example `reasoning_effort=minimal`) |
| `--agent` | `claude-agent-acp` | Agent name |
| `--model` | Agent default | Model ID |
| `--reasoning-effort` | — | Agent reasoning/thinking effort when the agent exposes one (e.g. `max`) |
| `--sandbox` | `docker` | Sandbox: docker, daytona, modal, or cua |
| `--usage-tracking` | `auto` | Token usage telemetry policy: `auto`, `required`, or `off` |
| `--environment-manifest` | — | Path to an Environment-plane manifest (`environment.toml`); applied to every rollout in the batch |
| `--prompt` | `instruction.md` | Prompt to send to the agent; repeatable for multi-prompt runs |
| `--concurrency` | `4` | Max concurrent tasks (batch mode only) |
| `--build-concurrency` | `--concurrency` | Max concurrent docker image builds; set lower (e.g. `8`) when `--concurrency` is high to avoid overwhelming the docker daemon |
| `--worker-concurrency` | — | Run batch eval through isolated worker subprocesses, each with at most this many concurrent tasks; `--concurrency` remains the aggregate target |
| `--worker-retries` | `1` | Retry a crashed worker shard this many times, resuming its jobs dir |
| `--worker-start-stagger-sec` | `1.0` | Seconds to stagger worker starts to avoid Daytona connection storms |
| `--agent-idle-timeout` | (built-in default) | Abort ACP prompts after this many idle seconds; `0` disables idle detection |
| `--jobs-dir` | `jobs` | Output directory |
| `--sandbox-user` | `agent` | Sandbox user (null for root) |
| `--sandbox-setup-timeout` | `120` | Timeout in seconds for sandbox user setup |
| `--skills-dir` | — | Advanced custom skills directory; valid only with `--skill-mode with-skill`. Omit it to use each task's `environment/skills`. |
| `--skill-mode` | `no-skill` | Skill mode: `no-skill`, `with-skill`, or `self-gen` |
| `--skill-creator-dir` | — | Path to a `skill-creator` directory (or a skills root containing it); used when `--skill-mode self-gen` |
| `--self-gen-no-internet` | `false` | Disable web tools for the self-generated skill run |
| `--agent-env` | — | Agent environment variable as `KEY=VALUE`; repeatable |
| `--include` | — | Only run these task names; repeatable (e.g. `--include jax-computing-basics --include data-to-d3`) |
| `--exclude` | — | Skip these task names; repeatable (e.g. `--exclude quantum-numerical-simulation`) |
| `--json` | `false` | Emit a machine-readable eval run report or error. Successful local/repo runs include the persisted `summary.json` payload. |

When mounting skills, the recommended docs default is
`--agent-env BENCHFLOW_SKILL_NUDGE=name`. See
[Architecture: skill loading](../architecture.md#skill-loading) for how
`with-skill` mode is registered with each agent and how the nudge modes differ.

Daytona batch runs collect provider token/cost telemetry by default with a
sandbox-local LiteLLM gateway. Use `--usage-tracking required` when missing telemetry
should fail the rollout, or `--usage-tracking off` for recovery runs that should
leave provider traffic untouched.

`--source-env` is for external hosted environment hubs. The first supported
runner is PrimeIntellect / Verifiers: BenchFlow preserves the hosted identity
(`env_uid`, `hub_url`), installs the versioned package into an isolated local
virtual environment, and runs `vf-eval`. `--sandbox` remains the BenchFlow task
sandbox selector for local/repo task sources; Verifiers source environments own
their own harness and sandbox behavior. `--model` is passed to the Verifiers
model endpoint; use a model id available to that provider. Provider-specific
sampling options are not inferred; pass them explicitly with
`--source-env-sampling-arg`.

### bench eval list

List completed evaluations from a jobs directory.

```bash
bench eval list jobs/
```

## bench skills

### bench skills list

List skills discovered under the default skills roots (or `--dir`).

```bash
bench skills list
bench skills list --dir ./skills
```

### bench skills eval

Evaluate a skill against its evals.json test cases.

```bash
bench skills eval skills/my-skill/ \
  --agent gemini \
  --model gemini-3.1-flash-lite-preview \
  --sandbox daytona
```


---

## bench tasks

### bench tasks init

Scaffold a new benchmark task.

```bash
bench tasks init my-new-task
bench tasks init my-new-task --dir tasks/
bench tasks init my-new-task --format legacy
```

| Flag | Default | Description |
|------|---------|-------------|
| `--format` | `task-md` | Task format: `task-md` (native single-document) or `legacy` (split `task.toml` + `instruction.md` layout) |

### bench tasks check

Validate a task directory (`task.md` or legacy `task.toml` + `instruction.md`, `environment/Dockerfile`, `verifier/` or legacy `tests/`).

```bash
bench tasks check tasks/my-task
bench tasks check tasks/my-task --level runtime-capability --sandbox cua --json
```

With `--level`, validation runs at a chosen depth: `schema`, `structural`,
`runtime-capability`, `publication-grade`, `acceptance`, or `acceptance-live`.
Acceptance-level errors such as
`acceptance validation requires benchflow.evidence mapping` refer to the
`benchflow.evidence` schema documented in the "Assets, Provenance, And
Evidence" section of `docs/task-standard.md`.

Use `--json` when an adapter-adoption loop needs a machine-readable result.
Supported foreign tasks emit `{"status": "valid", "adapter": ...}`; recognized
but unsupported foreign tasks emit `{"status": "unsupported-adapter-task",
"adapter": ..., "reason": ..., "details": ...}` and exit non-zero. Current
foreign task signatures include `browser-use-task.json`, `stagehand-task.json`,
`computer-use-task.json`, `iosworld-task.json`, `task.yaml`, and compatible
`task.toml` variants.

| Flag | Default | Description |
|------|---------|-------------|
| `--level` | `structural` | Validation depth |
| `--sandbox` | — | Validate runtime semantics for docker, daytona, modal, or cua |
| `--json` | `false` | Emit a machine-readable validation or unsupported-task report |

### bench tasks migrate

Convert a legacy `task.toml` + `instruction.md` task into the unified
`task.md` format. By default the legacy files are kept alongside the new
`task.md`.

```bash
bench tasks migrate tasks/my-task
bench tasks migrate tasks/my-task --overwrite --remove-legacy
```

| Flag | Default | Description |
|------|---------|-------------|
| `--overwrite` | `false` | Replace an existing task.md |
| `--remove-legacy` | `false` | Delete split files and promote tests/solution aliases after task.md is verified |

### bench tasks normalize

Expand minimal `task.md` authoring profiles into the canonical `task.md`
form. Prints the normalized document to stdout unless told otherwise.

```bash
bench tasks normalize tasks/my-task
bench tasks normalize tasks/my-task --write
bench tasks normalize tasks/my-task -o normalized-task.md
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | — | Write normalized task.md to this path instead of stdout |
| `--write` | `false` | Replace task.md in place with the normalized canonical form |

### bench tasks export

Export a `task.md` task to a Harbor/Pier-compatible split layout, with a
compatibility loss report written to `compatibility/export-report.json` in
the export directory.

```bash
bench tasks export tasks/my-task out/my-task-split
bench tasks export tasks/my-task --report-only
bench tasks export tasks/my-task out/my-task-split --target pier --overwrite
```

Arguments: `TASK_DIR` (task directory to export) and optional `OUTPUT_DIR`
(destination split-layout directory; may be omitted with `--report-only`).

| Flag | Default | Description |
|------|---------|-------------|
| `--target` | `harbor` | Compatibility target: `harbor` or `pier` |
| `--overwrite` | `false` | Replace an existing export directory |
| `--report-only` | `false` | Print the compatibility loss report without writing files |

### bench tasks generate

Generate benchmark task directories from real agent traces.

```bash
bench tasks generate --from-local --project my-repo --limit 5
bench tasks generate --from-file session.jsonl --dry-run
bench tasks generate --from-hf opentraces-test --limit 50
```

| Flag | Default | Description |
|------|---------|-------------|
| `--from-local` | — | Generate from local Claude Code sessions |
| `--from-file` | — | Generate from a JSONL trace file |
| `--from-hf` | — | Generate from a HuggingFace dataset ID or alias |
| `--output` | `tasks` | Output directory for generated tasks |
| `--projects-dir` | `~/.claude/projects/` | Claude Code projects directory |
| `--project` | — | Filter local sessions by project path substring |
| `--format` | `auto` | Trace format override |
| `--split` | `train` | HuggingFace dataset split |
| `--max-rows` | `100` | Max rows to download from HuggingFace |
| `--limit` | `20` | Max traces to process |
| `--min-steps` | `2` | Minimum steps per trace |
| `--outcome` | — | Filter by outcome: success, failure, unknown |
| `--author` | `benchflow-traces` | Author name for generated task metadata |
| `--task-format` | `task-md` | Generated task package format: `task-md` or `legacy` |
| `--dry-run` | `false` | Preview traces without generating tasks |

### bench tasks list-sources

List known HuggingFace trace datasets. The aliases listed here can be passed
to `bench tasks generate --from-hf`.

```bash
bench tasks list-sources
```

## bench environment

### bench environment create

Create an environment object from a task directory. This validates environment
construction but does not start the sandbox.

```bash
bench environment create tasks/my-task --sandbox daytona
bench environment create tasks/my-task --sandbox cua --dry-run
bench environment create tasks/my-task --sandbox cua --dry-run --json
```

Use `--json` when an adapter-adoption loop needs a parseable create report.
Dry runs emit `{"status": "dry-run", "adapter": ..., "environment_adapter":
..., "created": false}`. Foreign task materialization emits
`{"status": "created", "adapter": ..., "environment_adapter": ..., "native":
"materialized-temporary"}`. Browser Use tasks report
`environment_adapter.name = "browser"`; computer-use and use-computer cookbook
tasks report `environment_adapter.name = "desktop"`.

### bench environment check

Check task runtime compatibility and provider readiness without starting a
sandbox.

```bash
bench environment check tasks/my-task --sandbox docker
bench environment check tasks/my-task --sandbox cua
bench environment check tasks/my-task --sandbox cua --json
bench environment check tasks/my-task --sandbox cua --probe-runtime --json
```

Use `--json` to capture task compatibility and provider readiness in one record.
Supported tasks emit `{"status": "ready", "adapter": ...,
"environment_adapter": ..., "provider": ...}`. The `environment_adapter`
object names the world the agent acts in, required capabilities, verified
sandbox providers, and whether the selected provider has verified or unverified
support. Recognized but unsupported foreign tasks emit the same structured
`unsupported-adapter-task` payload as `bench tasks check --json`.

For desktop/computer-use tasks, Cua support is evidence-scoped. Local Cua
reports `environment_adapter.provider_support = "verified"` with
`provider_mode = "local"`. Cua cloud reports
`provider_support = "runtime-probe-required"` until a live
`--probe-runtime` check succeeds; a successful cloud probe upgrades the report
to verified with `verified_sandboxes` including `cua:cloud-probed`.

Use `--probe-runtime` for Cua provider dogfood when you need live runtime
evidence, not just SDK/auth readiness. The probe creates or connects to a Cua
runtime, runs a bounded shell command, checks upload/download file transfer,
screenshot/dimensions/display metadata, records non-sensitive diagnostics, and
then cleans up. Display URL support is reported as a capability result; local
Cua may report it unsupported while the required runtime checks still pass.
If cloud startup fails, the JSON runtime probe names `failed_capabilities` and
captures SDK background readiness errors under `background_errors`. It also
includes SDK/version metadata, request metadata such as `linux_kind`, and a
normalized `failure_class` when a known cloud-runtime pattern is detected.

### bench environment list

List active provider environments, or list a hosted hub. Provider listing is
supported for Docker, Daytona, and Cua. Docker listing is scoped to
BenchFlow-owned resources labeled `benchflow.owned=true`.

```bash
bench environment list
bench environment list --sandbox docker --json
bench environment list --sandbox cua --json
bench environment list --hub primeintellect --owner primeintellect --search general-agent --limit 5
```

### bench environment show

Show hosted environment metadata.

```bash
bench environment show primeintellect/general-agent --version 0.1.1
```

### bench environment inspect

Inspect a file from a hosted environment package.

```bash
bench environment inspect primeintellect/general-agent --version 0.1.1 --path README.md
```

### bench environment cleanup

Clean up orphaned provider sandboxes. By default this deletes sandboxes older
than 24 hours; use `--dry-run` to preview what would be deleted. Provider
cleanup is supported for Docker, Daytona, and Cua. Docker cleanup is scoped to
BenchFlow-owned containers and networks labeled `benchflow.owned=true`.

```bash
bench environment cleanup --dry-run --max-age 1440
bench environment cleanup --sandbox docker --dry-run --max-age 60 --json
bench environment cleanup --sandbox cua --dry-run --max-age 60
bench environment cleanup --sandbox cua --dry-run --max-age 60 --json
```

Daytona-backed evals also reap orphaned sandboxes automatically at run start
(failure states such as `BUILD_FAILED` are reaped sooner than healthy ones, and
an idle-activity guard means concurrent live runs are never reaped). Set
`BENCHFLOW_DAYTONA_AUTO_REAP` to any of `0`/`false`/`no`/`off` (case-insensitive)
to disable that automatic pass and rely on the manual command above.

For Docker and Cua, `--json` emits a cleanup report with `found`, `matched`,
`skipped`, `deleted`, and `candidates` fields. Use it in adapter dogfood to
prove cleanup would leave no BenchFlow-owned resources behind before scaling.

## bench compat

Third-party framework compatibility checks.

### bench compat harbor-registry

Inventory or structurally check representative Harbor registry tasks. Defaults
to running an inventory pass against the public Harbor registry JSON.

```bash
# Inventory the public Harbor registry
bench compat harbor-registry

# Structural check, two tasks per dataset, JSONL output
bench compat harbor-registry --level check --tasks-per-dataset 2 --out compat.jsonl
```

| Flag | Default | Description |
|------|---------|-------------|
| `--registry` | Harbor public registry URL | Harbor registry JSON URL or local file |
| `--tasks-per-dataset` | `2` | Representative tasks selected per dataset |
| `--level` | `inventory` | Compatibility level: `inventory` or `check` |
| `--out` | — | Optional JSONL output path |
| `--cache-dir` | `.cache/compat/harbor` | Cache directory for sparse clones |
| `--limit` | — | Optional cap on selected task refs |

## YAML Config Format

### Batch config with skills and skill nudge

```yaml
source:
  repo: benchflow-ai/skillsbench
  path: tasks
environment: daytona
concurrency: 64
sandbox_setup_timeout: 300
agent: gemini
model: gemini-3.1-flash-lite-preview
skill_mode: with-skill
skills_dir: shared-skills/
agent_env:
  BENCHFLOW_SKILL_NUDGE: name
max_retries: 2
```

### Multi-scene (BYOS skill generation)

Use the Python API for multi-scene experiments. `bench eval create --config` is for
batch job configs; scene configs are loaded with `benchflow._utils.yaml_loader` or built
directly in Python.

```yaml
task_dir: tasks/my-task
environment: daytona
sandbox_setup_timeout: 300

scenes:
  - name: skill-gen
    roles:
      - name: creator
        agent: gemini
        model: gemini-3.1-flash-lite-preview
    turns:
      - role: creator
        prompt: "Analyze the task and write a skill document to /app/generated-skill.md"

  - name: solve
    roles:
      - name: solver
        agent: gemini
        model: gemini-3.1-flash-lite-preview
    turns:
      - role: solver
```

---

## bench continue

Resume a previous, unfinished (timed-out) `openhands` run to completion via
record-replay. Standalone — it does not touch the normal run path. See
[Continuing timed-out runs](../continue-runs.md) for the full guide.

```bash
bench continue path/to/original/run-folder --tasks-dir path/to/tasks
```

Key options: `--model` (override the live-continuation model; defaults to the
original run's model), `--timeout`, `--output`, `--require-timeout`,
`--strict-divergence`, `--replay-only` (rebuild via replay and stop at the
cut-point — no live model or API key needed), and `--proxy-mode` (replay
proxy placement: `auto`, `host`, or `sandbox`; default `auto` uses
sandbox-local replay for Daytona/Modal and host replay for Docker).

### bench continue-batch

Continue all timed-out OpenHands runs found under a directory tree. Discovers
run folders (`config.json` + `trajectory/llm_trajectory.jsonl`) recursively,
continues each, and prints a JSON batch summary (exits 1 if any continuation
failed).

```bash
bench continue-batch path/to/jobs-root --tasks-dir path/to/tasks
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tasks-dir` | — | Directory holding task sources; required unless the recorded task path exists |
| `--model` | original run's model | Override the live-continuation model |
| `--timeout` | — | Wall-clock budget per continuation |
| `--output` | — | Output jobs dir for continued runs |
| `--concurrency` | `100` | Maximum number of continuation runs in flight |
| `--limit` | — | Limit discovered timeout folders |
| `--strict-divergence` | `false` | Abort a run if replay leaves the original rails |
| `--proxy-mode` | `auto` | Replay proxy placement: `auto`, `host`, or `sandbox` |
