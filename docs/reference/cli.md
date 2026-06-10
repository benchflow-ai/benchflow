# CLI reference
BenchFlow uses a resource-verb pattern: `bench <resource> <verb>`.

```bash
bench --version
```

---

## bench agent

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
| `--tasks-dir` | — | Local task dir (single task with task.toml, or parent of many) |
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
| `--sandbox` | `docker` | Sandbox: docker, daytona, or modal |
| `--usage-tracking` | `auto` | Token usage telemetry policy: `auto`, `required`, or `off` |
| `--environment-manifest` | — | Path to an Environment-plane manifest (`environment.toml`); applied to every rollout in the batch |
| `--prompt` | `instruction.md` | Prompt to send to the agent; repeatable for multi-prompt runs |
| `--concurrency` | `4` | Max concurrent tasks (batch mode only) |
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
```

### bench tasks check

Validate a task directory (Dockerfile, instruction.md, tests/).

```bash
bench tasks check tasks/my-task
```

On the development branch, `bench tasks check` gains a `--level` option
(`schema`, `structural`, `runtime-capability`, `publication-grade`,
`acceptance`, `acceptance-live`). Acceptance-level errors such as
`acceptance validation requires benchflow.evidence mapping` refer to the
`benchflow.evidence` schema documented in the "Assets, Provenance, And
Evidence" section of `docs/task-standard.md` on the same branch.

### bench tasks migrate

> Available on the development branch; not yet in this release.

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

> Available on the development branch; not yet in this release.

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
```

### bench environment list

List active Daytona sandboxes, or list a hosted hub.

```bash
bench environment list
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

Clean up orphaned Daytona sandboxes. By default this deletes sandboxes older
than 24 hours; use `--dry-run` to preview what would be deleted.

```bash
bench environment cleanup --dry-run --max-age 1440
```

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
`--strict-divergence`, and `--replay-only` (rebuild via replay and stop at the
cut-point — no live model or API key needed).
