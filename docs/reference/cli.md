# CLI reference
BenchFlow uses a resource-verb pattern: `bench <resource> <verb>`.

---

## bench agent

### bench agent list

List all registered agents with their protocol and auth requirements.

```bash
bench agent list
```

### bench agent show

Show details for a specific agent.

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

# From remote repo
bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks \
  --agent gemini \
  --model gemini-3.1-flash-lite-preview \
  --sandbox daytona \
  --concurrency 64 \
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
  --skills-dir tasks/pdf-fix/environment/skills \
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
| `--concurrency` | `4` | Max concurrent tasks (batch mode only) |
| `--jobs-dir` | `jobs` | Output directory |
| `--sandbox-user` | `agent` | Sandbox user (null for root) |
| `--sandbox-setup-timeout` | `120` | Timeout in seconds for sandbox user setup |
| `--skills-dir` | — | Skills directory to deploy into each task sandbox |
| `--agent-env` | — | Agent environment variable as `KEY=VALUE`; repeatable |

When mounting skills, the recommended docs default is
`--agent-env BENCHFLOW_SKILL_NUDGE=name`. It prepends a short hint telling the agent
which skills are available and where to read them. More verbose modes are
`description` and `full`. Omit the env var to leave BenchFlow's runtime default
off.

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
