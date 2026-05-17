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

---

## bench run

### bench run

Run one task directory with one agent. This is the most direct command for
single-task local, Daytona, or Modal checks.

```bash
# Single task with Gemini on Daytona
bench run tasks/edit-pdf \
  --agent gemini \
  --model gemini-3.1-flash-lite-preview \
  --sandbox daytona

# Single task with mounted skills and the recommended skill nudge
bench run tasks/pdf-fix \
  --agent gemini \
  --model gemini-3.1-flash-lite-preview \
  --sandbox daytona \
  --skills-dir tasks/pdf-fix/environment/skills \
  --ae BENCHFLOW_SKILL_NUDGE=name
```

| Flag | Default | Description |
|------|---------|-------------|
| `TASK_DIR` | — | Task directory containing `task.toml` |
| `--agent` | `claude-agent-acp` | Agent name from the registry |
| `--model` | Agent default | Model ID |
| `--sandbox` | `docker` | Sandbox: docker, daytona, or modal |
| `--prompt` | `instruction.md` | Prompt text; repeat for multi-turn |
| `--jobs-dir` | `jobs` | Output directory |
| `--agent-env`, `--ae` | — | Agent environment variable as `KEY=VALUE`; repeatable |
| `--skills-dir` | — | Skills directory to deploy into the sandbox |
| `--sandbox-user` | `agent` | Non-root sandbox user; pass `none` for root |

When mounting skills, the recommended docs default is
`--ae BENCHFLOW_SKILL_NUDGE=name`. It prepends a short hint telling the agent
which skills are available and where to read them. More verbose modes are
`description` and `full`. Omit the env var to leave BenchFlow's runtime default
off.

---

## bench eval

### bench eval create

Create and run an evaluation. Use it for YAML configs and batch runs; it also
accepts a single task directory.

```bash
# From YAML config
bench eval create --config benchmarks/skillsbench-claude-glm51.yaml

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
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | — | YAML config file |
| `--tasks-dir` | — | Local task dir (single task with task.toml, or parent of many) |
| `--source-repo` | — | Remote repo as `org/repo` (e.g. `benchflow-ai/skillsbench`) |
| `--source-path` | — | Subpath within the repo (e.g. `tasks`) |
| `--source-ref` | — | Branch or tag to clone (e.g. `main`) |
| `--agent` | `claude-agent-acp` | Agent name |
| `--model` | Agent default | Model ID |
| `--sandbox` | `docker` | Sandbox: docker, daytona, or modal |
| `--concurrency` | `4` | Max concurrent tasks (batch mode only) |
| `--jobs-dir` | `jobs` | Output directory |
| `--sandbox-user` | `agent` | Sandbox user (null for root) |
| `--sandbox-setup-timeout` | `120` | Timeout in seconds for sandbox user setup |
| `--skills-dir` | — | Skills directory to deploy into each task sandbox |

### bench eval list

List completed evaluations from a jobs directory.

```bash
bench eval list jobs/
```

---

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
bench tasks check tasks/my-task --rubric rubrics/quality.md
```

---

## bench train

### bench train create

Run a reward-based training sweep.

```bash
bench train create \
  --tasks-dir tasks/ \
  --agent gemini \
  --sweeps 5 \
  --export ./training-data
```

---

## bench environment

### bench environment create

Create an environment from a task directory (spins up sandbox).

```bash
bench environment create tasks/my-task --sandbox daytona
```

### bench environment list

List active Daytona sandboxes.

```bash
bench environment list
```

---

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
batch job configs; scene configs are loaded with `benchflow.trial_yaml` or built
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

## Deprecated Commands

These still work but are hidden from `--help`:

| Old command | Replacement |
|-------------|-------------|
| `benchflow run` | `bench run <task>` |
| `benchflow job` | `bench eval create --config <yaml>` |
| `benchflow agents` | `bench agent list` |
| `benchflow eval` | `bench skills eval` |
| `benchflow metrics` | `bench eval list --detail` |
| `benchflow view` | (planned: `bench trajectory show`) |
| `benchflow cleanup` | `bench environment list` + delete |
| `benchflow skills install` | Skills are folders, not packages |
