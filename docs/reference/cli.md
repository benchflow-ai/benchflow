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
| `--agent`, `-a` | `claude-agent-acp` | Agent name from the registry |
| `--model`, `-m` | Agent default | Model ID |
| `--sandbox`, `-b` | `docker` | Sandbox: docker, daytona, or modal |
| `--prompt`, `-p` | `instruction.md` | Prompt text; repeat for multi-turn |
| `--jobs-dir`, `-o` | `jobs` | Output directory |
| `--agent-env`, `--ae` | — | Agent environment variable as `KEY=VALUE`; repeatable |
| `--skills-dir`, `-s` | — | Skills directory to deploy into the sandbox |
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
bench eval create -f benchmarks/skillsbench-claude-glm51.yaml

# From remote repo
bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks \
  -a gemini \
  -m gemini-3.1-flash-lite-preview \
  -e daytona \
  -c 64 \
  --sandbox-setup-timeout 300

# From local directory
bench eval create -t ./tasks -a gemini -m gemini-3.1-flash-lite-preview
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config`, `-f` | — | YAML config file |
| `--tasks-dir`, `-t` | — | Local task dir (single task with task.toml, or parent of many) |
| `--source-repo` | — | Remote repo as `org/repo` (e.g. `benchflow-ai/skillsbench`) |
| `--source-path` | — | Subpath within the repo (e.g. `tasks`) |
| `--source-ref` | — | Branch or tag to clone (e.g. `main`) |
| `--agent`, `-a` | `claude-agent-acp` | Agent name |
| `--model`, `-m` | Agent default | Model ID |
| `--env`, `-e` | `docker` | Environment: docker, daytona, or modal |
| `--concurrency`, `-c` | `4` | Max concurrent tasks (batch mode only) |
| `--jobs-dir`, `-o` | `jobs` | Output directory |
| `--sandbox-user` | `agent` | Sandbox user (null for root) |
| `--sandbox-setup-timeout` | `120` | Timeout in seconds for sandbox user setup |
| `--skills-dir`, `-s` | — | Skills directory to deploy into each task sandbox |

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
  -a gemini \
  -m gemini-3.1-flash-lite-preview \
  --env daytona
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

### bench tasks generate

Generate benchmark tasks from agent traces. Supports local Claude Code sessions,
JSONL trace files, and HuggingFace datasets.

```bash
# From local Claude Code sessions
bench tasks generate --from-local
bench tasks generate --from-local --project my-repo --limit 5

# From a JSONL trace file (auto-detects Claude Code vs opentraces format)
bench tasks generate --from-file session.jsonl --dry-run
bench tasks generate --from-file traces.jsonl --format opentraces

# From a HuggingFace dataset (use alias or full repo ID)
bench tasks generate --from-hf opentraces-test -n 50 --outcome success
bench tasks generate --from-hf nlile/misc-merged-claude-code-traces-v1 -n 100
```

| Flag | Default | Description |
|------|---------|-------------|
| `--from-local` | — | Generate from local Claude Code sessions |
| `--from-file` | — | Generate from a JSONL trace file |
| `--from-hf` | — | Generate from a HuggingFace dataset (ID or alias) |
| `--output`, `-o` | `tasks` | Output directory for generated tasks |
| `--project`, `-p` | — | Filter local sessions by project path substring |
| `--projects-dir` | `~/.claude/projects/` | Claude Code projects directory |
| `--format`, `-f` | `auto` | Trace format: auto, claude-code, opentraces |
| `--split` | `train` | HuggingFace dataset split |
| `--max-rows` | `100` | Max rows to download from HuggingFace |
| `--limit`, `-n` | `20` | Max traces to process |
| `--min-steps` | `2` | Minimum steps per trace |
| `--outcome` | — | Filter by outcome: success, failure, unknown |
| `--author` | `benchflow-traces` | Author name for task.toml |
| `--dry-run` | — | Preview traces without generating tasks |

### bench tasks list-sources

List known HuggingFace trace datasets and their aliases.

```bash
bench tasks list-sources
```

---

## bench train

### bench train create

Run a reward-based training sweep.

```bash
bench train create \
  -t tasks/ \
  -a gemini \
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

Use the Python API for multi-scene experiments. `bench eval create -f` is for
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
| `benchflow job` | `bench eval create -f <yaml>` |
| `benchflow agents` | `bench agent list` |
| `benchflow eval` | `bench skills eval` |
| `benchflow metrics` | `bench eval list --detail` |
| `benchflow view` | (planned: `bench trajectory show`) |
| `benchflow cleanup` | `bench environment list` + delete |
| `benchflow skills install` | Skills are folders, not packages |
