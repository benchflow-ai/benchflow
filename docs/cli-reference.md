# CLI Reference

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

## bench eval

### bench eval create

Create and run an evaluation. This is the primary command for running benchmarks.

```bash
# From YAML config
bench eval create -f benchmarks/tb2-gemini-baseline.yaml

# Inline
bench eval create \
  -t .ref/terminal-bench-2 \
  -a gemini \
  -m gemini-3.1-flash-lite-preview \
  -e daytona \
  -c 64
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config`, `-f` | — | YAML config file |
| `--tasks-dir`, `-t` | — | Task dir (single task with task.toml, or parent of many tasks) |
| `--agent`, `-a` | `gemini` | Agent name |
| `--model`, `-m` | `gemini-3.1-flash-lite-preview` | Model ID |
| `--env`, `-e` | `docker` | Environment: docker or daytona |
| `--concurrency`, `-c` | `4` | Max concurrent tasks (batch mode only) |
| `--jobs-dir`, `-o` | `jobs` | Output directory |
| `--sandbox-user` | `agent` | Sandbox user (null for root) |

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
bench environment create tasks/my-task --backend daytona
```

### bench environment list

List active Daytona sandboxes.

```bash
bench environment list
```

---

## YAML Config Format

### Scene-based (recommended)

```yaml
task_dir: .ref/terminal-bench-2
environment: daytona
concurrency: 64

scenes:
  - name: solve
    roles:
      - name: agent
        agent: gemini
        model: gemini-3.1-flash-lite-preview
    turns:
      - role: agent
```

### Legacy flat (auto-converted)

```yaml
task_dir: .ref/terminal-bench-2
agent: gemini
model: gemini-3.1-flash-lite-preview
environment: daytona
concurrency: 64
max_retries: 2
```

### Multi-scene (BYOS skill generation)

```yaml
task_dir: tasks/
environment: daytona
concurrency: 10

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
| `benchflow run` | `bench eval create -t <task>` |
| `benchflow job` | `bench eval create -f <yaml>` |
| `benchflow agents` | `bench agent list` |
| `benchflow eval` | `bench skills eval` |
| `benchflow metrics` | `bench eval list --detail` |
| `benchflow view` | (planned: `bench trajectory show`) |
| `benchflow cleanup` | `bench environment list` + delete |
| `benchflow skills install` | Skills are folders, not packages |
