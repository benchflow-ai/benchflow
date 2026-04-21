<div align="center">
  <h1>BenchFlow</h1>
  <p>Multi-turn agent benchmarking — Scene-based lifecycle for any ACP agent</p>
  <a href="https://pypi.org/project/benchflow/" target="_blank">
    <img src="https://img.shields.io/badge/PyPI-0.3.0a3-blue?style=for-the-badge&logo=pypi" alt="PyPI">
  </a>
  <a href="https://discord.gg/mZ9Rc8q8W3" target="_blank">
    <img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
</div>

## What

BenchFlow runs AI agents against benchmark tasks in sandboxed environments. It supports single-agent, multi-agent, and multi-turn evaluation patterns through a Scene-based lifecycle.

- **Any ACP agent** — Gemini CLI, Claude, Codex, OpenClaw, Pi, or your own
- **Multi-scene trials** — skill generation → solve, coder → reviewer → revision
- **Cloud sandboxes** — Daytona backend for parallel execution at scale
- **YAML-driven** — same task folder, different trial configs for ablation

## Install

```bash
pip install benchflow==0.3.0a3
```

Requires Python 3.12+. For cloud sandboxes, set `DAYTONA_API_KEY`.

## Quick Start

### CLI

```bash
# Run a single task with Gemini
bench eval create -t tasks/my-task -a gemini -m gemini-3.1-flash-lite-preview -e daytona

# Run from YAML config (batch, concurrent)
bench eval create -f benchmarks/tb2-gemini-baseline.yaml

# List agents
bench agent list

# Check task validity
bench tasks check tasks/my-task
```

### Python

```python
import benchflow as bf
from benchflow.trial import TrialConfig, Scene, Role, Turn

# Simplest: one agent, one task
result = await bf.run("gemini", task_path="tasks/my-task", model="gemini-3.1-flash-lite-preview")
print(result.rewards)  # {"reward": 1.0}

# Scene-based: skill-gen → solve (BYOS pattern)
config = TrialConfig(
    task_path=Path("tasks/my-task"),
    scenes=[
        Scene(name="skill-gen",
              roles=[Role("gen", "gemini", "gemini-3.1-flash-lite-preview")],
              turns=[Turn("gen", "Analyze the task and write a skill to /app/generated-skill.md")]),
        Scene(name="solve",
              roles=[Role("solver", "gemini", "gemini-3.1-flash-lite-preview")],
              turns=[Turn("solver")]),  # None prompt = use instruction.md
    ],
    environment="daytona",
)
result = await bf.run(config)

# Multi-agent: coder + reviewer
config = TrialConfig(
    task_path=Path("tasks/my-task"),
    scenes=[
        Scene(name="review-loop",
              roles=[
                  Role("coder", "gemini", "gemini-3.1-flash-lite-preview"),
                  Role("reviewer", "gemini", "gemini-3.1-flash-lite-preview"),
              ],
              turns=[
                  Turn("coder", "Solve the task. Write to /app/.outbox/reviewer.json when done."),
                  Turn("reviewer", "Review the coder's work. Write feedback to /app/.outbox/coder.json."),
                  Turn("coder", "Read the reviewer's feedback and revise your solution."),
              ]),
    ],
    environment="daytona",
)
result = await bf.run(config)
```

### YAML Trial Config

```yaml
# trial-baseline.yaml
task_dir: .ref/terminal-bench-2
agent: gemini
model: gemini-3.1-flash-lite-preview
environment: daytona
concurrency: 89

# trial-byos.yaml (same tasks, different config)
task_dir: .ref/terminal-bench-2
scenes:
  - name: skill-gen
    roles: [{name: gen, agent: gemini, model: gemini-3.1-flash-lite-preview}]
    turns: [{role: gen, prompt: "Generate a skill for this task..."}]
  - name: solve
    roles: [{name: solver, agent: gemini, model: gemini-3.1-flash-lite-preview}]
```

## CLI Reference

```
bench agent list              List registered agents
bench agent show <name>       Agent details + conformance status

bench eval create             Create + run evaluation (returns job-id)
bench eval list               List completed evaluations

bench skills eval             Evaluate skill via evals.json

bench tasks init <name>       Scaffold new task
bench tasks check <dir>       Validate task (--rubric for custom)

bench train create            Reward-based training sweep

bench environment create      Spin up sandbox from task dir
bench environment list        List active sandboxes
```

## Architecture

```
Trial = sequence of Scenes in a shared sandbox
Scene = Roles + Turns (one interaction region)
Role  = agent + model
Turn  = one prompt for one role

bf.run(config)
  → Trial.create(config)
    → trial.setup()      # resolve config, create env object
    → trial.start()      # spin up sandbox, upload task files
    → for scene in config.scenes:
        → trial._run_scene(scene)  # connect/execute/disconnect per role
    → trial.verify()     # run verifier, score
    → trial.cleanup()    # stop sandbox
```

## Registered Agents

| Agent | Command | Auth |
|-------|---------|------|
| `gemini` | `gemini --acp --yolo` | GOOGLE_API_KEY |
| `claude-agent-acp` | `claude-agent-acp` | ANTHROPIC_API_KEY |
| `codex-acp` | `codex-acp` | OPENAI_API_KEY |
| `openclaw` | `openclaw-acp-shim` | inferred from model |
| `pi-acp` | `pi-acp` | ANTHROPIC_API_KEY |

## Adding a Custom Agent

Any ACP-native agent works. Create `agent.toml`:

```toml
name = "my-agent"
launch_cmd = "my-agent --acp"
install_cmd = "npm install -g my-agent"
requires_env = ["MY_API_KEY"]
```

## Development

```bash
uv venv -p 3.12 .venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest tests/       # 580+ unit tests
.venv/bin/ty check src/                 # type check
```
