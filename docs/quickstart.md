# Quickstart

Get a benchmark result in under 5 minutes.

## Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- A Daytona API key (`DAYTONA_API_KEY`) for cloud sandboxes
- An agent API key (e.g. `GEMINI_API_KEY` for Gemini)

## Install

```bash
uv tool install benchflow
```

## Run your first evaluation

```bash
# Set credentials
export DAYTONA_API_KEY="dtn_..."
export GEMINI_API_KEY="AIza..."

# Run one TB2 task with Gemini
bench eval create \
  -t .ref/terminal-bench-2/regex-log \
  -a gemini \
  -m gemini-3.1-flash-lite-preview \
  -e daytona \
  --sandbox-setup-timeout 300
```

BenchFlow will:
1. Download Terminal-Bench-2 tasks (first run only)
2. Spin up a Daytona sandbox
3. Install the Gemini CLI agent
4. Send the task instruction via ACP
5. Run the verifier
6. Print the reward (0.0 or 1.0)

## Run a full benchmark

```bash
# 89 TB2 tasks, 64 concurrent
bench eval create -f benchmarks/tb2-gemini-baseline.yaml
```

Example YAML config:
```yaml
task_dir: .ref/terminal-bench-2
agent: gemini
model: gemini-3.1-flash-lite-preview
environment: daytona
concurrency: 64
max_retries: 2
sandbox_setup_timeout: 300
```

## Python API

```python
import benchflow as bf

# One-liner
result = await bf.run("gemini", task_path="tasks/regex-log", model="gemini-3.1-flash-lite-preview")
print(f"reward={result.rewards}")

# With Trial for more control
from benchflow.trial import Trial, TrialConfig, Scene

config = TrialConfig(
    task_path=Path("tasks/regex-log"),
    scenes=[Scene.single(agent="gemini", model="gemini-3.1-flash-lite-preview")],
    environment="daytona",
    sandbox_setup_timeout=300,
)
trial = await Trial.create(config)
result = await trial.run()
```

If you are using the `Agent + Environment` path directly, pass the timeout through `RuntimeConfig`:

```python
from benchflow.runtime import Agent, Environment, Runtime, RuntimeConfig

agent = Agent("gemini", model="gemini-3.1-flash-lite-preview")
env = Environment.from_task("tasks/regex-log", backend="daytona")
runtime = Runtime(env, agent, config=RuntimeConfig(sandbox_setup_timeout=300))
result = await runtime.execute()
```

## Multi-agent (reviewer pattern)

```python
from benchflow.trial import TrialConfig, Scene, Role, Turn

config = TrialConfig(
    task_path=Path("tasks/regex-log"),
    scenes=[
        Scene(name="coder-reviewer",
              roles=[
                  Role("coder", "gemini", "gemini-3.1-flash-lite-preview"),
                  Role("reviewer", "gemini", "gemini-3.1-flash-lite-preview"),
              ],
              turns=[
                  Turn("coder"),
                  Turn("reviewer", "Review the code in /app/. Write feedback to /app/.outbox/coder.json"),
                  Turn("coder", "Read reviewer feedback and fix issues."),
              ]),
    ],
    environment="daytona",
    sandbox_setup_timeout=300,
)
result = await bf.run(config)
```

## Next steps

- [CLI Reference](cli-reference.md) — all commands
- [Task Authoring](task-authoring.md) — create your own tasks
- [API Reference](api-reference.md) — Trial/Scene API details
- [Skill Eval Guide](skill-eval-guide.md) — evaluate agent skills
