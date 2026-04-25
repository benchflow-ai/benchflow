# Getting started

A 5-minute path from install to first eval.

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- Docker (for local sandboxes) and/or `DAYTONA_API_KEY` (for cloud sandboxes)
- An API key or subscription auth for at least one agent (Anthropic, Gemini, OpenAI, etc.)

## Install

```bash
uv tool install benchflow
```

This gives you the `benchflow` (alias `bench`) CLI plus the Python SDK. To install for editable development:

```bash
git clone https://github.com/benchflow-ai/benchflow
cd benchflow
uv venv -p 3.12 .venv && uv pip install -e ".[dev]"
```

## Run your first eval

```bash
# Single task with Gemini
GEMINI_API_KEY=... bench eval create -t .ref/terminal-bench-2/regex-log -a gemini \
    -m gemini-3.1-pro-preview -e docker

# A whole batch with concurrency
GEMINI_API_KEY=... bench eval create -t .ref/terminal-bench-2 -a gemini \
    -m gemini-3.1-pro-preview -e daytona -c 32

# List the registered agents
bench agent list
```

`bench eval create -t <task>` runs once on a single task or, if the path contains multiple `task.toml`-bearing subdirectories, batches them. Results land under `jobs/<job-name>/<trial-name>/` — `result.json` for the verifier output, `trajectory/acp_trajectory.jsonl` for the full agent trace.

## Run from Python

The CLI is a thin shim over the Python API. For programmatic use:

```python
import benchflow as bf
from benchflow.trial import TrialConfig, Scene
from pathlib import Path

config = TrialConfig(
    task_path=Path(".ref/terminal-bench-2/regex-log"),
    scenes=[Scene.single(agent="gemini", model="gemini-3.1-pro-preview")],
    environment="docker",
)
result = await bf.run(config)
print(result.rewards)         # {'reward': 1.0}
print(result.n_tool_calls)
```

`Trial` is decomposable — invoke each lifecycle phase individually for custom flows. See [Concepts: trial lifecycle](./concepts.md#trial-lifecycle).

## What to read next

| If you want to… | Read |
|------------------|------|
| Understand the model — Trial, Scene, Role, Verifier | [`concepts.md`](./concepts.md) |
| Author a task | [`task-authoring.md`](./task-authoring.md) |
| Run multi-agent patterns (coder/reviewer, simulated user, BYOS) | [`use-cases.md`](./use-cases.md) |
| Run multi-round single-agent (progressive disclosure) | [`progressive-disclosure.md`](./progressive-disclosure.md) |
| Evaluate skills, not tasks | [`skill-eval.md`](./skill-eval.md) |
| Understand the security model | [`sandbox-hardening.md`](./sandbox-hardening.md) |
| CLI flags + commands | [`reference/cli.md`](./reference/cli.md) |
| Python API surface | [`reference/python-api.md`](./reference/python-api.md) |
