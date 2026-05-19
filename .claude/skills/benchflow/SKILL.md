---
name: benchflow
description: Run agent benchmarks, create tasks, analyze results, and manage agents using BenchFlow. Use when asked to benchmark an AI coding agent, run a benchmark suite, create tasks, view trajectories, or compare agent performance.
user-invocable: true
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
---

# BenchFlow — Agent Benchmarking

BenchFlow runs AI coding agents against tasks in sandboxed environments and
scores their output via ACP (Agent Communication Protocol).

Arguments passed: `$ARGUMENTS`

---

## Dispatch on arguments

### No args or `status` — show current state

1. Check if benchflow is installed: `uv tool list | grep benchflow`
2. Check if API keys are set (GEMINI_API_KEY, ANTHROPIC_API_KEY, etc.)
3. Check available agents: `bench agent list`
4. Show recent eval results if any exist in `evaluations/` or `jobs/`
5. Point to next action based on state

### `run <task-path>` — run a single task

```bash
bench eval create \
  --tasks-dir <task-path> \
  --agent gemini \
  --model gemini-3.1-flash-lite-preview \
  --sandbox daytona
```

Or via Python SDK:
```python
import asyncio
import benchflow as bf
from benchflow import RolloutConfig, Scene
from benchflow._utils.benchmark_repos import resolve_source

async def main():
    config = RolloutConfig(
        task_path=resolve_source("benchflow-ai/skillsbench", path="tasks/edit-pdf"),
        scenes=[Scene.single(agent="gemini", model="gemini-3.1-flash-lite-preview")],
        environment="daytona",
    )
    result = await bf.run(config)
    print(f"Reward: {result.rewards}, Tools: {result.n_tool_calls}")

asyncio.run(main())
```

Note: `resolve_source()` is required for remote repos in the SDK. The CLI
handles this transparently via `--source-repo` / `--source-path`.

API keys are auto-inherited from `os.environ` into the sandbox.

### `eval <tasks-dir>` — run a benchmark suite

```bash
bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks \
  --agent gemini \
  --model gemini-3.1-flash-lite-preview \
  --sandbox daytona \
  --concurrency 64
```

Or via YAML config:
```bash
bench eval create --config benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml
```

YAML format:
```yaml
source:
  repo: benchflow-ai/skillsbench
  path: tasks
agent: gemini
model: gemini-3.1-flash-lite-preview
environment: daytona
concurrency: 64
max_retries: 1
```

### `metrics <jobs-dir>` — analyze results

```bash
bench eval list jobs/
```

### `view <rollout-dir>` — view a trajectory

Results are in `evaluations/<eval-name>/<rollout-name>/` or `jobs/<job-name>/<rollout-name>/`:
```
rollout-dir/
├── result.json              # rewards, agent, timing
├── prompts.json             # prompts sent
├── trajectory/
│   └── acp_trajectory.jsonl # tool calls + agent thoughts
└── verifier/
    ├── reward.txt           # reward value
    └── ctrf.json            # test results
```

### `create-task` — create a new benchmark task

```bash
bench tasks init my-task
bench tasks init my-task --no-pytest --no-solution
```

Quick structure:
```
my-task/
├── task.toml          # timeouts, resources, metadata
├── instruction.md     # what the agent should do
├── environment/
│   └── Dockerfile     # sandbox setup
├── tests/
│   └── test.sh        # verifier -> writes to /logs/verifier/reward.txt
└── solution/          # optional reference solution
```

### `agents` — list available agents

```bash
bench agent list
```

| Agent | Protocol | Auth |
|-------|----------|------|
| `gemini` | ACP | GEMINI_API_KEY or host login |
| `claude-agent-acp` (alias: `claude`) | ACP | ANTHROPIC_API_KEY or host login |
| `codex-acp` (alias: `codex`) | ACP | OPENAI_API_KEY or host login |
| `opencode` | ACP | inferred from model |
| `openhands` (alias: `oh`) | ACP | LLM_API_KEY |
| `harvey-lab-harness` (alias: `harvey-lab`) | ACP | Provider key matching model |

Any agent can be prefixed with `acpx/` to run via ACPX (https://acpx.sh/):
```bash
bench eval create --tasks-dir tasks/edit-pdf --agent acpx/gemini --model gemini-3.1-flash-lite-preview --sandbox daytona
```

ACPX is a headless ACP client with persistent sessions and crash recovery.
The underlying agent's install, env vars, credentials, and skill paths are preserved.

### `compare` — multi-agent comparison

```python
import asyncio
from benchflow.evaluation import Evaluation

async def main():
    for agent_name in ["claude-agent-acp", "gemini", "opencode"]:
        eval_obj = Evaluation.from_yaml("benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml")
        result = await eval_obj.run()
        print(f"{agent_name}: {result.passed}/{result.total} ({result.score:.1%})")

asyncio.run(main())
```

---

## Setup

```bash
uv tool install benchflow    # or: uv sync --extra dev --locked (from source)
export GEMINI_API_KEY=...     # or ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.
export DAYTONA_API_KEY=...    # for cloud sandboxes
```

## Sandboxes

| Sandbox | Flag | Best for |
|---------|------|----------|
| `docker` | `--sandbox docker` | Local dev, small runs (<=10 tasks) |
| `daytona` | `--sandbox daytona` | Cloud runs with concurrency (needs DAYTONA_API_KEY) |
| `modal` | `--sandbox modal` | Serverless, high concurrency (needs Modal auth) |

Use `daytona` for benchmarks. Docker is limited by network exhaustion.

## Skills in tasks

Two approaches for deploying skills:

### Baked into Docker image (existing tasks)
```dockerfile
COPY skills /root/.claude/skills
```

### Runtime deployment via `--skills-dir`
```bash
bench eval create \
  --tasks-dir task-dir \
  --agent claude-agent-acp \
  --sandbox daytona \
  --skills-dir skills/ \
  --agent-env BENCHFLOW_SKILL_NUDGE=name
```

Skills are uploaded to `/skills/` in the sandbox and symlinked to agent-specific paths.

## Tips

- Use `gemini-3.1-flash-lite-preview` for testing. Use Pro/Sonnet for real benchmarks.
- Evaluations resume — re-running the same `jobs_dir` skips completed tasks.
- `None` in prompts list gets replaced with `instruction.md` content.
- Partial rewards work (verifier can write `0.5` to reward.txt).
- GEMINI_API_KEY requires explicit `--agent-env GEMINI_API_KEY=...` in CLI; SDK auto-inherits from os.environ.
