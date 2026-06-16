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
4. Show recent eval results if any exist under `jobs/` (the default `--jobs-dir`)
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
bench eval metrics jobs/      # aggregate pass-rate / tokens / cost (add --json to pipe)
bench eval list jobs/         # per-rollout table
```

### `view <rollout-dir>` — view a trajectory

Results land under `jobs/<job-name>/<rollout-name>/` (the default `--jobs-dir` is `jobs/`):
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
bench tasks init my-task                       # native task.md format (default)
bench tasks init my-task --no-pytest --no-oracle
bench tasks check tasks/my-task                # structural validation
```

Quick structure (native `task.md` format, the default):
```
my-task/
├── task.md            # YAML frontmatter (config) + prompt body
├── environment/
│   └── Dockerfile     # sandbox setup
├── verifier/
│   ├── test.sh        # verifier entrypoint -> writes /logs/verifier/reward.txt
│   └── test_outputs.py
└── oracle/            # optional reference solution (solve.sh)
```

`--format legacy` is retired in v0.6.2: `bench tasks init` always scaffolds a
native `task.md` package. To bring an existing split-layout task forward, run
`bench tasks migrate <dir> --remove-legacy`.

### `skills` — discover and evaluate agent skills

```bash
bench skills list                                   # discover skills on disk
bench skills eval skills/citation-management \
  --agent claude-agent-acp                          # score a skill against its evals/evals.json
```

### `hub` — check external-environment-hub compatibility

```bash
bench hub check          # inventory/structurally-check representative Harbor-registry tasks
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

Compare by running one config per agent (the `agent:` key lives in each YAML)
and printing the aggregate scores:
```python
import asyncio
from benchflow.evaluation import Evaluation

async def main():
    for config_path in [
        "benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml",
        "benchmarks/harvey-lab/harvey-lab-harness-parity.yaml",
    ]:
        result = await Evaluation.from_yaml(config_path).run()
        print(f"{config_path}: {result.passed}/{result.total} ({result.score:.1%})")

asyncio.run(main())
```

---

## Setup

```bash
# Install benchflow 0.6.0 from PyPI (--prerelease allow is for the pinned LiteLLM rc dep):
uv tool install --prerelease allow benchflow
# (or from source: uv sync --extra dev --locked)
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
  --skill-mode with-skill \
  --agent-env BENCHFLOW_SKILL_NUDGE=name
```

`--skill-mode with-skill` is required whenever you pass `--skills-dir` (omitting
it errors). Skills are uploaded to `/skills/` in the sandbox and symlinked to
agent-specific paths.

## Tips

- Use `gemini-3.1-flash-lite-preview` for testing. Use Pro/Sonnet for real benchmarks.
- Evaluations resume — re-running the same `jobs_dir` skips completed tasks.
- `None` in prompts list gets replaced with `instruction.md` content.
- Partial rewards work (verifier can write `0.5` to reward.txt).
- GEMINI_API_KEY requires explicit `--agent-env GEMINI_API_KEY=...` in CLI; SDK auto-inherits from os.environ.
