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
scores their output. It combines Harbor (environments, verifier) with ACP
(multi-turn agent communication).

Arguments passed: `$ARGUMENTS`

---

## Dispatch on arguments

### No args or `status` — show current state

1. Check if benchflow is installed: `pip show benchflow`
2. Check if `.env` exists with API keys
3. Check available agents: `benchflow agents`
4. Show recent job results if any exist in `jobs/`
5. Point to next action based on state

### `run <task-path>` — run a single task

```bash
source .env
benchflow run -t <task-path> -a claude-agent-acp -e daytona -m claude-haiku-4-5-20251001
```

Or via SDK:
```python
import asyncio
from benchflow import SDK

async def main():
    sdk = SDK()
    result = await sdk.run(
        task_path="<task-path>",
        agent="claude-agent-acp",
        model="claude-haiku-4-5-20251001",
        environment="daytona",
    )
    print(f"Reward: {result.rewards}, Tools: {result.n_tool_calls}")

asyncio.run(main())
```

API keys are auto-inherited from `os.environ`. No need to pass `agent_env`.

### `job <tasks-dir>` — run a benchmark suite

```bash
benchflow job -t <tasks-dir> -a claude-agent-acp -e daytona -c 64
```

Or via YAML config:
```bash
benchflow job -f examples/configs/tb2-haiku.yaml
```

YAML format (benchflow-native):
```yaml
tasks_dir: .ref/terminal-bench-2
jobs_dir: jobs/tb2-haiku
agent: claude-agent-acp
model: claude-haiku-4-5-20251001
environment: daytona
concurrency: 64
max_retries: 1
```

Harbor-compatible YAML also works:
```yaml
jobs_dir: jobs
n_attempts: 2
orchestrator:
  n_concurrent_trials: 8
environment:
  type: daytona
agents:
  - name: claude-agent-acp
    model_name: anthropic/claude-haiku-4-5-20251001
datasets:
  - path: .ref/terminal-bench-2
```

Multi-turn (adds a recheck prompt):
```yaml
tasks_dir: .ref/terminal-bench-2
jobs_dir: jobs/tb2-multiturn
agent: claude-agent-acp
model: claude-haiku-4-5-20251001
environment: daytona
concurrency: 64
prompts:
  - null  # uses instruction.md
  - "Review your solution. Check for errors, test it, and fix any issues."
```

### `metrics <jobs-dir>` — analyze results

```bash
benchflow metrics jobs/tb2-haiku/
benchflow metrics jobs/tb2-haiku/ --json
```

SDK:
```python
from benchflow import collect_metrics
metrics = collect_metrics("jobs/tb2-haiku", benchmark="TB2", agent="claude-agent-acp")
print(metrics.summary())
```

### `view <trial-dir>` — view a trajectory

```bash
benchflow view jobs/tb2-haiku/<trial-name>/
```

Opens HTML viewer at `http://localhost:8888`.

### `create-task` — create a new benchmark task

See `skills/benchflow/references/create-task.md` for the full guide.

### `smoke-test` — verify system works

See `skills/benchflow/references/smoke-test.md` for quick and full smoke tests.

### `dogfood` — end-to-end testing prompt

See `skills/benchflow/references/dogfood.md` for the full dogfood script.

Quick structure:
```
my-task/
├── task.toml          # timeouts, resources, metadata
├── instruction.md     # what the agent should do
├── environment/
│   └── Dockerfile     # sandbox setup
├── tests/
│   └── test.sh        # verifier → writes to /logs/verifier/reward.txt
└── solution/          # optional reference solution
```

### `agents` — list available agents

```bash
benchflow agents
```

| Agent | Status | Skills |
|-------|--------|--------|
| `claude-agent-acp` | Working | `~/.claude/skills/` |
| `pi-acp` | Working | `~/.claude/skills/` |
| `openclaw` | Working (via shim) | copies to `<workspace>/skills/` |
| `codex-acp` | Registered | needs OPENAI_API_KEY |
| `gemini` | Registered | needs GOOGLE_API_KEY |

### `compare` — multi-agent comparison

```python
import asyncio
from benchflow import Job, JobConfig

async def main():
    for agent in ["claude-agent-acp", "pi-acp", "openclaw"]:
        job = Job(
            tasks_dir="path/to/tasks",
            jobs_dir=f"jobs/compare-{agent}",
            config=JobConfig(agent=agent, environment="daytona", concurrency=64),
        )
        result = await job.run()
        print(f"{agent}: {result.passed}/{result.total} ({result.score:.1%})")

asyncio.run(main())
```

---

## Setup

```bash
pip install benchflow    # or: pip install -e . (from source)
source .env              # ANTHROPIC_API_KEY, DAYTONA_API_KEY
```

## Environments

| Environment | Concurrency | Setup |
|-------------|-------------|-------|
| `daytona` | 64+ | Set `DAYTONA_API_KEY` in `.env` |
| `docker` | ~4 | Docker must be running locally |

Use `daytona` for benchmarks. Docker is limited by network exhaustion.

## Skills in tasks

SkillsBench tasks bake skills into Docker images:
```dockerfile
COPY skills /root/.claude/skills
```

- `claude-agent-acp` / `pi-acp`: auto-discover `~/.claude/skills/`
- `openclaw`: shim copies from `.claude/skills/` → `<workspace>/skills/`
- Skills must load from the environment, never injected into prompts

## Output structure

```
jobs/{job_name}/{trial_name}/
├── result.json              # rewards, agent, timing
├── prompts.json             # prompts sent
├── trajectory/
│   └── acp_trajectory.jsonl # tool calls + agent thoughts
└── verifier/
    ├── reward.txt           # reward value
    └── ctrf.json            # test results
```

## Tips

- Use `claude-haiku-4-5-20251001` for testing. Use Sonnet for real benchmarks.
- Jobs resume — re-running the same `jobs_dir` skips completed tasks.
- `None` in prompts list gets replaced with `instruction.md` content.
- Partial rewards work (verifier can write `0.5` to reward.txt).
