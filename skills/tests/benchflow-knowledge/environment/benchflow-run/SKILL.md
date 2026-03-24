---
name: benchflow-run
description: Run agent benchmarks using BenchFlow — single tasks, full benchmark jobs, multi-turn evaluation, and results analysis. Use when asked to benchmark an AI coding agent, run a benchmark suite, or analyze benchmark results.
---

# BenchFlow Run Skill

Use this skill to run agent benchmarks with BenchFlow. BenchFlow evaluates AI coding agents by running them against tasks in sandboxed environments and scoring their output.

## When to Use

Activate this skill when the user asks to:
- Run a benchmark task or suite against an agent
- Compare agents on the same tasks
- Analyze benchmark results or metrics
- Set up a multi-turn evaluation
- Check which agents are available

## Prerequisites

```bash
source .env  # needs ANTHROPIC_API_KEY (and DAYTONA_API_KEY for cloud sandboxes)
pip install -e .  # if running from source
```

API keys are auto-inherited from the environment — no need to pass them explicitly.

## Single Task

```python
import asyncio
from benchflow import SDK

async def main():
    sdk = SDK()
    result = await sdk.run(
        task_path="path/to/task",       # directory with instruction.md + environment/
        agent="claude-agent-acp",        # or "pi-acp"
        model="claude-haiku-4-5-20251001",
        environment="daytona",           # or "docker"
    )
    print(f"Reward: {result.rewards}")
    print(f"Tools: {result.n_tool_calls}")
    print(f"Error: {result.error}")

asyncio.run(main())
```

## Multi-Turn

Send multiple prompts to the same agent session:

```python
result = await sdk.run(
    task_path="path/to/task",
    agent="claude-agent-acp",
    prompts=[
        None,  # None = use task's instruction.md
        "Review your solution. Check for errors, test it, and fix any issues.",
    ],
    environment="daytona",
)
```

## Full Benchmark Job

Run all tasks in a directory with concurrency and retries:

```python
from benchflow import Job, JobConfig, RetryConfig

job = Job(
    tasks_dir="path/to/tasks",        # directory of task subdirectories
    jobs_dir="jobs/my-run",            # output directory
    config=JobConfig(
        agent="claude-agent-acp",
        model="claude-haiku-4-5-20251001",
        environment="daytona",
        concurrency=64,                # parallel tasks
        retry=RetryConfig(max_retries=1),
    ),
)
result = await job.run()
print(f"Score: {result.passed}/{result.total} ({result.score:.1%})")
print(f"Errors: {result.errored}")
```

To run a **subset** of tasks, create a directory with symlinks:

```bash
mkdir -p subset/
for task in task-a task-b task-c; do
    ln -sf "../../full-tasks/$task" "subset/$task"
done
```

## Analyze Results

```python
from benchflow import collect_metrics
import json

metrics = collect_metrics("jobs/my-run", benchmark="TB2", agent="claude-agent-acp")
print(json.dumps(metrics.summary(), indent=2))
```

## CLI

```bash
benchflow run -t path/to/task -a claude-agent-acp -e daytona -m claude-haiku-4-5-20251001
benchflow job -t path/to/tasks -a claude-agent-acp -e daytona -c 64
benchflow agents
benchflow metrics jobs/my-run/
benchflow view jobs/my-run/task__abc123/
```

## Available Agents

| Agent | Description |
|-------|-------------|
| `claude-agent-acp` | Claude Code via ACP (primary, most tested) |
| `pi-acp` | Pi coding agent via ACP (tested, often outperforms on some tasks) |
| `codex-acp` | OpenAI Codex via ACP (needs OPENAI_API_KEY) |
| `gemini` | Google Gemini CLI via ACP (needs GOOGLE_API_KEY) |

## Task Format

Tasks follow the Harbor format:

```
my-task/
├── task.toml          # timeouts, resources
├── instruction.md     # what the agent should do
├── environment/
│   └── Dockerfile     # sandbox setup
├── tests/
│   └── test.sh        # verifier
└── solution/          # optional
```

## Output

Results are saved per trial:

```
jobs/{job_name}/{trial_name}/
├── result.json              # rewards, timing, agent info
├── prompts.json             # prompts sent
├── trajectory/
│   └── acp_trajectory.jsonl # full tool call trace
└── verifier/
    └── reward.txt           # score
```

## Tips

- Use `environment="daytona"` for high concurrency (64+). Docker is limited to ~4.
- Use `model="claude-haiku-4-5-20251001"` for testing. Use Sonnet for actual benchmark numbers.
- Jobs resume automatically — re-running the same `jobs_dir` skips already-completed tasks.
- `None` in the prompts list gets replaced with the task's `instruction.md` content.
