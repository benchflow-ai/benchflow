# Benchflow Dogfood Test

**Prompt for another agent to test benchflow end-to-end.**

## Setup

```bash
cd <repo-root>
source .env
pip install -e .
```

## Task 1: Run a single task with the SDK

```python
import asyncio
from benchflow import SDK

async def main():
    sdk = SDK()
    result = await sdk.run(
        task_path=".ref/terminal-bench-2/log-summary-date-ranges",
        agent="claude-agent-acp",
        model="claude-haiku-4-5-20251001",
        prompts=[None],
        jobs_dir="dogfood/single",
        environment="daytona",
    )
    print(f"Reward: {result.rewards}, Tools: {result.n_tool_calls}, Error: {result.error}")

asyncio.run(main())
```

Verify: reward should be `{'reward': 1.0}`, no errors.

Note: `ANTHROPIC_API_KEY` (and `OPENAI_API_KEY`, `GOOGLE_API_KEY`) are auto-inherited from your shell environment. You can override via `agent_env={"ANTHROPIC_API_KEY": "sk-..."}` if needed.

## Task 2: Run a Job with retries and concurrency

Create a 5-task subset directory first, then run as a Job:

```bash
mkdir -p dogfood/tb2-subset
cd dogfood/tb2-subset
for task in log-summary-date-ranges chess-best-move cancel-async-tasks break-filter-js-from-html circuit-fibsqrt; do
    ln -sf "../../.ref/terminal-bench-2/$task" "$task"
done
cd ../..
```

```python
import asyncio, logging
logging.basicConfig(level=logging.INFO)
from benchflow import Job, JobConfig, RetryConfig

async def main():
    job = Job(
        tasks_dir="dogfood/tb2-subset",
        jobs_dir="dogfood/job-test",
        config=JobConfig(
            agent="claude-agent-acp",
            model="claude-haiku-4-5-20251001",
            environment="daytona",
            concurrency=2,
            retry=RetryConfig(max_retries=1),
        ),
    )
    result = await job.run()
    print(f"Score: {result.passed}/{result.total} ({result.score:.1%})")
    print(f"Errors: {result.errored}")

asyncio.run(main())
```

Note: `Job(tasks_dir=...)` runs ALL tasks in the directory. To run a subset, create a directory with symlinks to just the tasks you want.

## Task 3: Collect and analyze metrics

```python
from benchflow import collect_metrics
import json

metrics = collect_metrics("dogfood/job-test", benchmark="TB2", agent="claude-agent-acp", model="haiku-4.5")
print(json.dumps(metrics.summary(), indent=2))
```

Verify: the summary should show pass/fail/error counts, avg tool calls, error breakdown.

## Task 4: List available agents

```python
from benchflow import list_agents
for agent in list_agents():
    print(f"{agent.name}: {agent.description} (requires: {agent.requires_env})")
```

## Task 5: Check the viewer

After running tasks, view a trajectory:

```bash
benchflow view dogfood/single/
```

This should open an HTML viewer at localhost.

## What to report

1. Did each task succeed? If not, what error?
2. Does the Job resume correctly if you run it twice?
3. Are result.json files written to the correct paths?
4. Does `collect_metrics` produce accurate numbers?
5. Any bugs, confusing APIs, or missing docs?
