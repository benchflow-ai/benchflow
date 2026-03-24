# Smoke Test Guide

Run these after any code change to verify the system works end-to-end.

## Quick smoke (1 task per agent, ~2 min each)

```python
import asyncio
from benchflow import SDK

async def main():
    sdk = SDK()
    for agent in ["claude-agent-acp", "pi-acp", "openclaw"]:
        result = await sdk.run(
            task_path=".ref/terminal-bench-2/log-summary-date-ranges",
            agent=agent,
            model="claude-haiku-4-5-20251001",
            environment="daytona",
            jobs_dir=f"jobs/smoke-{agent}",
        )
        status = "PASS" if result.rewards and result.rewards.get("reward") == 1.0 else "FAIL"
        print(f"[{agent}] {status} reward={result.rewards} tools={result.n_tool_calls} error={result.error}")

asyncio.run(main())
```

## Full smoke (5 tasks per benchmark type, ~15 min)

```python
import asyncio
from benchflow import Job, JobConfig

TASKS = {
    "tb2": ".ref/terminal-bench-2",
    "skills": ".ref/skillsbench/tasks",
}

async def main():
    for name, tasks_dir in TASKS.items():
        # Create 5-task subset via symlinks
        import os, pathlib
        subset = pathlib.Path(f"jobs/smoke-subset-{name}")
        subset.mkdir(parents=True, exist_ok=True)
        src = pathlib.Path(tasks_dir)
        for task in sorted(src.iterdir())[:5]:
            if task.is_dir() and (task / "task.toml").exists():
                link = subset / task.name
                if not link.exists():
                    link.symlink_to(task.resolve())

        job = Job(
            tasks_dir=str(subset),
            jobs_dir=f"jobs/smoke-{name}",
            config=JobConfig(
                agent="claude-agent-acp",
                model="claude-haiku-4-5-20251001",
                environment="daytona",
                concurrency=64,
            ),
        )
        result = await job.run()
        print(f"[{name}] {result.passed}/{result.total} ({result.score:.1%}) errors={result.errored}")

asyncio.run(main())
```

## Checklist

Every smoke test must verify:
1. **Execution** — reward > 0 on at least one task
2. **Trajectory** — non-empty `acp_trajectory.jsonl` with tool calls
3. **Skills** — for SkillsBench tasks, agent uses skill content (check trajectory for skill references)
4. **Errors** — 0 infra errors (all errors should be model-level, not framework)
5. **Multi-agent** — if testing agents, run same tasks on all agents for comparison

## YAML configs

TB2 single-turn:
```yaml
tasks_dir: .ref/terminal-bench-2
jobs_dir: jobs/tb2-haiku
agent: claude-agent-acp
model: claude-haiku-4-5-20251001
environment: daytona
concurrency: 64
max_retries: 1
```

TB2 multi-turn:
```yaml
tasks_dir: .ref/terminal-bench-2
jobs_dir: jobs/tb2-multiturn
agent: claude-agent-acp
model: claude-haiku-4-5-20251001
environment: daytona
concurrency: 64
max_retries: 1
prompts:
  - null
  - "Review your solution. Check for errors, test it, and fix any issues."
```

SkillsBench:
```yaml
tasks_dir: .ref/skillsbench/tasks
jobs_dir: jobs/skillsbench
agent: claude-agent-acp
model: claude-haiku-4-5-20251001
environment: daytona
concurrency: 64
max_retries: 1
```
