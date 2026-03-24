# Benchmark Runbook

Run full benchmark suites with benchflow.

## Prerequisites

```bash
source .env  # ANTHROPIC_API_KEY, DAYTONA_API_KEY
pip install -e .
```

## 1. Task Sources

Tasks are in `.ref/`:
- `.ref/terminal-bench-2/` — 89 tasks (TB2)
- `.ref/skillsbench/tasks/` — 87 tasks (SkillsBench)

## 2. Run via YAML Config

```bash
# TB2 single-turn (Haiku, Daytona, concurrency 64)
benchflow job -f examples/configs/tb2-haiku.yaml

# TB2 multi-turn with recheck prompt
benchflow job -f examples/configs/tb2-multiturn.yaml

# SkillsBench
benchflow job -f examples/configs/skillsbench.yaml
```

## 3. Run via SDK

```python
import asyncio
from benchflow import Job, JobConfig

async def main():
    job = Job(
        tasks_dir=".ref/terminal-bench-2",
        jobs_dir="jobs/tb2-haiku",
        config=JobConfig(
            agent="claude-agent-acp",
            model="claude-haiku-4-5-20251001",
            environment="daytona",
            concurrency=64,
        ),
    )
    result = await job.run()
    print(f"{result.passed}/{result.total} ({result.score:.1%})")

asyncio.run(main())
```

Multi-turn:
```python
config=JobConfig(
    agent="claude-agent-acp",
    model="claude-haiku-4-5-20251001",
    environment="daytona",
    concurrency=64,
    prompts=[None, "Review your solution. Check for errors, test it, and fix any issues."],
)
```

## 4. Analyze Results

```bash
benchflow metrics jobs/tb2-haiku/
benchflow metrics jobs/tb2-haiku/ --json
```

Or via SDK:
```python
from benchflow import collect_metrics
metrics = collect_metrics("jobs/tb2-haiku", benchmark="TB2", agent="claude-agent-acp")
print(metrics.summary())
```

## 5. View Trajectories

```bash
benchflow view jobs/tb2-haiku/task-name__abc123/
```

## 6. Expected Costs

| Benchmark | Tasks | Model | Approx Cost |
|-----------|-------|-------|-------------|
| TB2 single-turn | 89 | Haiku 4.5 | ~$5 |
| TB2 multi-turn | 89 | Haiku 4.5 | ~$9 |
| SkillsBench | 87 | Haiku 4.5 | ~$5 |

## 7. Multi-Agent Comparison

```python
# Run same tasks with different agents
for agent in ["claude-agent-acp", "pi-acp"]:
    job = Job(
        tasks_dir=".ref/terminal-bench-2",
        jobs_dir=f"jobs/tb2-{agent}",
        config=JobConfig(agent=agent, environment="daytona", concurrency=64),
    )
    result = await job.run()
```

## 8. Output Structure

```
jobs/{job-name}/
├── {task}__abc123/
│   ├── result.json
│   ├── prompts.json
│   ├── trajectory/acp_trajectory.jsonl
│   └── verifier/reward.txt
└── summary.json
```

## Notes

- API keys are auto-inherited from environment — no need to pass `agent_env`
- Jobs resume automatically — re-running skips completed tasks
- Use `environment="daytona"` for concurrency > 4 (Docker has network exhaustion issues)
- Model is set via ACP `session/set_model`, not env var
