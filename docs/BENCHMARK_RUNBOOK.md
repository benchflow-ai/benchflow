# Benchmark Runbook

Instructions for running full benchmark suites with benchflow.

## Prerequisites

```bash
cd /Users/lixiangyi/benchflow/benchflow
export $(cat .env | xargs)  # ANTHROPIC_API_KEY
docker ps  # Docker must be running
```

## 1. Download tasks

### Terminal-Bench 2.0 (89 tasks)

```bash
cd .ref
git clone --depth=1 https://github.com/laude-institute/terminal-bench-2.git
cd ..
```

Tasks at: `.ref/terminal-bench-2/`

### SkillsBench (77 self-contained tasks)

```bash
cd .ref
git clone https://github.com/benchflow-ai/skillsbench.git
cd ..
```

Tasks at: `.ref/skillsbench/tasks/`

Exclude tasks needing external API keys: `scheduling-email-assistant`, `mhc-layer-impl`

## 2. Run Terminal-Bench 2.0

### Single-turn (baseline)

```python
import asyncio, json, os
from pathlib import Path
from benchflow.sdk import SDK

sdk = SDK()
api_key = os.environ["ANTHROPIC_API_KEY"]
tasks_dir = Path(".ref/terminal-bench-2")
task_dirs = sorted([d for d in tasks_dir.iterdir() if d.is_dir() and (d / "task.toml").exists()])

results = []
for task_dir in task_dirs:
    result = await sdk.run(
        task_path=task_dir,
        agent="claude-agent-acp",
        model="claude-haiku-4-5-20251001",
        agent_env={"ANTHROPIC_API_KEY": api_key},
        jobs_dir="parity/terminal-bench-2.0/single-turn",
    )
    results.append({"task": result.task_name, "reward": result.rewards, "error": result.error,
                     "n_tool_calls": result.n_tool_calls, "agent": result.agent_name})
    print(f"{result.task_name}: {result.rewards} {'ERROR: ' + result.error if result.error else ''}")

# Save summary
with open("parity/terminal-bench-2.0/single-turn/summary.json", "w") as f:
    json.dump({"results": results, "total": len(results),
               "solved": sum(1 for r in results if r["reward"] and r["reward"].get("reward") == 1)}, f, indent=2)
```

### Multi-turn with recheck prompt

Same as above but with:
```python
result = await sdk.run(
    task_path=task_dir,
    agent="claude-agent-acp",
    model="claude-haiku-4-5-20251001",
    prompts=[None, "Review your solution. Check for errors, test it, and fix any issues."],
    agent_env={"ANTHROPIC_API_KEY": api_key},
    jobs_dir="parity/terminal-bench-2.0/multi-turn-recheck",
)
```

### Oracle control (should be 89/89)

```python
result = await sdk.run(
    task_path=task_dir,
    agent="claude-agent-acp",  # Not actually used — oracle runs solution/solve.sh
    prompts=["chmod +x /solution/solve.sh && /solution/solve.sh"],
    agent_env={"ANTHROPIC_API_KEY": api_key},
    jobs_dir="parity/terminal-bench-2.0/oracle",
)
```

Note: Oracle doesn't need ACP — it's a shell command. But running through benchflow validates the full pipeline.

## 3. Run SkillsBench

```python
tasks_dir = Path(".ref/skillsbench/tasks")
exclude = {"scheduling-email-assistant", "mhc-layer-impl"}
task_dirs = sorted([d for d in tasks_dir.iterdir()
                    if d.is_dir() and (d / "task.toml").exists() and d.name not in exclude])

for task_dir in task_dirs:
    result = await sdk.run(
        task_path=task_dir,
        agent="claude-agent-acp",
        model="claude-haiku-4-5-20251001",
        agent_env={"ANTHROPIC_API_KEY": api_key},
        jobs_dir="parity/skillsbench",
    )
```

## 4. Expected costs

- Haiku 4.5: ~$0.03-0.05 per task
- Terminal-Bench 89 tasks × 2 runs = ~$9
- SkillsBench 77 tasks = ~$4
- Total: ~$15

## 5. Output

```
parity/
├── terminal-bench-2.0/
│   ├── single-turn/          # 89 trial dirs + summary.json
│   ├── multi-turn-recheck/   # 89 trial dirs + summary.json
│   └── oracle/               # 89 trial dirs + summary.json (control)
├── skillsbench/              # 77 trial dirs + summary.json
└── PARITY.md                 # written analysis comparing results
```

## 6. Analysis

Compare:
- Single-turn vs multi-turn accuracy on Terminal-Bench
- Does the "recheck" prompt help?
- Which task categories benefit most from multi-turn?
- Oracle should be 100% — any failures = framework bug
