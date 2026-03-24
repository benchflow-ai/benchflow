---
name: benchflow-create-task
description: Create benchmark tasks for BenchFlow in the Harbor task format. Use when asked to create a new benchmark task, write a verifier, or set up a task environment.
---

# BenchFlow Create Task Skill

Use this skill to create benchmark tasks that BenchFlow can run. Tasks follow the Harbor format — a directory with an instruction, environment, and verifier.

## When to Use

Activate this skill when the user asks to:
- Create a new benchmark task
- Write a verifier/test for a task
- Set up a task environment (Dockerfile)
- Convert an existing problem into a benchmark task

## Task Structure

```
my-task/
├── task.toml          # required: timeouts and resources
├── instruction.md     # required: what the agent should do
├── environment/
│   └── Dockerfile     # required: sandbox setup
├── tests/
│   └── test.sh        # required: verifier script
└── solution/          # optional: reference solution
    └── solve.sh
```

## Step 1: task.toml

```toml
version = "1.0"

[metadata]
author_name = "Your Name"
difficulty = "medium"           # easy, medium, hard
category = "engineering"
tags = ["python", "testing"]

[verifier]
timeout_sec = 300.0

[agent]
timeout_sec = 600.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 4096
storage_mb = 10240
```

## Step 2: instruction.md

Write a clear, self-contained instruction. The agent sees only this text (plus whatever is in the environment).

Good instructions:
- State the goal clearly in the first sentence
- Specify exact file paths for input and output
- Define what "success" looks like
- Don't assume the agent knows the context

```markdown
Create a Python script at `/app/solve.py` that reads the CSV file at `/app/data.csv`
and outputs a JSON summary to `/app/output.json`.

The summary should contain:
- `total_rows`: number of data rows (excluding header)
- `columns`: list of column names
- `missing_values`: count of empty cells per column
```

## Step 3: Dockerfile

The Dockerfile sets up the sandbox. Keep it minimal — install only what the task needs.

```dockerfile
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy task-specific files
COPY data.csv /app/data.csv
```

Put any task data files in `environment/` alongside the Dockerfile.

## Step 4: Verifier (test.sh)

The verifier runs after the agent finishes. It must write a reward to `/logs/verifier/reward.txt`.

```bash
#!/bin/bash
set -e

REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

# Check if output exists
if [ ! -f /app/output.json ]; then
    echo "0" > "$REWARD_FILE"
    echo "FAIL: /app/output.json not found"
    exit 0
fi

# Validate output with Python
python3 -c "
import json, sys

with open('/app/output.json') as f:
    data = json.load(f)

errors = []
if 'total_rows' not in data:
    errors.append('missing total_rows')
if 'columns' not in data:
    errors.append('missing columns')
if 'missing_values' not in data:
    errors.append('missing missing_values')
if data.get('total_rows') != 100:
    errors.append(f'wrong total_rows: {data.get(\"total_rows\")} != 100')

if errors:
    print('FAIL:', '; '.join(errors))
    sys.exit(1)
print('PASS')
"

if [ $? -eq 0 ]; then
    echo "1" > "$REWARD_FILE"
else
    echo "0" > "$REWARD_FILE"
fi
```

## Step 5: Test Locally

```bash
# Run with benchflow
benchflow run -t my-task/ -a claude-agent-acp -e daytona

# Or with SDK
python -c "
import asyncio
from benchflow import SDK
result = asyncio.run(SDK().run('my-task', agent='claude-agent-acp', environment='daytona'))
print(f'Reward: {result.rewards}, Error: {result.error}')
"
```

## Tips

- **Verifier should be deterministic** — same agent output should always get the same reward.
- **Use partial rewards** for complex tasks — write `0.5` to reward.txt if the agent got halfway.
- **Keep Dockerfiles small** — large images slow down Daytona workspace creation.
- **Test the verifier independently** — run `test.sh` against the reference solution before using it.
- **Put data files in `environment/`** — they get copied into the Docker build context.
- **WORKDIR matters** — the agent starts in whatever WORKDIR the Dockerfile sets (usually `/app` or `/root`).

## Adding Skills to Tasks

To give the agent skills (like in SkillsBench), create a `skills/` directory in `environment/` and add to the Dockerfile:

```dockerfile
COPY skills /root/.claude/skills
```

Each skill is a directory with a `SKILL.md` (instructions) and optional scripts. Claude Code auto-discovers skills in `~/.claude/skills/`.
