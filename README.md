<div align="center">
  <h1>BenchFlow</h1>
  <p>Multi-turn agent benchmarking with ACP</p>
  <a href="https://discord.gg/mZ9Rc8q8W3" target="_blank">
    <img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
</div>

## What

BenchFlow runs AI coding agents against benchmark tasks and captures their full trajectory. It combines [Harbor](https://github.com/benchflow-ai/harbor) (environments, verifier, orchestration) with [ACP](https://agentclientprotocol.com/) (multi-turn agent communication).

The agent runs inside a Docker container. BenchFlow connects to it via ACP over a live stdio pipe. You can send one prompt or many — the agent stays alive between prompts, maintaining full context.

## Install

```bash
pip install benchflow
```

Requires Python 3.12+ and Docker.

## Usage

### SDK

```python
from benchflow.sdk import SDK

sdk = SDK()

# Single-turn
result = await sdk.run(
    task_path="path/to/task",
    agent="claude-agent-acp",
    agent_env={"ANTHROPIC_API_KEY": "..."},
)

# Multi-turn
result = await sdk.run(
    task_path="path/to/task",
    agent="claude-agent-acp",
    prompts=[
        "Set up the database schema",
        "Now write the API endpoints",
        "Add input validation",
    ],
    agent_env={"ANTHROPIC_API_KEY": "..."},
)

print(result.rewards)      # {"reward": 1.0}
print(result.trajectory)   # tool calls, messages, thoughts
print(result.n_tool_calls) # 17
```

### CLI

```bash
# Run a task
benchflow run -t path/to/task -a claude-agent-acp --ae ANTHROPIC_API_KEY=...

# Multi-turn
benchflow run -t task/ -a claude-agent-acp \
  -p "solve the task" \
  -p "now test your solution"

# View trajectory
benchflow view jobs/2026-03-22__20-00-00/extract-elf__abc123/
```

## How it works

```
benchflow (host)                          Docker container
     |                                         |
     |  1. Start container (Harbor)            |
     |  2. Install ACP agent (npm install)     |
     |  3. docker compose exec -i -----> claude-agent-acp
     |                                         |
     |  ACP: initialize                        |
     |  ACP: session/new(cwd=/app) ----------> agent sees /app, skills, settings
     |  ACP: session/prompt("solve this") ---> agent uses Bash, Read, Write, Edit
     |  ACP: session/update <----------------- tool calls, messages, thoughts
     |  ACP: session/prompt("now test it") --> agent continues same session
     |  ACP: session/update <----------------- more tool calls
     |                                         |
     |  4. Run verifier (Harbor) ------------> tests/test.sh → reward.txt
     |  5. Stop container                      |
```

## Agents

Any [ACP-compatible agent](https://agentclientprotocol.com/get-started/agents) works:

```bash
benchflow run -t task/ -a claude-agent-acp    # Claude Code via ACP
benchflow run -t task/ -a "openclaw acp"      # OpenClaw
```

## Task format

Tasks follow the [Harbor task format](https://github.com/benchflow-ai/harbor):

```
my-task/
├── task.toml              # timeouts, resources, metadata
├── instruction.md         # what the agent should do
├── environment/
│   └── Dockerfile         # container setup
├── tests/
│   └── test.sh            # writes 0 or 1 to /logs/verifier/reward.txt
└── solution/              # optional reference solution
    └── solve.sh
```

## Trajectory

Every tool call, message, and thought is captured via ACP `session/update` notifications. View with:

```bash
benchflow view jobs/my-job/my-trial/
```

## Architecture

BenchFlow is a superset of [Harbor](https://github.com/benchflow-ai/harbor). Harbor is imported as a dependency — all of Harbor's environments (Docker, Daytona, E2B, Modal), agents (15+), verifier, orchestrators, metrics, and CLI are available.

BenchFlow adds:
- **ACP client** — multi-turn agent communication via live stdio pipe to container
- **Trajectory capture** — from ACP protocol, HTTP proxy, or OTel
- **Viewer** — HTML trajectory visualization

## License

MIT
