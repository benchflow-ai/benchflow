<div align="center">
  <h1>BenchFlow</h1>
  <p>Multi-turn agent benchmarking with ACP</p>
  <a href="https://discord.gg/mZ9Rc8q8W3" target="_blank">
    <img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
</div>

## What

BenchFlow runs AI coding agents against benchmark tasks and captures their full trajectory. It combines [Harbor](https://github.com/benchflow-ai/harbor) (environments, verifier, orchestration) with [ACP](https://agentclientprotocol.com/) (multi-turn agent communication).

The agent runs inside a sandboxed environment (Docker or Daytona). BenchFlow connects to it via ACP over a live stdio pipe. You can send one prompt or many — the agent stays alive between prompts, maintaining full context.

## Install

```bash
pip install benchflow
```

Requires Python 3.12+ and Docker (or a Daytona API key for cloud sandboxes).

## Quick Start

```bash
source .env  # ANTHROPIC_API_KEY (auto-inherited by SDK)

# Run a single task
benchflow run -t path/to/task -a claude-agent-acp -e daytona

# Run a full benchmark (89 tasks, 64 concurrent)
benchflow job -t .ref/terminal-bench-2 -e daytona -c 64

# List available agents
benchflow agents

# View results
benchflow metrics jobs/
benchflow view jobs/my-job/my-trial/
```

## SDK

```python
import asyncio
from benchflow import SDK, Job, JobConfig, collect_metrics

async def main():
    sdk = SDK()

    # Single task — API keys auto-inherited from os.environ
    result = await sdk.run(
        task_path="path/to/task",
        agent="claude-agent-acp",
        model="claude-haiku-4-5-20251001",
        environment="daytona",  # or "docker"
    )
    print(result.rewards)       # {"reward": 1.0}
    print(result.n_tool_calls)  # 17

    # Multi-turn — None = use task's instruction.md
    result = await sdk.run(
        task_path="path/to/task",
        agent="claude-agent-acp",
        prompts=[
            None,
            "Review your solution. Check for errors, test it, and fix any issues.",
        ],
        environment="daytona",
    )

    # Job — run a full benchmark with concurrency and retries
    job = Job(
        tasks_dir="path/to/tasks",
        jobs_dir="jobs/tb2",
        config=JobConfig(
            agent="claude-agent-acp",
            model="claude-haiku-4-5-20251001",
            environment="daytona",
            concurrency=64,
        ),
    )
    result = await job.run()
    print(f"{result.passed}/{result.total} ({result.score:.1%})")

    # Metrics — aggregate results from a jobs directory
    metrics = collect_metrics("jobs/tb2", benchmark="TB2")
    print(metrics.summary())

asyncio.run(main())
```

## CLI

```bash
# Run a single task
benchflow run -t task/ -a claude-agent-acp -m claude-haiku-4-5-20251001 -e daytona

# Run a benchmark job
benchflow job -t tasks/ -a claude-agent-acp -c 64 -e daytona --retries 1

# List agents
benchflow agents

# View metrics
benchflow metrics jobs/tb2/ --json
benchflow metrics jobs/tb2/

# View trajectory
benchflow view jobs/tb2/my-trial/
```

## Agents

Any [ACP-compatible agent](https://agentclientprotocol.com/get-started/agents) works. Registered agents are auto-installed in sandboxes:

| Agent | Description | Status |
|-------|-------------|--------|
| `claude-agent-acp` | Claude Code via ACP | Tested |
| `pi-acp` | Pi coding agent via ACP | Tested |
| `codex-acp` | OpenAI Codex via ACP | Registered |
| `gemini` | Google Gemini CLI via ACP | Registered |
| `openclaw` | OpenClaw via ACP | Incompatible (needs gateway) |

```bash
benchflow run -t task/ -a pi-acp -e daytona
```

## Environments

| Environment | Concurrency | Notes |
|-------------|-------------|-------|
| `docker` | ~4 | Local Docker. Limited by network exhaustion. |
| `daytona` | 64+ | Cloud sandboxes. Requires `DAYTONA_API_KEY`. |

## How it Works

```
benchflow (host)                          Sandbox (Docker/Daytona)
     |                                         |
     |  1. Start environment (Harbor)          |
     |  2. Install ACP agent (npm)             |
     |  3. stdio pipe (exec/SSH) --------> claude-agent-acp
     |                                         |
     |  ACP: initialize                        |
     |  ACP: session/new(cwd) --------------> agent sees workspace, skills
     |  ACP: session/set_model(haiku) ------> model configured
     |  ACP: session/prompt("solve this") --> agent uses Bash, Read, Write
     |  ACP: session/update <---------------- tool calls, messages, thoughts
     |  ACP: session/prompt("test it") -----> same session, full context
     |  ACP: session/update <---------------- more tool calls
     |                                         |
     |  4. Run verifier (Harbor) -----------> tests/test.sh → reward.txt
     |  5. Stop environment                    |
```

## Task Format

Tasks follow the [Harbor task format](https://github.com/benchflow-ai/harbor):

```
my-task/
├── task.toml              # timeouts, resources, metadata
├── instruction.md         # what the agent should do
├── environment/
│   └── Dockerfile         # sandbox setup
├── tests/
│   └── test.sh            # verifier → reward.txt
└── solution/              # optional reference solution
```

## Results

Every run produces structured output:

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

## Benchmark Results

| Benchmark | Model | Score | Reference |
|-----------|-------|-------|-----------|
| TB2 single-turn | Sonnet 4.6 | **58.4%** (52/89) | 59.1% (Anthropic) |
| TB2 multi-turn | Haiku 4.5 | **37.1%** (33/89) | 27.5% (tbench.ai) |

See [docs/parity/RESULTS.md](docs/parity/RESULTS.md) for full analysis.

## Skills

BenchFlow ships skills that teach agents how to use the framework. Place them in `~/.claude/skills/` (or bake into task Dockerfiles) for auto-discovery:

| Skill | Purpose |
|-------|---------|
| `skills/benchflow-run/` | Run benchmarks — SDK, Job, multi-turn, metrics |
| `skills/benchflow-create-task/` | Create Harbor-format benchmark tasks |

To bake skills into a task environment:

```dockerfile
COPY skills /root/.claude/skills
```

Validation tasks in `skills/tests/` confirm agents can use the skills correctly (both eval'd at reward 1.0 with Haiku 4.5).

## Architecture

BenchFlow is a superset of [Harbor](https://github.com/benchflow-ai/harbor). Harbor provides environments (Docker, Daytona), verifier, task format, and orchestration. BenchFlow adds:

- **ACP client** — multi-turn agent communication via live stdio pipe
- **Job orchestration** — concurrency, retries, resume, metrics
- **Multi-agent registry** — auto-install agents in sandboxes
- **Trajectory capture** — from ACP protocol
- **Skills** — teach agents to use BenchFlow itself
- **Viewer** — HTML trajectory visualization
- **CLI** — `run`, `job`, `agents`, `metrics`, `view`

## License

MIT
