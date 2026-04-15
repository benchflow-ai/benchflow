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
from benchflow import SDK, Job, JobConfig, collect_metrics

result = await SDK().run(task_path="path/to/task", agent="claude-agent-acp")
print(result.rewards)  # {"reward": 1.0}
```

Single task, multi-turn, full benchmark jobs, and programmatic metrics — see [docs/getting-started.md](docs/getting-started.md).

## CLI

```bash
benchflow run -t path/to/task -a claude-agent-acp   # single task
benchflow job -t tasks/ -a claude-agent-acp -c 1    # benchmark job
benchflow metrics jobs/                              # aggregate results
benchflow view jobs/my-job/my-trial/                # trajectory viewer
```

Full flag reference for all 8 subcommands: [docs/cli-reference.md](docs/cli-reference.md).

## Agents

Any [ACP-compatible agent](https://agentclientprotocol.com/get-started/agents) works. Registered agents are auto-installed in sandboxes.

```bash
benchflow agents              # list registered agents
benchflow run -t task/ -a pi-acp -e daytona
```

See [docs/architecture.md](docs/architecture.md#registry-pattern) for the full tested agent × model/provider matrix and how to add your own.

## Environments

| Environment | Concurrency | Notes |
|-------------|-------------|-------|
| `docker` | ~4 | Local Docker. Limited by network exhaustion. |
| `daytona` | 64+ | Cloud sandboxes. Requires `DAYTONA_API_KEY`. |

## How it Works

BenchFlow starts a sandboxed environment, connects to the agent via ACP over a live stdio pipe, sends one or more prompts (the agent retains full context between turns), then runs the verifier and captures the full trajectory.

See [docs/architecture.md](docs/architecture.md) for SDK run phases, ACP protocol details, and the registry pattern.

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

Full `task.toml` schema, verifier contract, and a worked example: [docs/task-authoring.md](docs/task-authoring.md).

## Results

Every run produces structured output:

```
jobs/{job_name}/{trial_name}/
├── config.json              # SDK.run() parameters (agent, model, environment)
├── result.json              # rewards, agent, timing breakdown
├── timing.json              # {environment_setup, agent_setup, agent_execution, verifier, total}
├── prompts.json             # prompts sent
├── agent/
│   ├── install-stdout.txt   # agent install output
│   └── {agent_name}.txt     # agent stderr/debug output (hyphens → underscores)
├── trajectory/
│   └── acp_trajectory.jsonl # tool calls + agent thoughts
└── verifier/
    └── reward.txt           # reward value
```

## Benchmarks

Tasks are auto-downloaded on first run (cloned into `.ref/`).

**SkillsBench** (86 tasks — tool use, file editing, API calls):

```bash
python benchmarks/run_skillsbench.py benchmarks/skillsbench-claude-glm51.yaml   # Claude
python benchmarks/run_skillsbench.py benchmarks/skillsbench-codex-gpt54.yaml   # Codex
```

**Terminal-Bench 2** (89 tasks — shell, git, compilers, daemons):

```bash
python benchmarks/run_tb2.py benchmarks/tb2_single-codex-gpt54.yaml      # single-turn
python benchmarks/run_tb2.py benchmarks/tb2_multiturn-codex-gpt54.yaml   # multi-turn
```

Shipped configs use `environment: daytona` and `concurrency: 8`. For local Docker: `--env docker --concurrency 1`.

| Benchmark | Agent | Model | Score |
|-----------|-------|-------|-------|
| TB2 single-turn | codex-acp | GPT-5.4* | **69.7%** (62/89) |
| TB2 single-turn | claude-agent-acp | Sonnet 4.6 | **58.4%** (52/89) |
| TB2 multi-turn | codex-acp | GPT-5.4* | **62.9%** (56/89) |
| TB2 multi-turn | claude-agent-acp | Haiku 4.5 | **37.1%** (33/89) |
| SkillsBench | codex-acp | GPT-5.4* | **37.2%** (32/86) |

*GPT-5.4 runs used effort=medium.

## Skills

BenchFlow ships a Claude Code skill in `.claude/skills/benchflow/` that teaches agents how to use the framework. Place skills in `~/.claude/skills/` (or bake into task Dockerfiles) for auto-discovery.

Validation tasks in `.claude/skills/benchflow/tasks/` confirm agents can use the skill correctly.

## Architecture

ACP client, job orchestration, multi-agent registry, trajectory capture, skills, viewer, and CLI — see [docs/architecture.md](docs/architecture.md).

## Citation

If you use BenchFlow in academic work, please cite:

```bibtex
@software{BenchFlow_Team_BenchFlow_2026,
author = {{BenchFlow Team}},
month = mar,
title = {{BenchFlow: Multi-turn agent benchmarking with ACP}},
url = {https://github.com/benchflow-ai/benchflow},
year = {2026}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
