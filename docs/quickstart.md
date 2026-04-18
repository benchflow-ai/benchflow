# Quickstart — Zero to Score in 5 Minutes

Get your first benchmark score in under 5 minutes.

## Prerequisites

- Python 3.12+
- Docker running locally
- An API key for at least one agent (Claude, Codex, or Gemini)

## 1. Install

```bash
uv tool install benchflow
```

Verify:
```
$ benchflow --help

 Usage: benchflow [OPTIONS] COMMAND [ARGS]...

 ACP-native agent benchmarking framework.

╭─ Commands ──────────────────────────────────────────╮
│ agents    List available agents.                    │
│ eval      Evaluate a skill against multiple tasks.  │
│ job       Run all tasks with concurrency + retries. │
│ metrics   Collect and display metrics.              │
│ run       Run a single task with an ACP agent.      │
│ skills    Skill discovery and evaluation.           │
│ tasks     Task authoring commands.                  │
│ view      Serve trajectory viewer in browser.       │
╰─────────────────────────────────────────────────────╯
```

## 2. See available agents

```
$ bench agent list

              Registered Agents
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Name              ┃ Description         ┃ Protocol ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ claude-agent-acp  │ Claude Code via ACP │ acp      │
│ codex-acp         │ OpenAI Codex CLI    │ acp      │
│ gemini            │ Google Gemini CLI   │ acp      │
│ pi-acp            │ Pi agent            │ acp      │
│ openclaw          │ OpenClaw agent      │ acp      │
└───────────────────┴─────────────────────┴──────────┘
```

## 3. Authenticate

Pick your agent — each needs one auth step:

```bash
# Claude — login with subscription (no API key needed)
claude login

# Or use an API key
export ANTHROPIC_API_KEY=sk-ant-...

# Codex
export OPENAI_API_KEY=sk-...

# Gemini
gemini   # follow OAuth flow
```

## 4. Run a single task

```
$ bench run -t tasks/citation-check -a claude-agent-acp -e docker

Task:       citation-check
Agent:      Claude Code (claude-agent-acp)
Rewards:    {'reward': 1.0}
Tool calls: 5
```

What happened:
1. Docker container spun up with the task environment (~30s)
2. Claude Code installed inside the container (~10s)
3. Task instruction sent to the agent via ACP
4. Agent worked (read files, made tool calls, edited code)
5. Verifier ran and scored the result
6. Result saved to `jobs/`

## 5. Run a full benchmark

```
$ bench job -t tasks/ -a claude-agent-acp -e docker -c 4

Score: 32/86 (37.2%), errors=2
```

View aggregate metrics:
```bash
benchflow metrics jobs/
```

## 6. Evaluate a skill

Test whether a skill actually helps agents:

```
$ bench skills eval ./my-skill/ -a claude-agent-acp

Skill eval: my-skill (3 cases)
  Agents: claude-agent-acp
  Environment: docker

              Skill Eval: my-skill
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Agent             ┃ Mode       ┃ Score ┃ Avg Reward ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ claude-agent-acp  │ with-skill │ 3/3   │ 0.92       │
│ claude-agent-acp  │ baseline   │ 1/3   │ 0.35       │
│ claude-agent-acp  │ LIFT       │ +2    │ +0.57      │
└───────────────────┴────────────┴───────┴────────────┘
```

See the [Skill Eval Guide](skill-eval-guide.md) for the full walkthrough.

## 7. View trajectories

See exactly what the agent did:

```bash
benchflow view jobs/citation-check/
```

Opens a browser showing every tool call, message, and thought.

## 8. Scale with Daytona (optional)

For 64+ concurrent tasks, use Daytona cloud sandboxes:

```bash
export DAYTONA_API_KEY=your-key
bench job -t tasks/ -a claude-agent-acp -e daytona -c 64
```

## 9. Use the Python API

```python
import asyncio
import benchflow as bf

agent = bf.Agent("claude-agent-acp", model="claude-haiku-4-5-20251001")
env = bf.Environment.from_task("tasks/my-task", backend="docker")
result = asyncio.run(bf.run(agent, env))

print(f"Reward: {result.reward}")   # 1.0
print(f"Passed: {result.passed}")   # True
```

See the [Runtime API Guide](runtime-guide.md) for full API reference.

## What next?

- **[Skill Eval Guide](skill-eval-guide.md)** — Test whether skills help agents
- **[Runtime API](runtime-guide.md)** — Use benchflow programmatically
- **[Skill Eval Tutorial](examples/skill-eval-tutorial.ipynb)** — Interactive notebook walkthrough
