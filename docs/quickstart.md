# Quickstart — Zero to Score in 5 Minutes

Get your first benchmark score in under 5 minutes.

## Prerequisites

- Python 3.12+
- Docker running locally
- An API key for at least one agent (Claude, Codex, or Gemini)

## 1. Install benchflow

```bash
# Recommended: install as a CLI tool
uv tool install benchflow

# Or with pip
pip install benchflow
```

## 2. Authenticate and run

Pick your agent — each one needs just one auth step:

### Claude Code (Anthropic)

```bash
# Login with your Claude subscription (no API key needed)
claude login

# Run a task
benchflow run -t .ref/skillsbench/tasks/citation-check -a claude-agent-acp -e docker
```

Or use an API key: `export ANTHROPIC_API_KEY=sk-ant-...`

### Codex (OpenAI)

```bash
# Login with your OpenAI subscription (no API key needed)
codex --login

# Run with GPT-5.4
benchflow run -t .ref/skillsbench/tasks/citation-check -a codex-acp -m gpt-5.4 -e docker
```

Or use an API key: `export OPENAI_API_KEY=sk-...`

### Gemini (Google)

```bash
# Login with Google account
gemini   # follow OAuth flow

# Run with Gemini
benchflow run -t .ref/skillsbench/tasks/citation-check -a gemini -m gemini-3-flash-preview -e docker
```

Or use an API key: `export GEMINI_API_KEY=...`

### OpenClaw (any model, any provider)

```bash
# Works with any provider — just set the right key
export OPENAI_API_KEY=sk-...
benchflow run -t .ref/skillsbench/tasks/citation-check -a openclaw -m gpt-5.4 -e docker

# Or with Gemini on Vertex
export GOOGLE_CLOUD_PROJECT=my-project
benchflow run -t .ref/skillsbench/tasks/citation-check -a openclaw -m gemini-3-flash-preview -e docker
```

## 3. What happens when you run

benchflow downloads the task on first run, then:

This will:
1. Pull the task's Docker image (~30s)
2. Install Claude Code inside the container (~10s)
3. Send the task instruction to the agent
4. Agent works on the task (tool calls, file edits, etc.)
5. Run the verifier to score the result
6. Print the reward

```
Task:       citation-check
Agent:      Claude Code (claude-agent-acp)
Rewards:    {'reward': 1.0}
Tool calls: 5
```

## 4. View the trajectory

See exactly what the agent did:

```bash
benchflow view jobs/citation-check/
```

Opens a browser at `localhost:8888` showing every tool call, message, and
thought the agent had.

## 5. Run a full benchmark

Run all SkillsBench tasks with concurrency:

```bash
benchflow job \
  -t .ref/skillsbench/tasks \
  -a claude-agent-acp \
  -e docker \
  -c 4
```

View aggregate results:

```bash
benchflow metrics jobs/
```

```
Score: 32/86 (37.2%), errors=2
```

## 6. Try a different agent

```bash
# Codex
benchflow run -t .ref/skillsbench/tasks/citation-check -a codex-acp -e docker

# Gemini
benchflow run -t .ref/skillsbench/tasks/citation-check -a gemini -e docker

# List all agents
benchflow agents
```

## 7. Run at scale with Daytona (optional)

For 64+ concurrent tasks, use Daytona cloud sandboxes:

```bash
export DAYTONA_API_KEY=your-key

benchflow job \
  -t .ref/skillsbench/tasks \
  -a claude-agent-acp \
  -e daytona \
  -c 64
```

## What next?

- **[Evaluate a skill](skill-eval-guide.md)** — Test whether a skill
  actually helps agents
- **[Create your own tasks](create-tasks.md)** — Build custom benchmark
  tasks for your domain
- **[SDK Reference](sdk-reference.md)** — Use benchflow programmatically
- **[Security](harden-sandbox.md)** — How benchflow protects against
  reward hacking
