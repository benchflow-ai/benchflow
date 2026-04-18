# Runtime API Guide

The 0.3 Runtime API is the primary way to run agent benchmarks programmatically.

## Install

```bash
uv tool install benchflow
```

## Quick Start

```python
import asyncio
import benchflow as bf

agent = bf.Agent("claude-agent-acp", model="claude-haiku-4-5-20251001")
env = bf.Environment.from_task("tasks/my-task", backend="docker")
result = asyncio.run(bf.run(agent, env))

print(f"Reward: {result.reward}")
print(f"Passed: {result.passed}")
print(f"Tool calls: {result.n_tool_calls}")
```

Expected output:
```
Reward: 1.0
Passed: True
Tool calls: 3
```

## Core Types

### Agent

Thin wrapper around a registered agent + model.

```python
# Using a registered agent
agent = bf.Agent("claude-agent-acp", model="claude-haiku-4-5-20251001")

# With custom environment variables
agent = bf.Agent("codex-acp", model="gpt-5.4", env={"OPENAI_API_KEY": "..."})

# Using aliases (resolved via parse_agent_spec)
from benchflow.agents.registry import resolve_agent
config = resolve_agent("claude")  # → claude-agent-acp
config = resolve_agent("codex")   # → codex-acp
```

### Environment

Wraps a Docker or Daytona sandbox. Owns the container lifecycle.

```python
# From a task directory
env = bf.Environment.from_task("tasks/my-task", backend="docker")

# Daytona (cloud sandboxes, faster for batch)
env = bf.Environment.from_task("tasks/my-task", backend="daytona")

# As context manager (auto-cleanup)
async with bf.Environment.from_task("tasks/X", backend="docker") as env:
    result = await bf.run(agent, env)
```

### RuntimeConfig

Controls sandbox, timeouts, skills, and execution policy.

```python
config = bf.RuntimeConfig(
    sandbox_user="agent",           # non-root user (default: "agent")
    timeout=900,                    # max seconds for agent execution
    max_rounds=10,                  # max conversation rounds
    jobs_dir="jobs",                # output directory
    skills_dir="skills/",           # deploy skills into sandbox
    snapshot_policy="none",         # "none", "on_reward", "every_round"
    reward_stream=True,             # emit reward events
)

result = await bf.run(agent, env, config)
```

### RuntimeResult

Artifact-oriented result with structured access to everything.

```python
result = await bf.run(agent, env)

# Core fields
result.reward          # float | None — terminal reward
result.passed          # bool — reward > 0
result.verified        # bool — verifier ran without error
result.n_tool_calls    # int — ACP-sourced tool call count
result.error           # str | None — agent error
result.verifier_error  # str | None — verifier error

# Trajectories and artifacts
result.trajectory      # list[dict] — ACP trajectory events
result.messages        # list[dict] — inter-agent messages (multi-agent)
result.trial_dir       # Path — directory with all artifacts

# Timing
result.started_at      # datetime
result.finished_at     # datetime

# Backward compat
run_result = result.to_run_result()  # → legacy RunResult
```

Guaranteed artifacts in `trial_dir/`:
```
trial_dir/
├── result.json           # reward, timing, error, metadata
├── rewards.jsonl         # terminal + rubric reward events (ORS-compatible)
├── trajectory/
│   └── acp_trajectory.jsonl  # one JSON per ACP event
├── timing.json           # phase-level timing
├── config.json           # run configuration snapshot
└── prompts.json          # prompts sent to agent
```

## Runtime (Explicit Form)

For more control, use `Runtime` directly:

```python
runtime = bf.Runtime(env, agent, config)
result = await runtime.execute()
```

`Runtime.execute()` is the single execution path — everything else
(`bf.run()`, `SDK.run()`, `Job.run()`) delegates to it.

## Execution Phases

```
Runtime.execute()
  │
  ├─ 1. SETUP     — resolve env, create container, write config
  ├─ 2. START     — boot container, upload task files
  ├─ 3. AGENT     — install agent, connect ACP, send prompts
  │                  ├─ initialize() → agent name
  │                  ├─ session_new() → session ID
  │                  ├─ prompt() × N → trajectory events
  │                  └─ close()
  ├─ 4. VERIFY    — run verifier (tests/test.sh), collect rewards
  └─ 5. RESULT    — write artifacts, return RuntimeResult
```

## Multi-Agent (Preview)

```python
# Coming in 0.3 — multi-agent scenes
from benchflow.runtime import Agent, Environment, RuntimeConfig

coder = Agent("claude-agent-acp", model="claude-haiku-4-5-20251001")
reviewer = Agent("gemini", model="gemini-3.1-flash-lite-preview")

# Scene with multiple agents and rounds
# (API still stabilizing — check docs/0.3-plan.rendered.html §A1)
```

## Registered Agents

```bash
$ benchflow agents

  Agent              Protocol   Description
  ─────────────────────────────────────────────
  claude-agent-acp   acp        Claude Code via ACP
  codex-acp          acp        OpenAI Codex CLI
  gemini             acp        Google Gemini CLI
  pi-acp             acp        Pi agent
  openclaw           acp        OpenClaw agent
```

Aliases: `claude` → `claude-agent-acp`, `codex` → `codex-acp`, `gemini` → `gemini`.

## Conformance Task

Every registered agent must pass the ACP conformance smoke test:

```bash
benchflow run -t tests/conformance/acp_smoke -a claude-agent-acp -e docker
```

Expected: reward=1.0, verifier passes, ACP handshake completes.
