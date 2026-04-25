# Runtime API Guide

The Trial/Scene API is the primary way to run agent benchmarks programmatically.

## Install

```bash
pip install benchflow==0.3.0a3
```

## Quick Start

```python
import asyncio
import benchflow as bf

result = asyncio.run(bf.run("gemini", task_path="tasks/my-task", model="gemini-3.1-flash-lite-preview"))

print(f"Reward: {result.rewards}")
print(f"Tool calls: {result.n_tool_calls}")
```

## Core Types

### TrialConfig

Declarative configuration for a trial — a sequence of Scenes in a shared sandbox.

```python
from benchflow.trial import TrialConfig, Scene, Role, Turn

# Single-agent (simplest)
config = TrialConfig(
    task_path=Path("tasks/my-task"),
    scenes=[Scene.single(agent="gemini", model="gemini-3.1-flash-lite-preview")],
    environment="daytona",
)

# Multi-scene BYOS (skill-gen → solve)
config = TrialConfig(
    task_path=Path("tasks/my-task"),
    scenes=[
        Scene(name="prep", roles=[Role("gen", "gemini", "gemini-3.1-flash-lite-preview")],
              turns=[Turn("gen", "Generate a skill for this task...")]),
        Scene(name="solve", roles=[Role("solver", "gemini", "gemini-3.1-flash-lite-preview")],
              turns=[Turn("solver")]),
    ],
    environment="daytona",
)
```

### Scene

One interaction region — roles take turns executing prompts.

```python
# Single-role shortcut
scene = Scene.single(agent="gemini", model="gemini-3.1-flash-lite-preview")

# Multi-role with turn order (coder-reviewer pattern)
# Agents communicate via outbox: write /app/.outbox/{recipient}.json
# Scheduler reads outbox after each turn, injects into next role's prompt
scene = Scene(
    name="coder-reviewer",
    roles=[
        Role("coder", "gemini", "gemini-3.1-flash-lite-preview"),
        Role("reviewer", "gemini", "gemini-3.1-flash-lite-preview"),
    ],
    turns=[
        Turn("coder"),                    # None prompt = instruction.md
        Turn("reviewer", "Review the code. Write feedback to "
             '/app/.outbox/coder.json as {"to":"coder","content":"..."}'),
        Turn("coder", "Fix the issues."), # reviewer's feedback auto-injected
    ],
)
```

### Trial

The execution engine — decomposed into independently-callable phases.

```python
from benchflow.trial import Trial

trial = await Trial.create(config)

# Full lifecycle (most common)
result = await trial.run()

# Manual composition (for custom flows)
await trial.setup()
await trial.start()
await trial.install_agent()
await trial.connect()
await trial.execute(prompts=["custom prompt"])
await trial.disconnect()
await trial.verify()
await trial.cleanup()
```

### bf.run()

Convenience function — multiple calling conventions:

```python
import benchflow as bf

# 1. TrialConfig (full control)
result = await bf.run(config)

# 2. Agent + Environment (0.3 style)
agent = bf.Agent("gemini", model="gemini-3.1-flash-lite-preview")
env = bf.Environment.from_task("tasks/X", backend="daytona")
result = await bf.run(agent, env)

# 3. String shortcut (simplest)
result = await bf.run("gemini", task_path="tasks/X", model="gemini-3.1-flash-lite-preview")
```

## Trial Lifecycle

```
Trial.run()
  │
  ├─ setup()          — resolve config, create env object
  ├─ start()          — spin up sandbox, upload task files, start services
  ├─ install_agent()  — install agent binary, credentials, sandbox user
  │                    (sandbox user setup: create non-root user, prepare
  │                     small config/auth dirs, chown the workspace — no
  │                     recursive copy of /root tool trees; agent binaries
  │                     must live on shared prefixes like /usr/local/bin)
  ├─ for scene in scenes:
  │    └─ _run_scene(scene)
  │         ├─ setup /app/.outbox/ — (multi-role scenes only)
  │         └─ for turn in scene.turns:
  │              ├─ read outbox     — inject messages into prompt
  │              ├─ connect_as(role) — open ACP session for this role
  │              ├─ execute(prompts) — send prompts, collect trajectory
  │              └─ disconnect()    — kill agent process, clean up
  ├─ verify()         — run verifier, collect rewards
  └─ cleanup()        — stop sandbox
```

Key: `disconnect()` kills the agent process between scenes to prevent context bleed. Each scene gets a fresh agent session.

## Multi-Turn vs Multi-Round

| Pattern | Roles | Turns | Communication | Example |
|---------|-------|-------|---------------|---------|
| **Single-turn** | 1 | 1 | — | Baseline benchmark |
| **Multi-turn** | 1 | 2+ | Same session, sequential prompts | Self-review |
| **Multi-round** | 2+ | 2+ | Outbox files between roles | Coder + Reviewer |

**Multi-turn** = same agent gets multiple prompts. Use when a second pass catches errors (self-review, iterative refinement). The agent keeps its context across turns.

**Multi-round** = different agents exchange turns. Use when tasks need multiple perspectives (code review, client-advisor). The scheduler reads outbox files and injects messages.

Both use the same API — `TrialConfig` with different `Scene` configurations.

## Multi-Agent Patterns

### Coder + Reviewer (followup-bench)

```python
config = TrialConfig(
    task_path=task_path,
    scenes=[Scene(
        roles=[Role("coder", "gemini", "flash"), Role("reviewer", "gemini", "flash")],
        turns=[
            Turn("coder"),
            Turn("reviewer", "Review /app/. Write feedback to /app/.outbox/coder.json"),
            Turn("coder", "Read feedback and fix."),
        ],
    )],
    environment="daytona",
)
```

### Skill Generation + Solve (BYOS)

```python
config = TrialConfig(
    task_path=task_path,
    scenes=[
        Scene(name="skill-gen",
              roles=[Role("gen", "gemini", "flash")],
              turns=[Turn("gen", "Generate a skill document to /app/generated-skill.md")]),
        Scene(name="solve",
              roles=[Role("solver", "gemini", "flash")],
              turns=[Turn("solver")]),
    ],
    environment="daytona",
)
```

## 0.3 Limitations

The Scene API in 0.3 covers coder-reviewer and multi-turn patterns. It does **not** yet support:

- **Dynamic termination** — turn count is fixed at config time. A "user" role cannot decide to stop early based on agent output. Workaround: use `max_rounds` in the standalone `_scene.py` scheduler.
- **Oracle access** — no mechanism for a "user" role to read `/solution` during setup.
- **Per-round verification** — `verify()` runs once after all scenes complete, not between rounds.
- **Inter-round trajectory inspection** — a "user" role cannot read the agent's trajectory between turns.

These are tracked for 0.4. See the [Harbor PR #1462 mapping](docs/notebooks/scene-patterns.ipynb) for details.

## YAML Trial Configs

```python
from benchflow.trial_yaml import trial_config_from_yaml

config = trial_config_from_yaml("trial.yaml")
result = await bf.run(config)
```

## Registered Agents

| Agent | Protocol | Auth | Aliases |
|-------|----------|------|---------|
| `gemini` | ACP | GOOGLE_API_KEY | — |
| `claude-agent-acp` | ACP | ANTHROPIC_API_KEY | `claude` |
| `codex-acp` | ACP | OPENAI_API_KEY | `codex` |
| `pi-acp` | ACP | ANTHROPIC_API_KEY | `pi` |
| `openclaw` | ACP | inferred from model | — |

## Retry and Error Handling

Trial.run() catches common errors:
- `TimeoutError` — agent exceeded timeout
- `ConnectionError` — SSH/ACP pipe closed (retried 3x with exponential backoff)
- `ACPError` — agent protocol error

Job-level retry with `RetryConfig`:
```python
from benchflow.job import Job, JobConfig, RetryConfig

config = JobConfig(
    retry=RetryConfig(
        max_retries=2,
        wait_multiplier=2.0,
        min_wait_sec=1.0,
        max_wait_sec=30.0,
    ),
)
```
