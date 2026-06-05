# Python API
The Rollout/Scene API is the primary way to run agent benchmarks programmatically.

## Install

```bash
uv tool install --prerelease allow 'benchflow==0.5.2'
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

### RolloutConfig

Declarative configuration for a rollout — a sequence of Scenes in a shared sandbox.

```python
from pathlib import Path
from benchflow import RolloutConfig, Scene, Role, Turn

# Single-agent (simplest)
config = RolloutConfig(
    task_path=Path("tasks/my-task"),
    scenes=[Scene.single(agent="gemini", model="gemini-3.1-flash-lite-preview")],
    environment="daytona",
    sandbox_setup_timeout=120,
)

# Multi-scene BYOS (skill-gen → solve)
config = RolloutConfig(
    task_path=Path("tasks/my-task"),
    scenes=[
        Scene(name="prep", roles=[Role("gen", "gemini", "gemini-3.1-flash-lite-preview")],
              turns=[Turn("gen", "Generate a skill for this task...")]),
        Scene(name="solve", roles=[Role("solver", "gemini", "gemini-3.1-flash-lite-preview")],
              turns=[Turn("solver")]),
    ],
    environment="daytona",
    sandbox_setup_timeout=120,
)
```

Set `sandbox_setup_timeout` when sandbox user setup needs more than the default 120 seconds.
The same field is also available on `JobConfig` and `RuntimeConfig`.

### Scene

Authoring sugar for role, prompt, and skill attribution. Scenes compile to
explicit rollout Steps before execution; there is no runtime Scene object or
message scheduler.

```python
# Single-role shortcut
scene = Scene.single(agent="gemini", model="gemini-3.1-flash-lite-preview")

# Multi-role with explicit turn order
scene = Scene(
    name="coder-reviewer",
    roles=[
        Role("coder", "gemini", "gemini-3.1-flash-lite-preview"),
        Role("reviewer", "gemini", "gemini-3.1-flash-lite-preview"),
    ],
    turns=[
        Turn("coder"),                    # None prompt = instruction.md
        Turn("reviewer", "Review the current workspace."),
        Turn("coder", "Fix the issues."),
    ],
)
```

### Rollout

The execution engine — decomposed into independently-callable phases.

```python
from benchflow import Rollout

rollout = await Rollout.create(config)

# Full lifecycle (most common)
result = await rollout.run()

# Manual composition (for custom flows)
await rollout.setup()
await rollout.start()
await rollout.install_agent()
await rollout.connect()
await rollout.execute(prompts=["custom prompt"])
await rollout.disconnect()
await rollout.verify()
await rollout.cleanup()
```

### RuntimeConfig

Runtime-level configuration for the `Agent + Environment` execution path.

```python
from benchflow.runtime import Agent, Environment, Runtime, RuntimeConfig

config = RuntimeConfig(sandbox_setup_timeout=300)
agent = Agent("gemini", model="gemini-3.1-flash-lite-preview")
env = Environment.from_task("tasks/X", sandbox="daytona")
runtime = Runtime(env, agent, config=config)
result = await runtime.execute()
```

### bf.run()

Convenience function — multiple calling conventions:

```python
import benchflow as bf

# 1. RolloutConfig (full control)
result = await bf.run(config)

# 2. Agent + Environment (0.3 style)
agent = bf.Agent("gemini", model="gemini-3.1-flash-lite-preview")
env = bf.Environment.from_task("tasks/X", sandbox="daytona")
runtime_config = bf.RuntimeConfig(sandbox_setup_timeout=300)
result = await bf.run(agent, env, runtime_config)

# 3. String shortcut (simplest)
result = await bf.run(
    "gemini",
    task_path="tasks/X",
    model="gemini-3.1-flash-lite-preview",
    config=bf.RuntimeConfig(sandbox_setup_timeout=300),
)
```

## Rollout Lifecycle

```
Rollout.run()
  │
  ├─ setup()          — resolve config, create env object
  ├─ start()          — spin up sandbox, upload task files, start services
  ├─ install_agent()  — install agent binary, credentials, sandbox user
  │                    (sandbox user setup: create non-root user, prepare
  │                     small config/auth dirs, chown the workspace — no
  │                     recursive copy of /root tool trees; agent binaries
  │                     must live on shared prefixes like /usr/local/bin)
  ├─ compile scenes → Steps
  ├─ for step in steps:
  │    ├─ connect_as(role) — open/reuse ACP session for this role
  │    └─ execute(prompt)  — send prompt, collect trajectory, grow tree
  ├─ verify()         — run verifier, collect rewards
  └─ cleanup()        — stop sandbox
```

Key: scene boundaries are gone by execution time; role changes are represented
as Step metadata and handled by the rollout executor.

## Multi-Turn vs Multi-Round

| Pattern | Roles | Turns | Communication | Example |
|---------|-------|-------|---------------|---------|
| **Single-turn** | 1 | 1 | — | Baseline benchmark |
| **Multi-turn** | 1 | 2+ | Same session, sequential prompts | Self-review |
| **Multi-role** | 2+ | 2+ | Explicit prompt sequence | Coder + Reviewer |

**Multi-turn** = same agent gets multiple prompts. Use when a second pass catches errors (self-review, iterative refinement). The agent keeps its context across turns.

**Multi-role** = different agents receive explicit turns. Use when tasks need multiple perspectives (code review, client-advisor). Any handoff text must be part of the declared prompt or agent-native communication, not a BenchFlow Scene scheduler.

Both use the same API — `RolloutConfig` with different `Scene` configurations.

## Multi-Agent Patterns

### Coder + Reviewer (followup-bench)

```python
config = RolloutConfig(
    task_path=task_path,
    scenes=[Scene(
        roles=[Role("coder", "gemini", "flash"), Role("reviewer", "gemini", "flash")],
        turns=[
            Turn("coder"),
            Turn("reviewer", "Review /app/. Summarize any issues."),
            Turn("coder", "Read feedback and fix."),
        ],
    )],
    environment="daytona",
)
```

### Skill Generation + Solve (BYOS)

```python
config = RolloutConfig(
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

## User-Driven Loops

Use `BaseUser` or `FunctionUser` when one agent should run multiple rounds and
Python should decide the next prompt from verifier feedback. This is the
progressive-disclosure path: the user callback can stop early, read
`RoundResult` after each `soft_verify()`, and optionally receive the oracle
solution during `setup()` when `oracle_access=True`.

```python
from pathlib import Path

from benchflow import FunctionUser, RolloutConfig, RoundResult, Scene


def user(round: int, instruction: str, rr: RoundResult | None) -> str | None:
    if round == 0:
        return instruction.splitlines()[0]
    if rr and (rr.rewards or {}).get("reward") == 1.0:
        return None
    return f"Tests failed:\n{rr.verifier_output}\n\nUse the full spec:\n{instruction}"


config = RolloutConfig(
    task_path=Path("tasks/my-task"),
    scenes=[Scene.single(agent="gemini", model="gemini-3.1-flash-lite-preview")],
    user=FunctionUser(user),
    max_user_rounds=3,
    environment="daytona",
)
result = await bf.run(config)
```

Use multi-role Scenes when another LLM should act as the reviewer or simulated
user. Use `BaseUser` when the loop is deterministic or verifier-driven. See
[`progressive-disclosure.md`](../progressive-disclosure.md) and
[`docs/examples/scene-patterns.ipynb`](../examples/scene-patterns.ipynb).

## YAML Rollout Configs

```python
from benchflow._utils.yaml_loader import rollout_config_from_yaml

config = rollout_config_from_yaml("rollout.yaml")
result = await bf.run(config)
```

## Registered Agents

| Agent | Protocol | Auth | Aliases |
|-------|----------|------|---------|
| `gemini` | ACP | GEMINI_API_KEY | — |
| `claude-agent-acp` | ACP | ANTHROPIC_API_KEY | `claude` |
| `codex-acp` | ACP | OPENAI_API_KEY, CODEX_API_KEY, CODEX_ACCESS_TOKEN, or host login | `codex` |
| `opencode` | ACP | inferred from model/provider | — |
| `openhands` | ACP | LLM_API_KEY | `oh` |
| `pi-acp` | ACP | ANTHROPIC_API_KEY | `pi` |
| `openclaw` | ACP | inferred from model | — |

The Auth column shows each agent's native/default credentials. Provider-prefixed
models can use provider-specific credentials instead; for example, Azure
Foundry models use `AZURE_API_KEY` plus `AZURE_API_ENDPOINT` with prefixes such
as `azure-foundry-openai/gpt-5.5` or
`azure-foundry-anthropic/claude-opus-4-5`. BenchFlow routes these providers
through LiteLLM on both Docker and Daytona.

Any agent can be prefixed with `acpx/` to run via [ACPX](https://acpx.sh/) (e.g. `acpx/gemini`, `acpx/claude`). ACPX is a headless ACP client with persistent sessions and crash recovery. The underlying agent's install, env, credentials, and skill paths are preserved.

## Retry and Error Handling

Rollout.run() catches common errors:
- `TimeoutError` — agent exceeded timeout
- `ConnectionError` — SSH/ACP pipe closed (retried 3x with exponential backoff)
- `ACPError` — agent protocol error

Evaluation-level retry with `RetryConfig`:
```python
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig

config = EvaluationConfig(
    retry=RetryConfig(
        max_retries=2,
        wait_multiplier=2.0,
        min_wait_sec=1.0,
        max_wait_sec=30.0,
    ),
)
```

---

## v0.5 Sandbox and Reward Types

### Sandbox Protocol

The `Sandbox` protocol defines the interface any sandbox backend must implement.
Docker and Daytona are built-in; you can bring your own (Modal, Firecracker, E2B, etc.).

```python
from benchflow import Sandbox, ImageBuilder, ImageConfig, ImageRef

# Sandbox is a runtime-checkable Protocol
class MySandbox:
    async def exec(self, cmd: list[str], ...) -> SandboxExecResult: ...
    async def read_file(self, path: str) -> str: ...
    async def write_file(self, path: str, content: str) -> None: ...
    async def stop(self) -> None: ...
    # ... see sandbox/ package for full protocol

assert isinstance(my_sandbox, Sandbox)  # works at runtime
```

### Rubric + RewardFunc (Composable Rewards)

Declarative scoring via composable reward functions.

```python
from benchflow import Rubric, RewardFunc, RewardEvent, VerifyResult
from benchflow import TestRewardFunc, StringMatchRewardFunc, LLMJudgeRewardFunc

# Built-in reward functions
test_reward = TestRewardFunc()          # runs pytest, binary pass/fail
match_reward = StringMatchRewardFunc(expected="hello world")

# Compose into a weighted Rubric
rubric = Rubric(
    reward_funcs=[test_reward, match_reward],
    weights=[0.7, 0.3],
)

# Score a workspace
result: VerifyResult = await rubric.score(rollout_dir=my_rollout_dir)
print(result.reward)      # weighted float [0.0, 1.0]
print(result.events)      # list[RewardEvent] — per-function breakdown
```

### Adapters (Inspect AI + ORS)

Convert between BenchFlow types and external frameworks.

```python
from benchflow import InspectAdapter, ORSAdapter, to_inspect_task, to_ors_reward

# BenchFlow Scene → Inspect AI task format
inspect_task = to_inspect_task(scene, rubric=rubric)

# BenchFlow VerifyResult → ORS reward format
ors_payload = to_ors_reward(verify_result)
```

### Evaluation

Batch orchestration with concurrency and retries.

```python
from benchflow import Evaluation, EvaluationConfig, EvaluationResult

# EvaluationConfig wraps multiple RolloutConfigs
config = EvaluationConfig(
    rollouts=[rollout_config_1, rollout_config_2, ...],
    concurrency=8,
    retry=RetryConfig(max_retries=2),
)
eval_result: EvaluationResult = await Evaluation.run(config)
```
