# Concepts
The mental model for benchflow. Read once, then refer back from the how-tos.

---

## The three planes

Every BenchFlow eval is one selection from each of three orthogonal **planes**.
Pin this down once and the rest of the docs fall into place:

| Plane | The question it answers | Modes / choices | Guide |
|---|---|---|---|
| **Environment** | How is the world built, reset, and secured? | container · service-catalog · hosted · aux-VM; network policy; hardening | [Environment plane](./environment-plane.md), [Sandbox hardening](./sandbox-hardening.md) |
| **Interaction** | Who acts, and in what loop? | single-shot · multi-round · simulated-user · multi-agent-sequential · arena | [Use cases](./use-cases.md), [Progressive disclosure](./progressive-disclosure.md) |
| **Verifier** | What surface is scored, and how? | workspace-test · trajectory · rubric / LLM-judge · agent-judge · leaderboard | [LLM-as-judge](./llm-judge.md), [Skill eval](./skill-eval.md) |

A task declares its choices on each plane in one `task.md` — see [the task
standard](./task-standard.md) for the full contract. Everything below (Rollout,
Scene, Role, User, Verifier) is the machinery that executes those choices. The
five primitives map onto the planes: **Environment** is the Environment plane;
**Agent / Scene / Role / User** drive the Interaction plane; **Verifier** is the
Verifier plane; **Rollout** ties one selection from each plane into a single
scored run.

---

## The five primitives

| Primitive | What it is |
|-----------|------------|
| **Task** | A directory on disk: `task.md` for config, prompt, roles, scenes, and simulated-user notes + `verifier/` for scoring + optional `oracle/solve.sh` for oracle runs + `environment/Dockerfile` for the sandbox. Legacy `instruction.md`, `task.toml`, `tests/`, and `solution/` tasks still load. Authored once, evaluated many times. |
| **Agent** | A registered ACP-speaking program (Claude Code, Gemini CLI, OpenCode, etc.). Identified by name (`"gemini"`, `"opencode"`) plus an optional model ID. Use the `acpx/` prefix (e.g. `acpx/gemini`) to route through [ACPX](https://acpx.sh/), a headless ACP client with persistent sessions and crash recovery. |
| **Environment** | The sandbox where the agent runs and the verifier checks the result. Docker locally, Daytona for cloud, Modal for serverless/GPU. Abstracted behind the `Sandbox` protocol — bring your own sandbox backend. |
| **Verifier** | The test runner that scores the rollout. Native tasks mount `verifier/` at `/verifier/`; legacy tasks mount `tests/` at `/tests/`. For subjective tasks, use an [LLM-as-judge](./llm-judge.md) verifier with a `rubric.toml`. Outputs `rewards: {reward: float}`. |
| **Rollout** | One agent run on one task. Holds the lifecycle (setup → start → install → execute → verify → cleanup). All higher-level primitives below are built on Rollouts. |

---

## Rollout lifecycle

A `Rollout` is decomposable: each phase is a callable method, you can either run them in sequence or invoke `Rollout.run()` to execute all six in order. Multi-agent flows reuse phases (e.g. `connect` + `execute` + `disconnect` repeats per role).

```
┌──────────────────────────────────────────────────────────────┐
│                    Rollout.run()                             │
│                                                              │
│  setup()         resolve config, create sandbox env handle   │
│    ↓                                                         │
│  start()         start container, upload task files          │
│    ↓                                                         │
│  install_agent() install agent binary, write credentials,    │
│                  set up sandbox user                         │
│    ↓                                                         │
│  ┌─ connect_as(role)  ◄─── multi-agent loops here            │
│  │  execute(prompts)        each role's turn                 │
│  └─ disconnect()                                             │
│    ↓                                                         │
│  verify()        harden sandbox, run pytest, score           │
│    ↓                                                         │
│  cleanup()       kill agent procs, stop container            │
└──────────────────────────────────────────────────────────────┘
```

Each phase has a name, a clear contract, and is independently testable. `Rollout.run()` is the convenience that calls them in order.

```python
import benchflow as bf
from benchflow import RolloutConfig, Scene
from pathlib import Path

config = RolloutConfig(
    task_path=Path("tasks/edit-pdf"),
    scenes=[Scene.single(agent="gemini", model="gemini-3.1-pro-preview")],
    environment="daytona",
)
result = await bf.run(config)   # full lifecycle
print(result.rewards)            # {'reward': 1.0}
```

---

## Scenes, Roles, Turns

A **Scene** is authoring sugar for Step metadata. Inside a Scene:
- **Roles** are the agents that participate (one or more).
- **Turns** are the prompt sequence — which Role acts when, and what they're told.
- All Roles share the same sandbox filesystem.

Before rollout execution, BenchFlow desugars Scenes into explicit rollout Steps carrying role, prompt, and skill attribution. Scene has no runtime object, scheduler, message router, or lifecycle.

```python
Scene(
    name="review-loop",
    roles=[
        Role(name="coder",    agent="opencode", model="anthropic/claude-sonnet-4-6"),
        Role(name="reviewer", agent="gemini",   model="gemini-3.1-pro-preview"),
    ],
    turns=[
        Turn(role="coder"),
        Turn(role="reviewer", prompt="Review the current workspace."),
        Turn(role="coder",    prompt="Read the reviewer's feedback and revise."),
    ],
)
```

A Rollout may have multiple Scenes — used for staged flows like "skill generation → solve" (BYOS / Bring Your Own Skill). Same sandbox, sequential Scenes.

---

## The User abstraction (multi-round, single-agent)

Sometimes you want the agent to take multiple turns guided not by another LLM but by a Python callback that watches what happened and decides what to say next. That's a **User**.

A User is a `BaseUser` subclass (or `FunctionUser` wrapping a function) with two methods:
- `setup(instruction, solution)` — once, before round 0
- `run(round, instruction, round_result) → str | None` — per round; return `None` to stop the loop

Between rounds, BenchFlow executes `soft_verify()` (verifier without the destructive parts of full hardening), gives the user the round's `RoundResult` (trajectory, rewards, verifier output, tool count), and lets the user decide round N+1's prompt.

Document-declared `task.md` users are a bounded runtime on top of this
abstraction: deterministic and model-linear private-fact users can run across a
linear sequence of single-role scenes. Multi-role team handoff remains a scene
orchestration problem, not generic `User` callback behavior.

Use `BaseUser` when the loop logic is rule-based (compress instruction → show test failures as hints → stop on pass). See [`progressive-disclosure.md`](./progressive-disclosure.md) for the full guide.

---

## Verifier, sandbox, hardening

Once the agent stops, the verifier runs against the workspace the agent left behind. Native tasks mount `verifier/` at `/verifier` and may run `verifier/test.sh` by default or select `script`, `llm-judge`, `reward-kit`, or scoped `agent-judge` strategies via `verifier/verifier.md`; legacy tasks run `/tests/test.sh` from `tests/`.

Between agent and verifier, benchflow **hardens** the sandbox to prevent the agent from gaming the score:
- Kill any lingering agent processes
- Restore build-config files (setup.py, pyproject.toml, …) to their pre-agent snapshots
- Delete agent-injected `conftest.py`, `sitecustomize.py`, `.pth` files
- Lock the workspace to root, set restrictive PYTHONPATH/PATH for the verifier process
- Run pytest with plugin auto-discovery off, only allow plugins declared in `task.toml`

This catches the BenchJack and Meerkat exploit families documented in [`labs/benchjack-sandbox-hardening/`](../labs/benchjack-sandbox-hardening/) and [`labs/reward-hack-matrix/`](../labs/reward-hack-matrix/).

When a task ships a legitimate `conftest.py` (e.g. qutebrowser uses one to break a real circular import), the task opts out via `task.toml`:

```toml
[verifier.hardening]
cleanup_conftests = false
```

See [`progressive-disclosure.md`](./progressive-disclosure.md#per-task-hardening-opt-outs) for the full opt-out list.

---

## Multi-turn vs multi-round vs multi-scene

Three different axes — easy to confuse, worth pinning down:

| Axis | What changes | Example |
|------|--------------|---------|
| **Multi-turn** | Same Role, multiple prompts within one Scene. The ACP session persists; the agent has continuous memory. | One coder gets prompted twice: "fix the bug", then "now write a test". |
| **Multi-round** | Same Role, multiple `connect → execute → disconnect` cycles. New ACP session each round; sandbox state persists; a Python `User` callback decides each round's prompt. | Progressive disclosure on SWE-bench Pro: round 0 terse spec, round 1 hints with failing tests, round 2 full spec. |
| **Multi-scene** | Multiple Scenes in one Rollout. Sandbox state persists; agent process and ACP session restart between Scenes. | BYOS: Scene 1 generates a skill, Scene 2 solves the task using it. |

Single-agent simple runs use none of these. Pick the axis based on what state needs to persist (memory? sandbox? both?).

---

## Trajectories and rewards

Every agent action is captured as an event in the **trajectory** — tool calls, agent messages, agent thoughts. A `RolloutResult` (aliased as `RunResult`) has the full trajectory plus tool count, plus rewards from the verifier and any error.

`rewards` is a dict produced by the task's verifier. Convention: `{"reward": float}` where 1.0 = pass, 0.0 = fail. Tasks may add additional metrics (e.g. `exact_match`, `partial_credit`).

Trajectories are written to `<evaluations_dir>/<evaluation_name>/<rollout_name>/trajectory/acp_trajectory.jsonl`. Use them for replay, debugging, or training data.

---

## Where to go next

By plane:

- **The standard** — [Task standard](./task-standard.md) (the full contract), [Task authoring](./task-authoring.md) (write a `task.md` with `verifier/` and optional `oracle/`).
- **Environment plane** — [Environment plane](./environment-plane.md) (how the world is built/reset), [Sandbox hardening](./sandbox-hardening.md) (the anti-reward-hacking security model).
- **Interaction plane** — [Use cases](./use-cases.md) (multi-agent: coder/reviewer, simulated user, BYOS, stateful envs), [Progressive disclosure](./progressive-disclosure.md) (multi-round single-agent; SWE-bench Pro case study).
- **Verifier plane** — [LLM-as-judge](./llm-judge.md) (score subjective tasks with `rubric.toml`), [Skill eval](./skill-eval.md) (when the artifact is a skill, not a workspace).
- **Operate** — [Getting started](./getting-started.md), [Running benchmarks](./running-benchmarks.md), [CLI reference](./reference/cli.md), [Python API reference](./reference/python-api.md).
