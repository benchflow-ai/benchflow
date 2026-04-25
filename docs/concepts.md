# Concepts

The mental model for benchflow. Read once, then refer back from the how-tos.

---

## The five primitives

| Primitive | What it is |
|-----------|------------|
| **Task** | A directory on disk: `instruction.md` for the agent + `tests/` for the verifier + (optional) `solution/solve.sh` for oracle runs + `environment/Dockerfile` for the sandbox. Authored once, evaluated many times. |
| **Agent** | A registered ACP-speaking program (Claude Code, Gemini CLI, OpenCode, etc.). Identified by name (`"gemini"`, `"opencode"`) plus an optional model ID. |
| **Environment** | The sandbox where the agent runs and the verifier checks the result. Backed by Harbor — Docker locally, Daytona for cloud. |
| **Verifier** | The test runner that scores the trial. By default `pytest /tests/...` against the workspace the agent left behind. Outputs `rewards: {reward: float}`. |
| **Trial** | One agent run on one task. Holds the lifecycle (setup → start → install → execute → verify → cleanup). All higher-level primitives below are built on Trials. |

---

## Trial lifecycle

A `Trial` is decomposable: each phase is a callable method, you can either run them in sequence or invoke `Trial.run()` to execute all six in order. Multi-agent flows reuse phases (e.g. `connect` + `execute` + `disconnect` repeats per role).

```
┌──────────────────────────────────────────────────────────────┐
│                    Trial.run()                               │
│                                                              │
│  setup()         resolve config, create Harbor env handle    │
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

Each phase has a name, a clear contract, and is independently testable. `Trial.run()` is the convenience that calls them in order.

```python
import benchflow as bf
from benchflow.trial import TrialConfig, Scene
from pathlib import Path

config = TrialConfig(
    task_path=Path("tasks/regex-log"),
    scenes=[Scene.single(agent="gemini", model="gemini-3.1-pro-preview")],
    environment="daytona",
)
result = await bf.run(config)   # full lifecycle
print(result.rewards)            # {'reward': 1.0}
```

---

## Scenes, Roles, Turns

A **Scene** is one interaction region. Inside a Scene:
- **Roles** are the agents that participate (one or more).
- **Turns** are the prompt sequence — which Role acts when, and what they're told.
- All Roles share the same sandbox filesystem.

Single-agent runs are a Scene with one Role and one Turn. Multi-agent patterns (coder + reviewer, simulated user + assistant) are Scenes with multiple Roles and ordered Turns.

```python
Scene(
    name="review-loop",
    roles=[
        Role(name="coder",    agent="opencode", model="anthropic/claude-sonnet-4-6"),
        Role(name="reviewer", agent="gemini",   model="gemini-3.1-pro-preview"),
    ],
    turns=[
        Turn(role="coder"),
        Turn(role="reviewer", prompt="Read /app/ and write feedback to /app/.outbox/coder.json."),
        Turn(role="coder",    prompt="Read the reviewer's feedback and revise."),
    ],
)
```

Roles communicate via **outbox files**: write JSON to `/app/.outbox/{recipient}.json` and the scheduler injects it into the next Turn's prompt.

A Trial may have multiple Scenes — used for staged flows like "skill generation → solve" (BYOS / Bring Your Own Skill). Same sandbox, sequential Scenes.

---

## The User abstraction (multi-round, single-agent)

Sometimes you want the agent to take multiple turns guided not by another LLM but by a Python callback that watches what happened and decides what to say next. That's a **User**.

A User is a `BaseUser` subclass (or `FunctionUser` wrapping a function) with two methods:
- `setup(instruction, solution)` — once, before round 0
- `run(round, instruction, round_result) → str | None` — per round; return `None` to stop the loop

Between rounds, benchflow runs `soft_verify()` (verifier without the destructive parts of full hardening), gives the user the round's `RoundResult` (trajectory, rewards, verifier output, tool count), and lets the user decide round N+1's prompt.

The User is the lighter-weight alternative to a Scene with a simulated-user Role: no second LLM, no outbox protocol, just a Python function. Use it when the loop logic is rule-based (compress instruction → show test failures as hints → stop on pass). See [`progressive-disclosure.md`](./progressive-disclosure.md) for the full guide.

---

## Verifier, sandbox, hardening

Once the agent stops, the verifier runs. By default that's `pytest -c /dev/null --confcutdir=/tests --rootdir=/app -p no:cacheprovider /tests/test.sh` (or whatever the task's `tests/test.sh` does), against the workspace the agent left behind.

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
| **Multi-scene** | Multiple Scenes in one Trial. Sandbox state persists; agent process and ACP session restart between Scenes. | BYOS: Scene 1 generates a skill, Scene 2 solves the task using it. |

Single-agent simple runs use none of these. Pick the axis based on what state needs to persist (memory? sandbox? both?).

---

## Trajectories and rewards

Every agent action is captured as an event in the **trajectory** — tool calls, agent messages, agent thoughts. A `RunResult` has the full trajectory plus tool count, plus rewards from the verifier and any error.

`rewards` is a dict produced by the task's verifier. Convention: `{"reward": float}` where 1.0 = pass, 0.0 = fail. Tasks may add additional metrics (e.g. `exact_match`, `partial_credit`).

Trajectories are written to `<jobs_dir>/<job_name>/<trial_name>/trajectory/acp_trajectory.jsonl`. Use them for replay, debugging, or training data.

---

## Where to go next

- [Getting started](./getting-started.md) — install, run your first eval.
- [Task authoring](./task-authoring.md) — write a task with `task.toml` + `tests/` + `solution/`.
- [Progressive disclosure](./progressive-disclosure.md) — the User abstraction; SWE-bench Pro case study.
- [Use cases](./use-cases.md) — multi-agent patterns (coder/reviewer, simulated user, BYOS, stateful environments).
- [CLI reference](./reference/cli.md), [Python API reference](./reference/python-api.md).
- [Skill evaluation](./skill-eval.md) — when the artifact is a skill, not a workspace.
