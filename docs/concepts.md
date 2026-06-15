# Concepts
The mental model for benchflow. Read once, then refer back from the how-tos.

---

## The five primitives

| Primitive | What it is |
|-----------|------------|
| **Task** | A directory on disk: a `task.md` document (YAML frontmatter + prompt body) plus `environment/Dockerfile` for the sandbox, `verifier/` checks, and optional `oracle/` — or the legacy split layout (`task.toml` + `instruction.md` + `tests/` + `solution/`). Authored once, evaluated many times. |
| **Agent** | A registered ACP-speaking program (Claude Code, Gemini CLI, OpenCode, etc.). Identified by name (`"gemini"`, `"opencode"`) plus an optional model ID. Use the `acpx/` prefix (e.g. `acpx/gemini`) to route through [ACPX](https://acpx.sh/), a headless ACP client with persistent sessions and crash recovery. |
| **Environment** | The sandbox where the agent runs and the verifier checks the result. Docker locally, Daytona for cloud, Modal for serverless/GPU. Abstracted behind the `Sandbox` protocol — bring your own sandbox backend. |
| **Verifier** | The test runner that scores the rollout. Its entry point is a `test.sh` script (native `verifier/test.sh`, legacy `tests/test.sh`) — which typically runs `pytest` against the workspace the agent left behind. For subjective tasks, use an [LLM-as-judge](./llm-judge.md) verifier scored against a rubric. Outputs `rewards: {reward: float}`. See the [verifier file map](#verifier-file-map) for which file lives where in native vs legacy packages. |
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

Use `BaseUser` when the loop logic is rule-based (compress instruction → show test failures as hints → stop on pass). See [`progressive-disclosure.md`](./progressive-disclosure.md) for the full guide.

---

## Verifier, sandbox, hardening

Once the agent stops, the verifier runs. Its entry point is the task's
`test.sh` script — uploaded to `/verifier` for native packages (`/tests` for
legacy ones) — executed against the workspace the agent left behind. benchflow
runs `test.sh` **as a script** (it `chmod +x`'s the file and executes it
directly; a native `script` strategy runs `cd /verifier && <command>`). It
never hands `test.sh` to `pytest` — pytest cannot collect a shell script as a
test target.

Most `test.sh` scripts *invoke* pytest internally. For those invocations,
benchflow applies hardening through `PYTEST_ADDOPTS` in the verifier
environment — every pytest run inside `test.sh` inherits roughly:

```text
PYTEST_ADDOPTS="-c /dev/null --confcutdir=<verifier-dir> --rootdir=<workspace> -p no:cacheprovider"
```

where `<verifier-dir>` is `/verifier` for native packages (`/tests` for
legacy), and `<workspace>` is the agent workspace (`/app` for SWE-bench
conventions, `/root` for SkillsBench — injected dynamically). `-c /dev/null`
blocks `pyproject.toml`/`pytest.ini` discovery and `--confcutdir` blocks
`conftest.py` walk-up beyond the verifier dir. Tasks that do not use pytest
(e.g. a `test.sh` that diffs files and writes `reward.txt` directly) are
scored the same way — pytest is just the most common tool, not a requirement.

Between agent and verifier, benchflow **hardens** the sandbox to prevent the agent from gaming the score:
- Kill any lingering agent processes
- Restore build-config files (setup.py, pyproject.toml, …) to their pre-agent snapshots
- Delete agent-injected `conftest.py`, `sitecustomize.py`, `.pth` files
- Lock the workspace to root, set restrictive PYTHONPATH/PATH for the verifier process
- Run pytest with plugin auto-discovery off, only allowing plugins declared in the task config (`[verifier] pytest_plugins` in the `task.md` front-matter, or `task.toml` for split-layout tasks, or auto-discovered root-owned plugins)

This catches the BenchJack and Meerkat exploit families.

When a task ships a legitimate `conftest.py` (e.g. qutebrowser uses one to break a real circular import), the task opts out in its task config (`task.md` front-matter, or `task.toml` for split-layout tasks):

```toml
[verifier.hardening]
cleanup_conftests = false
```

See [`progressive-disclosure.md`](./progressive-disclosure.md#per-task-hardening-opt-outs) for the full opt-out list.

### Verifier file map

Native `task.md` packages and the legacy split layout name their verifier
files differently. The runtime resolves native files first and falls back to
the legacy names, so a task ships **one** of each row, not both:

| What it is | Native (`task.md`) package | Legacy split layout | Sandbox path |
|---|---|---|---|
| Verifier directory | `verifier/` | `tests/` | `/verifier` (native), `/tests` (legacy) |
| Script entry point | `verifier/test.sh` | `tests/test.sh` | executed as a script (chmod +x then run) inside the verifier dir |
| Strategy declaration (how it's scored) | `verifier/verifier.md` | — (legacy uses `[verifier]` in `task.toml`) | not uploaded as a runtime target; selects the strategy |
| LLM-judge rubric | `verifier/rubrics/verifier.md` + `verifier/rubrics/verifier.toml` | `tests/rubric.toml` (also `rubric.json`, Harvey-LAB style) | downloaded for the judge |

A plain `test.sh` is a complete verifier on its own: with no `verifier.md`
strategy declared, the runtime just executes it. `verifier/verifier.md`
declares *how* a task is scored (script / llm-judge / reward-kit / agent-judge
/ ors-episode) and is the native equivalent of the legacy `[verifier]` section
in `task.toml`. The native LLM-judge rubric lives under `verifier/rubrics/`
(both a human-readable `verifier.md` and a machine-readable `verifier.toml`),
not in a single top-level `rubric.toml`. For the native verifier document and
its strategy table see [Native task.md authoring](./task-authoring-task-md.md);
for the legacy `[verifier.judge]` rubric path see [LLM-as-judge](./llm-judge.md).

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

Trajectories are written to `<jobs_dir>/<job_name>/<rollout_name>/trajectory/acp_trajectory.jsonl` (the `--jobs-dir` directory, default `jobs/`). Use them for replay, debugging, or training data.

---

## Where to go next

- [Getting started](./getting-started.md) — install, run your first eval.
- [Task authoring (native task.md)](./task-authoring-task-md.md) — write a task as a single `task.md` document plus `environment/` and `verifier/`.
- [Migrating a legacy task](./task-authoring.md) — convert an existing `task.toml` + `instruction.md` split package to `task.md` (the split layout is no longer a first-class authoring path).
- [LLM-as-judge](./llm-judge.md) — use an LLM to score subjective tasks against a rubric (see the [verifier file map](#verifier-file-map) for native vs legacy rubric paths).
- [Progressive disclosure](./progressive-disclosure.md) — the User abstraction; SWE-bench Pro case study.
- [Use cases](./use-cases.md) — multi-agent patterns (coder/reviewer, simulated user, BYOS, stateful environments).
- [CLI reference](./reference/cli.md), [Python API reference](./reference/python-api.md).
- [Skill evaluation](./skill-eval.md) — when the artifact is a skill, not a workspace.
