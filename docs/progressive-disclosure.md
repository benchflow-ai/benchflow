# Progressive Disclosure with `BaseUser`

## TL;DR

`BaseUser` is a Python callback that drives a benchflow trial across multiple rounds. Each round: the callback sees the previous verifier result and decides what to tell the agent next, or stops the loop. No second LLM, no outbox protocol — just a function that knows how to grade and hint.

It was built for [Josh @ GitHub/Microsoft's SWE-bench Pro use case](https://www.swebench.com/multimodal.html): the dataset's instructions are long structured specs that overwhelm agents in a single turn. A `BaseUser` lets you compress the spec for round 0, watch which tests fail, then disclose hints from the spec on subsequent rounds — all driven by deterministic Python, not by another LLM acting as a "user."

It is also benchflow's parity answer to [Harbor's simulated-user proposal (#1316)](https://github.com/harbor-ai/harbor/issues/1316) for the no-second-LLM case. Harbor's proposal required a FastMCP sidecar container; benchflow's `BaseUser` is in-process Python.

```python
import benchflow as bf
from benchflow import FunctionUser, RoundResult
from benchflow.trial import TrialConfig, Scene
from pathlib import Path


def progressive(round: int, instruction: str, rr: RoundResult | None) -> str | None:
    if round == 0:
        return instruction.split("\n")[0]                # terse: first line only
    if rr and (rr.rewards or {}).get("reward", 0) >= 1.0:
        return None                                      # passed, stop
    if round >= 3:
        return None                                      # cap at 3 rounds
    return (
        f"Tests failed:\n{rr.verifier_output}\n\n"       # show failures + spec
        f"Full spec:\n{instruction}"
    )


config = TrialConfig(
    task_path=Path(".ref/swebenchpro/instance_flipt-io__flipt-..."),
    scenes=[Scene.single(agent="opencode", model="anthropic/claude-sonnet-4-6")],
    user=FunctionUser(progressive),
    max_user_rounds=3,
    environment="daytona",
)
result = await bf.run(config)
```

---

## Case study: SWE-bench Pro (Josh @ GitHub)

SWE-bench Pro tasks ship long, structured `instruction.md` specs (typically 2-5KB) describing API requirements, test fixtures, and expected behaviors. Single-shot agents either drown in the spec or under-engineer because they bail before reading to the bottom.

The SWE-bench Pro eval that motivated this feature wanted exactly this loop:

```
round 0   "Fix the bug described here: <one-line summary>"
            agent attempts → tests fail
round 1   "Tests <names> failed. Here is the full requirements section: <half of spec>."
            agent retries → tests still fail
round 2   "Still failing. Here's the full original spec: <complete instruction>"
            agent makes final attempt
```

Rule-based, deterministic, and the "user" never needs to think — the disclosure schedule is fixed. Spinning up a second LLM to play the user role would (a) cost double, (b) introduce nondeterminism, and (c) require an outbox protocol the agent has to learn.

### Validation (2026-04-24, 5 SWE-bench Pro tasks, Daytona)

| Task | Oracle | Single-round (Gemini 3.1 Pro) | 3-round progressive |
|------|--------|-------------------------------|---------------------|
| ansible | ✅ 1.0 | ✅ 1.0 (23 tools, 207s) | (already passed baseline) |
| flipt | ✅ 1.0 | ❌ 0.0 (61 tools, 1444s) | ❌ 0.0 (203 tools, 3 rounds) |
| openlibrary | ✅ 1.0 | ✅ 1.0 (32 tools, 340s) | (already passed baseline) |
| navidrome | ✅ 1.0 | (not tested) | — |
| qutebrowser | ✅ 1.0 (with `cleanup_conftests=false`) | ❌ 0.0 (verifier broken pre-fix) | — |

flipt is the case where progressive disclosure *should* help — oracle passes but baseline fails after 24 minutes — and on this run with Gemini 3.1 Pro it didn't. The infrastructure works (round-by-round trajectories captured, hints injected, soft_verify between rounds, oracle access semantics), but a single-model evaluation isn't conclusive — flipt is genuinely hard and the failure mode may not be one progressive disclosure addresses. Different models or hint schedules may show different lifts. The notebook at [`examples/swebench_pro_progressive_disclosure.ipynb`](../examples/swebench_pro_progressive_disclosure.ipynb) has the full data.

---

## Where it lives in the trial lifecycle

`BaseUser` plugs into the existing `Trial` lifecycle ([concepts](./concepts.md#trial-lifecycle)) without changing any of the existing phases. When `TrialConfig.user` is set, `Trial._run_user_loop()` replaces the single-pass `connect → execute → disconnect` block with a per-round version:

```
setup() → start() → install_agent()
    ↓
[oracle setup if oracle_access=True: read /solution, hide it from agent]
    ↓
user.setup(instruction, solution)        ← once
    ↓
┌─ user.run(round, instruction, rr) → str | None
│      │ None: break
│      ↓
│   connect_as(role)
│   execute(prompts=[prompt])
│   disconnect()
│      ↓
│   soft_verify()                         ← partial hardening, sandbox stays alive
│      ↓
│   build RoundResult, log, repeat
└─    │
      ↓ (loop ends when user returns None or max_user_rounds reached)
[oracle restore: mv /solution_oracle_backup → /solution for final verify]
    ↓
verify()                                  ← full hardening, final reward
    ↓
cleanup()
```

Multi-scene / multi-role configs are not compatible with `User` — the loop assumes one Scene with one Role. Setting both raises `ValueError`.

---

## Soft-verify and full-verify: two different verifiers

Between rounds, benchflow needs to score the agent's progress so the user can react. But the final, end-of-trial verifier does destructive things (kills the agent, restores the workspace, chowns to root) that would prevent the next round from running. So benchflow runs **two** verifier passes:

| | Soft-verify (between rounds) | Full-verify (end of trial) |
|---|---|---|
| Kills agent processes | ❌ no | ✅ yes |
| Restores workspace from snapshot | ❌ no | ✅ optional, task-driven |
| Purges agent-injected `conftest.py`, `sitecustomize.py`, `.pth` | ✅ yes | ✅ yes |
| Locks down PATH/PYTHONPATH | ✅ yes | ✅ yes |
| `chmod 777 /logs/verifier` | ✅ yes (so non-root verifier can write) | n/a (root) |
| Runs verifier | ✅ yes | ✅ yes |
| Result | feeds `RoundResult.rewards` | the trial's final score |

Soft-verify is intentionally weaker than full-verify — losing some score-gaming protection in exchange for keeping the sandbox alive. The cleanup step still purges agent-injected hook files (`CLEANUP_CMD`), so an agent can't plant a `conftest.py` that flips the round score.

---

## API

### `BaseUser`

```python
from benchflow import BaseUser, RoundResult


class MyUser(BaseUser):
    async def setup(self, instruction: str, solution: str | None = None) -> None:
        """Called once before round 0.

        instruction — the original task instruction (from instruction.md)
        solution    — gold answer if oracle_access=True, else None
        """
        self.spec = instruction
        self.gold = solution

    async def run(
        self,
        round: int,
        instruction: str,
        round_result: RoundResult | None = None,
    ) -> str | None:
        """Return the next prompt, or None to stop.

        round — 0-indexed
        instruction — original task instruction (unchanged each round)
        round_result — None on round 0; previous round's outcome on subsequent rounds
        """
        ...
```

### `RoundResult`

Dataclass passed to `run()` from round 1 onward.

```python
@dataclass
class RoundResult:
    round: int                     # 0-indexed
    trajectory: list[dict]         # ACP events from this round only
    rewards: dict | None           # verifier rewards (None if verifier crashed)
    verifier_output: str | None    # raw verifier stdout/log
    verifier_error: str | None     # exception message if verifier failed
    n_tool_calls: int              # tool calls in this round
```

### `PassthroughUser`

Sends the instruction unchanged on round 0, stops on round 1. Use it as the explicit single-round-equivalent.

### `FunctionUser`

Wraps a plain function as a `BaseUser`. Sync or async — uses `inspect.isawaitable` to detect.

```python
def fn(round, instruction, rr): ...
user = FunctionUser(fn)

async def afn(round, instruction, rr): ...
user = FunctionUser(afn)
```

### `TrialConfig` fields

```python
user: BaseUser | None = None     # the callback
max_user_rounds: int = 5         # cap on rounds (loop also stops when user returns None)
oracle_access: bool = False      # expose gold solution to user.setup()
```

---

## Oracle access

When `oracle_access=True`:

1. Before round 0, the trial reads `/solution/solve.sh` and passes its contents to `user.setup(instruction, solution=...)`.
2. The trial moves `/solution` → `/solution_oracle_backup` so the agent can't read it during its rounds.
3. Between rounds, soft-verify temporarily restores `/solution` (some verifiers consult it) then re-hides it.
4. Before the final `verify()`, the trial permanently restores `/solution`.

Step 4 is wrapped in `try/finally` against the user loop: if a round throws, the restore still runs.

> ⚠️ Setting `oracle_access=True` *without* a `User` is a misconfiguration — the solution stays exposed to the agent for the entire trial. benchflow logs a `WARNING` at setup time when this happens.

Use cases for oracle access:
- **Dataset generation** — the user has the answer, generates an optimal prompt for the agent
- **Curriculum learning** — progressively reveal pieces of the solution
- **Research** — measure how much oracle information is required for an agent to succeed

---

## Per-task hardening opt-outs

The verifier's pre-run cleanup deletes `conftest.py` outside `/tests/` to prevent reward-hacking. Some tasks (qutebrowser) ship legitimate `conftest.py` files that fix real circular imports — deleting them breaks pytest collection.

Tasks opt out in `task.toml`:

```toml
[verifier.hardening]
cleanup_conftests = false
```

| Flag | Default | Effect when `false` |
|------|---------|---------------------|
| `cleanup_conftests` | `true` | Don't delete `conftest.py` outside `/tests/` before verify |

`sitecustomize.py`, `.pth` files, and `*.py` in `/tmp` always get cleaned — they have no legitimate use in a test artifact and disabling them broadens the attack surface beyond what real-world tasks need.

Unknown keys in `[verifier.hardening]` are warned and ignored. String values for boolean flags are rejected.

---

## Failure modes

The user loop catches exceptions from `user.run()` and stops, with the exception message stored in `Trial._error`:

```
[User] round 2: prompt='Try again, focusing on...'
ERROR  user.run() failed at round 2: KeyError: 'spec_section'
```

`soft_verify()` between rounds catches its own timeouts and crashes — they surface as `RoundResult.verifier_error`, not as a trial-level failure. The next round still runs and the user can decide what to do.

Trajectory and tool counts are sliced per round from `Trial._trajectory`. The session counters reset on `disconnect()`, so each round's `RoundResult.trajectory` and `n_tool_calls` reflect only that round's events, not cumulative.

---

## Comparison with multi-agent simulated user (Harbor #1316 parity)

benchflow has two patterns for multi-round agent runs. Both are functionally at parity with [Harbor #1316](https://github.com/harbor-ai/harbor/issues/1316) — neither requires a FastMCP sidecar.

| Pattern | What "user" is | When to use |
|---------|---------------|-------------|
| **`BaseUser` callback (this doc)** | Python function in the scheduler process | Programmatic, deterministic, rule-based. No second LLM. Cheap. Best for progressive disclosure, curriculum, scripted hints. |
| **Multi-role Scene with simulated-user role** ([use-cases §1](./use-cases.md#1-interactive-user-simulation-harbor-1316-equivalent)) | Another LLM with full tool access | Open-ended, conversational. The "user" can read files, check outputs, give nuanced feedback. Best when the user's behavior must itself be adaptive or LLM-quality. |

The two coexist. Choose based on whether your "user" needs to think (Scene-based) or just decide (`BaseUser`). For Josh's SWE-bench Pro use case, the disclosure schedule is fixed, the grading is the verifier, and there's nothing for a second LLM to add — `BaseUser` wins on cost and determinism.

---

## Worked examples

- [`examples/swebench_pro_progressive_disclosure.ipynb`](../examples/swebench_pro_progressive_disclosure.ipynb) — the SWE-bench Pro case study, executable end-to-end with the latest oracle/baseline data.
- [`examples/swebench_pro_user_dogfood.py`](../examples/swebench_pro_user_dogfood.py) — runnable script for any of the 5 SWE-bench Pro tasks. `--task flipt --max-rounds 3`.
- [`examples/user_dogfood.py`](../examples/user_dogfood.py) — minimal regex-log task with `FunctionUser`, useful as a starting template.
- [`experiments/swebench_pro_oracle_and_baseline.py`](../experiments/swebench_pro_oracle_and_baseline.py) — the oracle-validation + baseline experiment script that produced the table above.
