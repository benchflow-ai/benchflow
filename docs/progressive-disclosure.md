# Progressive Disclosure with `BaseUser`

A pattern for multi-round agent runs where a Python callback drives the loop, deciding what to tell the agent next based on what happened in the previous round.

This is BenchFlow's lightweight alternative to multi-agent "user simulation" Scenes (see [use-cases.md](./use-cases.md#1-interactive-user-simulation-harbor-1316-equivalent)). Use a `BaseUser` callback when:

- You need programmatic control over the loop (e.g. terse prompt → hints on test failure → stop on pass).
- You don't want to spin up a second LLM just to play the "user" role.
- Your "user" logic is rule-based or oracle-guided rather than open-ended.

For comparison: a Scene-based simulated user is another LLM with its own tool access, useful for nuanced feedback. A `BaseUser` is a sync/async Python function, useful for deterministic, scriptable progressive disclosure.

---

## Why this exists

This was built for [Josh's SWE-bench Pro use case](https://github.com/swe-bench-pro/swe-bench-pro): the dataset's instructions are long structured specs that overwhelm agents in a single turn. A `BaseUser` lets you compress the spec to a terse prompt for round 0, watch which tests fail, then disclose hints from the spec on subsequent rounds.

It is also benchflow's parity answer to [Harbor #1316](https://github.com/harbor-ai/harbor/issues/1316) for the no-second-LLM case — Harbor's proposal required a FastMCP sidecar; BenchFlow's `BaseUser` is in-process Python.

---

## Quick start

```python
import asyncio
from pathlib import Path
import benchflow as bf
from benchflow import FunctionUser, RoundResult
from benchflow.trial import TrialConfig, Scene


def my_user(round: int, instruction: str, rr: RoundResult | None) -> str | None:
    if round == 0:
        # Round 0: terse prompt, no hints
        return instruction.split("\n")[0]
    if rr and rr.rewards and rr.rewards.get("reward", 0) >= 1.0:
        return None  # passed, stop
    if round >= 3:
        return None  # cap at 3 rounds
    # Otherwise: show the failing tests as a hint for next round
    return (
        f"The previous attempt failed these tests:\n{rr.verifier_output}\n"
        f"Here is the full spec for context:\n{instruction}"
    )


config = TrialConfig(
    task_path=Path(".ref/swebenchpro/instance_flipt-io__flipt-..."),
    scenes=[Scene.single(agent="opencode", model="anthropic/claude-sonnet-4-6")],
    user=FunctionUser(my_user),
    max_user_rounds=3,
    environment="daytona",
)
result = asyncio.run(bf.run(config))
```

---

## API

### `BaseUser`

Subclass and override `run()`. Optionally override `setup()` for one-time initialization.

```python
from benchflow import BaseUser, RoundResult


class MyUser(BaseUser):
    async def setup(self, instruction: str, solution: str | None = None) -> None:
        """Called once before the first round.

        instruction — the original task instruction (from instruction.md)
        solution    — the gold answer if oracle_access=True, else None
        """
        self.spec_lines = instruction.split("\n")
        self.gold = solution  # only set if oracle_access=True

    async def run(
        self,
        round: int,
        instruction: str,
        round_result: RoundResult | None = None,
    ) -> str | None:
        """Produce the next prompt, or None to stop the loop.

        round         — 0-indexed round number
        instruction   — the original task instruction
        round_result  — None on round 0; previous round's outcome on subsequent rounds
        """
        ...  # return prompt str or None
```

### `RoundResult`

Dataclass passed to `run()` from round 1 onward.

```python
@dataclass
class RoundResult:
    round: int                                # 0-indexed
    trajectory: list[dict]                    # ACP events from this round only
    rewards: dict[str, Any] | None            # verifier rewards (None if verifier crashed)
    verifier_output: str | None               # raw verifier stdout/log content
    verifier_error: str | None                # exception message if verifier failed
    n_tool_calls: int                         # tool calls in this round
```

### `PassthroughUser`

Sends the instruction unchanged on round 0, stops on round 1. Backward-compatible single-round behavior.

### `FunctionUser`

Wraps a plain function as a `BaseUser`. Sync and async both supported (via `inspect.isawaitable`).

```python
def fn(round, instruction, rr): return None if round > 0 else instruction
user = FunctionUser(fn)

async def afn(round, instruction, rr): ...
user = FunctionUser(afn)
```

### `TrialConfig` fields

```python
user: BaseUser | None = None     # the callback
max_user_rounds: int = 5         # hard cap on rounds (loop stops earlier if user returns None)
oracle_access: bool = False      # expose gold solution to user.setup()
```

A `User` requires a single-scene, single-role config. Multi-scene or multi-role configs raise `ValueError`.

---

## Oracle access

When `oracle_access=True`, the trial:

1. Reads `/solution/solve.sh` before the agent starts and passes its content to `user.setup(instruction, solution=...)`.
2. Moves `/solution` → `/solution_oracle_backup` so the agent cannot read it during its rounds.
3. Temporarily restores `/solution` for `soft_verify()` between rounds (and re-hides it).
4. Restores `/solution` permanently before the final `verify()`.

Step 4 is wrapped in a `try/finally`, so if a round throws, the restore still runs.

> ⚠️ Setting `oracle_access=True` without a `User` is a misconfiguration — the solution stays exposed to the agent for the entire trial. BenchFlow logs a `WARNING` at setup time when this happens.

Use cases for oracle access:
- Dataset generation: have the user generate optimal prompts based on knowing the answer.
- Curriculum learning: progressively reveal hints from the gold solution.
- Research: study how much oracle information is needed for an agent to succeed.

---

## Per-task hardening opt-outs

The verifier's pre-run cleanup deletes `conftest.py` files outside `/tests/` to prevent agent reward-hacking. Some tasks (e.g. qutebrowser) ship legitimate `conftest.py` that sets up Python's import order to break a real circular dependency. The default cleanup deletes them, breaking pytest collection.

Tasks declare opt-outs in `task.toml`:

```toml
[verifier]
timeout_sec = 3000

[verifier.hardening]
cleanup_conftests = false
```

Available flags (all default `true` — secure-by-default):

| Flag | Effect when `false` |
|------|---------------------|
| `cleanup_conftests` | Don't delete `conftest.py` outside `/tests/` before verify |

Other cleanup steps (`sitecustomize.py`, `.pth` files, `*.py` in `/tmp`) always run — they have no legitimate use case in repo source trees and broaden the attack surface if disabled.

Unknown keys in `[verifier.hardening]` are logged as warnings and ignored. String values for boolean flags are rejected (must be TOML `true` / `false`).

---

## Failure modes

The user loop catches exceptions from `user.run()` and logs them as the trial error, breaking out of the loop:

```python
[User] round 2: prompt='Try again, focusing on...'
ERROR: user.run() failed at round 2: KeyError: 'spec_section'
```

`soft_verify()` between rounds catches its own timeouts and crashes — they surface as `RoundResult.verifier_error`, not as trial-level failures. The next round still runs; the user sees the error and decides whether to continue.

Trajectory and tool counts are sliced per round from `Trial._trajectory`. The session counters reset on `disconnect()` between rounds, so each round's `RoundResult.trajectory` and `n_tool_calls` reflect only that round's events.

---

## Worked example

See [`examples/swebench_pro_progressive_disclosure.ipynb`](../examples/swebench_pro_progressive_disclosure.ipynb) for a 5-task SWE-bench Pro comparison: oracle vs single-round baseline vs 3-round progressive disclosure on flipt and openlibrary.

For a minimal end-to-end script, see [`examples/user_dogfood.py`](../examples/user_dogfood.py).

---

## Comparison with multi-agent Scene-based user simulation

| Pattern | When to use |
|---------|-------------|
| `BaseUser` callback (this doc) | Programmatic, rule-based, deterministic. No second LLM. Cheap. |
| Multi-role Scene with simulated-user role ([use-cases.md §1](./use-cases.md#1-interactive-user-simulation-harbor-1316-equivalent)) | Open-ended, conversational. The "user" is another LLM with full tool access. Better for nuanced human-like interaction. |

Both patterns coexist. Choose `BaseUser` for the lighter-weight case; choose Scenes when you actually want a second agent in the loop.
