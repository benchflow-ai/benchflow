# Multi-turn tasks

Some work isn't one prompt. A user adds requirements as the agent goes, or answers its questions, or several agents each do their part in the same workspace. BenchFlow runs both, inside one trial and one sandbox: a user who replies between rounds, written as a Python function, and several agents taking turns. The task's test scores the end result. When each stage needs its own check, use a [multi-step task](./multi-step.md) (0.10) instead.

## Turns and rounds

- A **turn** is one prompt to one agent.
- A **scene** is a list of turns, each by a named role; a task can have several scenes, run in order.
- A **round** is one exchange with a user: the agent works, the task's test runs as a soft check, and the user decides the next prompt or stops.

The trial's reward is the final check after the last turn, with the agent stopped, as in any task ([Stop reward hacking](./reward-hacking.md)).

## A scripted user

`tasks/half-life` is the task [Write a task](./write-a-task.md) builds and shows how to copy.

A scripted user is a Python function that gets the round number, the instruction and the previous round's result, and returns the next prompt, or `None` to stop. This one gives Claude Code only the data at first, and the whole task with the score if the soft check fails:

```python
import asyncio
from pathlib import Path

import benchflow as bf
from benchflow import FunctionUser, RolloutConfig, RoundResult, Scene


def coach(round: int, instruction: str, rr: RoundResult | None) -> str | None:
    if round == 0:
        return instruction.split("\n\n")[0]          # the data, but not what to write
    reward = (rr.rewards or {}).get("reward", 0) if rr else 0
    if reward >= 1.0:
        return None                                  # passed: stop
    return f"Not yet: the tests gave {reward}. The full task:\n{instruction}"


config = RolloutConfig(
    task_path=Path("tasks/half-life"),
    scenes=[Scene.single(agent="claude-agent-acp", model="xiaomi/mimo-v2.6-flash")],
    user=FunctionUser(coach),
    max_user_rounds=2,
)
result = asyncio.run(bf.run(config))
print("rewards:", result.rewards)
print("error:", result.error)
```

The prompts the agent received, from the trial's `prompts.json`:

```text
round 0: The file `/app/data/counts.csv` holds the counts recorded in each minute of a two-hour measurement of a radioactive sample (columns `minute` and `counts`). The room adds a constant background, which was not measured separately.
round 1: Not yet: the tests gave 0.0. The full task:
         The file `/app/data/counts.csv` holds the counts ... Estimate the sample's half-life in minutes ...
```

Round 0 had the data but not what to write, so its soft check gave 0.0 (14 tool calls, 262 s, 314,848 tokens, from the trial's `user_rounds.jsonl`). Round 1 had the whole task and was still working when it reached the task's 15-minute limit for one prompt, so the run stopped it, and `user_rounds.jsonl` has no record of it. The answer was already written, so the final check passed:

```text
rewards: {'reward': 1.0}
error: Agent prompt exceeded wall-clock budget 900s
```

An earlier run of the same task finished round 1 in 865 s. Give a multi-round task's agent time for its longest round, and read `error` as well as the reward.

In 0.9.2, know this before you write one:

- **`rr.verifier_output` is `None`.** The soft check's reward reaches the user function, as `rr.rewards`; the test's output doesn't. Decide from the rewards. 0.10.0rc2 passes the test's output.
- **The soft check is less isolated than the final one.** Between rounds the run warns `Mid-loop verifier isolation skipped at round 0: ... Cannot lock symlink /tests. Final verify() still hardens before scoring`. The reward that counts comes from the hardened final check.

## Several agents in one sandbox

A task can name roles and scenes in its `task.md`, with each scene's prompt in a `## scene:<name>` section. Two OpenCode agents, one writing a script and the other running it ([the task](../examples/two-roles/task.md)):

```markdown
---
agents:
  roles:
    writer:
      agent: opencode
      model: deepseek/deepseek-v4-flash
    runner:
      agent: opencode
      model: deepseek/deepseek-v4-flash
scenes:
  - name: write
    turns:
      - role: writer
  - name: run
    turns:
      - role: runner
---
# two-roles

## scene:write

Write a Python script, `/app/count.py`, that prints how many data rows `/app/data/counts.csv` has; the header row doesn't count. Don't run it.

## scene:run

Run `python3 /app/count.py` and write what it prints to `/app/rows.txt`.
```

Get the task from the BenchFlow repository, then run it without `-a` or `-m`; the task names its agents:

```bash
git clone --depth 1 https://github.com/benchflow-ai/benchflow.git
mkdir -p tasks && cp -r benchflow/docs/examples/two-roles tasks/two-roles
bench run tasks/two-roles
```

```text
[two-roles] [Step] scene-0-turn-0-writer scene=write role=writer
[two-roles] Prompt 1/1: Write a Python script, `/app/count.py`, that prints how many data rows `/app/dat...
[two-roles]   → end_turn, 3 total tool calls
[two-roles] [Step] scene-1-turn-0-runner scene=run role=runner
[two-roles] Prompt 1/1: Run `python3 /app/count.py` and write what it prints to `/app/rows.txt`....
[two-roles]   → end_turn, 2 total tool calls
[two-roles] Running verifier...
[two-roles] Rewards: {'reward': 1.0}
[PASS] two-roles (reward=1.00, tools=5)
Cost:      $0.00, 54,728 tokens
Time:      48s
```

Each role starts as a new agent with a new session, in the same sandbox: files carry over, conversations don't. What tripped us:

- **Each role gets only its scene's text.** The task's `## prompt` section isn't sent; put everything a role needs in its scene.
- **The task's roles win over the command line.** `-m` doesn't reach a role that names no model: our first version had none, and the writer sat idle until `Agent integration failure [no_activity]: 0 tool calls, 0 messages, 0 thoughts, 0 output tokens after the prompt in 300.3 s`. Name each role's model in `task.md`.
- **Controls don't run on such a task.** `-a oracle` prints `--agent oracle was requested, but two-roles pins scene agents (agents.roles / document scenes), which take precedence. Oracle mode will NOT run; the task's scene agent(s) will execute instead.` To check the test, run the oracle on a copy of the task without the roles and scenes.
