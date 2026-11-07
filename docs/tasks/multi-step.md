# Multi-step tasks

> Needs BenchFlow `0.10.0rc1`, a prerelease: [install it](../start/install.md#install-the-010-prerelease). 0.9.1 refuses multi-step tasks with `UnsupportedTaskFeatureError: ... steps: multi-step tasks are not supported (one prompt per task)`.

A multi-step task gives the agent its work in steps, in one sandbox. After each step, the step's own tests check the workspace, and a step that falls below its `min_reward` ends the task. Use it when the work has stages whose results the later stages build on, and you want to know which stage an agent failed. For several prompts that aren't graded one by one, use a [multi-turn task](./multi-turn.md).

## Declare the steps

This example uses the `task.toml` layout; a `task.md` header takes the same keys, `steps` and `multi_step_reward_strategy` ([Files BenchFlow reads and writes](../reference/files.md)). Each step gets a folder under `steps/` with its instruction, its tests and, for the oracle, its solution:

```text
two-steps/
├── task.toml
├── instruction.md
├── environment/Dockerfile    FROM python:3.12-slim, WORKDIR /app
└── steps/
    ├── write/
    │   ├── instruction.md        Write the number 1 to /app/a.txt.
    │   ├── tests/test.sh
    │   └── solution/solve.sh
    └── append/
        ├── instruction.md        Keep /app/a.txt, and write the number 2 to /app/b.txt.
        ├── tests/test.sh
        └── solution/solve.sh
```

```toml
version = "1.0"
multi_step_reward_strategy = "final"

[agent]
timeout_sec = 300

[verifier]
timeout_sec = 60

[[steps]]
name = "write"
min_reward = 1.0        # below this, the task stops after this step

[[steps]]
name = "append"
```

Each step's test checks what that step should have left, and writes the reward. The two here, and the oracle's two solutions:

```bash
# steps/write/tests/test.sh
if [ "$(cat /app/a.txt 2>/dev/null)" = 1 ]; then echo "PASS: a.txt holds 1"; echo 1; else echo "FAIL: a.txt does not hold 1"; echo 0; fi | tail -1 > /logs/verifier/reward.txt
# steps/append/tests/test.sh
if [ "$(cat /app/a.txt 2>/dev/null)" = 1 ] && [ "$(cat /app/b.txt 2>/dev/null)" = 2 ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
# steps/write/solution/solve.sh
mkdir -p /app && echo 1 > /app/a.txt
# steps/append/solution/solve.sh
echo 2 > /app/b.txt
```

The whole task is in [`docs/examples/multi-step/two-steps`](../examples/multi-step/two-steps/task.toml).

A step without its own `[steps.agent] timeout_sec` gets the task's whole `agent.timeout_sec`, once per step.

## Run it

Get the task from the BenchFlow repository; the oracle runs each step's `solve.sh` and that step's tests in turn:

```bash
git clone --depth 1 https://github.com/benchflow-ai/benchflow.git
mkdir -p tasks && cp -r benchflow/docs/examples/multi-step/two-steps tasks/two-steps
bench run tasks/two-steps -a oracle
```

```text
[two-steps] Step 1/2 write: reward 1.00
[two-steps] Step 2/2 append: reward 1.00
[two-steps] Rewards of the final verify (step 2/2 append): {'reward': 1.0}
[two-steps] Steps: write 1.00 (soft check), append 1.00 (final verify); task reward 1.00, the last step's (append)
[PASS] two-steps (reward=1.00, tools=0)
```

`nop` does nothing, so the first step's check fails and the task stops there:

```bash
bench run tasks/two-steps -a nop
```

```text
[two-steps] Step 1/2 write: reward 0.00, below its min_reward 1: the task stops here and the 1 later step counts 0
[two-steps] Rewards of the final verify (step 1/2 write): {'reward': 0.0}
[two-steps] Steps: write 0.00 (final verify), append not run; task reward 0.00: the last step (append) did not run
[FAIL] two-steps (reward=0.00, tools=0)
```

An agent gets each step's instruction in turn. OpenCode on DeepSeek V4 Flash:

```bash
bench run tasks/two-steps -a opencode -m deepseek/deepseek-v4-flash
```

```text
[two-steps] Step 1/2 write: prompt='Write the number 1 to /app/a.txt.'
...
[two-steps] Step 1/2 write: reward 1.00 after 1 tool call(s)
[two-steps] Step 2/2 append: prompt='Keep /app/a.txt, and write the number 2 to /app/b.txt.'
...
[two-steps] Step 2/2 append: reward 1.00 after 1 tool call(s)
[two-steps] Running verifier...
[two-steps] Rewards of the final verify (step 2/2 append): {'reward': 1.0}
[two-steps] Steps: write 1.00 (soft check), append 1.00 (final verify); task reward 1.00, the last step's (append)
[PASS] two-steps (reward=1.00, tools=2)
```

The agent starts again for each step, with a new session, in the same sandbox: files carry over from one step to the next, the conversation doesn't. Write each step's instruction so it stands on its own.

Give the agent a workspace it can write to. The task's `Dockerfile` sets `WORKDIR /app`, and BenchFlow makes that folder the agent's. An earlier version of this task used `docker_image = "python:3.12-slim"`, with no `WORKDIR`: the workspace was `/root`, which the agent's user can't write, and the agent spent its 300 seconds on `Permission denied` and `su root`.

## How the reward is made

Each step's tests run twice at most. Right after the step, a **soft check** runs on the workspace as the agent left it; it removes start-up files the agent planted, but doesn't stop the agent's processes. After the last step that ran, a **final check** runs with the full protection described in [Stop reward hacking](./reward-hacking.md).

- `final` (the default): the task's reward is the final check of the last step, and 0 if an earlier step stopped the task.
- `mean`: the average of every step's reward, with earlier steps scored by their soft checks and a step that didn't run counting 0.

Use `final` when the reward must resist tampering: under `mean`, a file the agent plants can raise an earlier step's soft reward. Each trial's `steps.json` records every step's reward and where it came from:

```json
{
  "version": 1,
  "strategy": "final",
  "steps": [
    {"name": "write", "reward": 0.0, "source": "final", "soft_reward": 0.0},
    {"name": "append", "reward": null, "source": "not-run"}
  ],
  "reward": 0.0,
  "stopped_after": "write"
}
```

Not supported yet: artifacts and health checks per step, and a separate verifier sandbox; a run refuses them before it starts. `bench regrade` doesn't rescore multi-step trials.
