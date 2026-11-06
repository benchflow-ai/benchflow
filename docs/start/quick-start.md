# Quick start

BenchFlow runs an AI agent on a task in a sandbox, then scores what the agent left with the task's own tests. Here you score one task three ways: with its reference solution, with an agent that does nothing, and with Claude Code. You need BenchFlow [installed](./install.md) and Docker running.

## Score it with the reference solution

The task is `econ-detrending-correlation` from [SkillsBench](https://github.com/benchflow-ai/skillsbench), a public benchmark: the agent detrends two economic time series and reports their correlation, and the task's tests check the number. This run and the next work without a model or a key:

```bash
TASK=benchflow-ai/skillsbench/tasks/econ-detrending-correlation@f860ecf02ef53412a030f8661c83d644b739ff01
bench run $TASK --fresh
```

```text
Fetching tasks/econ-detrending-correlation from benchflow-ai/skillsbench (https://github.com/benchflow-ai/skillsbench.git) ...
Job: 1 task, 0 done, 1 to run (concurrency=4, warm pool: up to 1 prepared sandbox)
[econ-detrending-correlation] Starting environment: econ-detrending-correlation
[econ-detrending-correlation] Setting up sandbox user: agent
[econ-detrending-correlation] Sandbox user agent ready (workspace=/root)
[econ-detrending-correlation] Oracle mode: running oracle solve.sh
[econ-detrending-correlation] Running verifier...
[econ-detrending-correlation] Rewards: {'reward': 1.0}
[PASS] econ-detrending-correlation (reward=1.00, tools=0)
Job complete: 1/1 (100.0%), mean_reward=1.00, errors=0, idle_timeouts=0, time=0.2min

✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
Outcomes: 1 scored (1 passed, 0 failed), 0 unscored, 0 errored
Cost:      none: no model tokens were used
Time:      15s
Artifacts: jobs/2026-11-04__16-43-09
Summary:   jobs/2026-11-04__16-43-09/summary.json
View:      bench view ./jobs/2026-11-04__16-43-09
```

`--fresh` starts a new job even if this folder already holds a run of the task, such as the docs home's first command; without it, `bench run` would print that earlier run's result. With no `-a`, `bench run` uses the **oracle**: it runs the task's reference solution (`solve.sh`) in a fresh sandbox, then the task's tests score the result. A reward of 1.0 means the task can be solved in this sandbox. `TASK` names the task and SkillsBench's commit f860ecf0, the version these outputs came from, and the later steps reuse it; give the full hash, since a short one isn't found. The `Job:` line says how many trials run at once (`concurrency`) and how many sandboxes are prepared ahead of their turn (the warm pool).

## Check that doing nothing scores 0

```bash
bench run $TASK -a nop --fresh
```

```text
...
[FAIL] econ-detrending-correlation (reward=0.00, tools=0)
Job complete: 0/1 (0.0%), mean_reward=0.00, errors=0, idle_timeouts=0, time=0.2min

• Score: 0/1 (0.0%), mean reward 0.00, errors=0
nop control: 0/1, as expected: doing nothing passes none of the tasks
...
```

`nop` is an agent that does nothing. A task that passes with it is broken. `--fresh` starts a new job even if one with the same settings exists, so you can run these steps again. Without it, the same command run again resumes its job, and a command with other settings, such as another agent, starts a new job by itself ([Resume, retries and regrade](../scale/resume.md#start-fresh-instead)).

## Run Claude Code on it

Claude Code needs a Claude login or an Anthropic API key. With a Claude subscription, `claude setup-token` prints a token: export it as `CLAUDE_CODE_OAUTH_TOKEN`. With an API key, export `ANTHROPIC_API_KEY` instead. This run took about two minutes:

```bash
export CLAUDE_CODE_OAUTH_TOKEN=...   # from `claude setup-token`; or export ANTHROPIC_API_KEY=...
bench run $TASK -a claude -m sonnet --fresh --jobs-dir jobs/claude-sonnet
```

```text
Job: 1 task, 0 done, 1 to run (concurrency=4, warm pool: up to 1 prepared sandbox)
[econ-detrending-correlation] Starting environment: econ-detrending-correlation
[econ-detrending-correlation] claude-agent-acp installed from the cached agent layer
...
[econ-detrending-correlation] ACP model config option 'model' set to: sonnet
[econ-detrending-correlation] Prompt 1/1: In macroeconomics, understanding business cycle dynamics requires separating the...
[econ-detrending-correlation]   … 0.8min, 8 tool calls (last: python3 << 'EOF')
[econ-detrending-correlation]   → end_turn, 11 total tool calls
...
[econ-detrending-correlation] Rewards: {'reward': 1.0}
[econ-detrending-correlation] Cost estimated from Claude Code's session log: $0.2330 (claude-code-cost)
[PASS] econ-detrending-correlation (reward=1.00, tools=11)
Job complete: 1/1 (100.0%), mean_reward=1.00, errors=0, idle_timeouts=0, time=2.2min

✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
Outcomes: 1 scored (1 passed, 0 failed), 0 unscored, 0 errored
Cost:      $0.23, 413,003 tokens
Time:      2m 10s
Artifacts: jobs/claude-sonnet/2026-11-05__02-52-57
```

After the `@`, BenchFlow takes a branch, a tag or a full commit hash. `-m` takes a name the agent offers. In 0.9.1, Claude Code on a subscription login offers `default`, `sonnet`, `opus`, `haiku` and `claude-fable-5-1`, and any other name stops the trial before a model call:

```text
agent integration failure [agent_model]: claude-agent-acp does not offer model 'claude-sonnet-5-5' for this login; it offers: default, sonnet, claude-fable-5-1, opus, haiku
```

### On another provider

A model from another provider runs through BenchFlow's model proxy. The runs on the other pages use Xiaomi's MiMo; the key and the base URL come from your MiMo plan's console ([A model from another provider](../evaluate/agents.md#a-model-from-another-provider)):

```bash
export XIAOMI_API_KEY=...      # your MiMo API key
export XIAOMI_BASE_URL=...     # your MiMo plan's base URL, ending in /v1
bench run $TASK -a claude -m xiaomi/mimo-v2.6-flash --fresh
```

```text
...
[econ-detrending-correlation] Skipping ACP model configuration for claude-agent-acp — launch/env config owns model selection
...
[econ-detrending-correlation]   → end_turn, 20 total tool calls
...
[PASS] econ-detrending-correlation (reward=1.00, tools=20)
Job complete: 1/1 (100.0%), mean_reward=1.00, errors=0, idle_timeouts=0, time=6.5min

✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
Outcomes: 1 scored (1 passed, 0 failed), 0 unscored, 0 errored
Cost:      not reported in USD (subscription logins and unpriced models report tokens only), 511,746 tokens
Time:      6m 32s
Artifacts: jobs/2026-11-04__16-43-38
```

On a provider run, the log says it skips the agent's own model setting: the model reaches Claude Code through the proxy's settings instead, and each request's model is in the trial's `trajectory/llm_trajectory.jsonl`. BenchFlow has no price for MiMo, so its cost is in tokens only.

Codex, OpenCode, Pi and other agents work the same way; [Agents and models](../evaluate/agents.md) lists them and how each logs in.

## See what happened

`bench results` takes the job folder a run printed on its `Artifacts:` line. Ours was `jobs/claude-sonnet/2026-11-05__02-52-57`; use the one your run printed:

```bash
bench results jobs/claude-sonnet/2026-11-05__02-52-57
```

```text
jobs/claude-sonnet/2026-11-05__02-52-57 (evaluation): 1/1 attempted passed (100%), 1/1 scored passed (100%), 0 verifier errors, 0 unscored, 0 control runs left out
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Task                        ┃ Reward ┃ Exec      ┃ Assess ┃ Control ┃ Agent            ┃ Model  ┃ Cost     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ econ-detrending-correlation │ 1      │ completed │ scored │         │ claude-agent-acp │ sonnet │ ~$0.2330 │
└─────────────────────────────┴────────┴───────────┴────────┴─────────┴──────────────────┴────────┴──────────┘
```

`bench results jobs` with no job name lists every job in the folder.

Each run is a **job**: a folder under `jobs/`, holding one **trial** (one agent's attempt at one task) per task. In a trial's folder, `result.json` holds the reward (`rewards.reward`), `trajectory/acp_trajectory.jsonl` holds every message and tool call the agent made, and `verifier/test-stdout.txt` holds what the tests printed. `bench view jobs/<job>` opens the same in a browser.

## Next

- Run a whole benchmark, or many tasks at once: [Run a benchmark](../evaluate/run-a-benchmark.md).
- Write your own task: [Write a task](../tasks/write-a-task.md).
- How the pieces fit: [How BenchFlow works](./how-it-works.md).
