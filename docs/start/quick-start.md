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
Run card (jobs/2026-11-18__14-01-29/run-card.json):
  Tasks:     1 from benchflow-ai/skillsbench/tasks/econ-detrending-correlation at f860ecf0
  Agent:     oracle, the task's reference solution
  Model:     none (a control run)
...
Job: 1 task, 0 done, 1 to run (concurrency=4, warm pool: up to 1 prepared sandbox)
[econ-detrending-correlation] Starting environment: econ-detrending-correlation
[econ-detrending-correlation] Building the task image for econ-detrending-correlation: the first run of a task builds it, later runs reuse it (this can take a few minutes)
...
[econ-detrending-correlation] Built the task image for econ-detrending-correlation in 3:40
[econ-detrending-correlation] Setting up sandbox user: agent
[econ-detrending-correlation] Sandbox user agent ready (workspace=/root)
[econ-detrending-correlation] Oracle mode: running oracle solve.sh
[econ-detrending-correlation] Running verifier...
[econ-detrending-correlation] Rewards: {'reward': 1.0}
...
[PASS] econ-detrending-correlation (reward=1.00, tools=0)
Job complete: 1/1 (100.0%), mean_reward=1.00, errors=0, idle_timeouts=0, time=7.9min

Task                         Outcome  Reward  Cost  Tokens    Time
econ-detrending-correlation  passed        1  $0         —  7m 14s

Score       Model cost  Tokens  Wall time
1/1 passed  $0          —       7m 52s
Results: jobs/2026-11-18__14-01-29/
```

`--fresh` starts a new job even if this folder already holds a run of the task, such as the docs home's first command; without it, `bench run` would print that earlier run's result. With no `-a`, `bench run` uses the **oracle**: it runs the task's reference solution (`solve.sh`) in a fresh sandbox, then the task's tests score the result. A reward of 1.0 means the task can be solved in this sandbox. `TASK` names the task and SkillsBench's commit f860ecf0, the version these outputs came from, and the later steps reuse it; give the full hash, since a short one isn't found. The [run card](../evaluate/before-you-run.md#the-run-card-010) says what the run will do before anything starts. The `Job:` line says how many trials run at once (`concurrency`) and how many sandboxes are prepared ahead of their turn (the warm pool). The first run of a task builds its image, here in 3:40 on a busy machine; later runs reuse it. The run ends with a row per trial, the score, and the job folder on its `Results:` line.

## Check that doing nothing scores 0

```bash
bench run $TASK -a nop --fresh
```

```text
...
[econ-detrending-correlation] Running verifier...
[econ-detrending-correlation] Rewards: {'reward': 0.0}
[FAIL] econ-detrending-correlation (reward=0.00, tools=0)
...
Job complete: 0/1 (0.0%), mean_reward=0.00, errors=0, idle_timeouts=0, time=5.8min
nop control: 0/1, as expected: doing nothing passes none of the tasks

Task                         Outcome  Reward  Cost  Tokens    Time
econ-detrending-correlation  failed        0  $0         —  5m 10s

Score       Model cost  Tokens  Wall time
0/1 passed  $0          —       5m 50s
Results: jobs/2026-11-18__14-11-28/
```

`nop` is an agent that does nothing. A task that passes with it is broken. `--fresh` starts a new job even if one with the same settings exists, so you can run these steps again. Without it, the same command run again resumes its job, and a command with other settings, such as another agent, starts a new job by itself ([Resume, retries and regrade](../scale/resume.md#start-fresh-instead)).

## Run Claude Code on it

Claude Code needs a Claude login or an Anthropic API key. With a Claude subscription, `claude setup-token` prints a token: export it as `CLAUDE_CODE_OAUTH_TOKEN`. With an API key, export `ANTHROPIC_API_KEY` instead. This run took about two minutes:

```bash
export CLAUDE_CODE_OAUTH_TOKEN=...   # from `claude setup-token`; or export ANTHROPIC_API_KEY=...
bench run $TASK -a claude -m sonnet --fresh --jobs-dir jobs/claude-sonnet
```

It prints the same rows and footer as the oracle's run, with the agent's tool calls, cost and tokens; its `Results:` line names a job folder under `jobs/claude-sonnet/`. The footer's tokens count cache reads; `bench eval metrics` gives input tokens without them, so its in and out add up to less. After the `@`, BenchFlow takes a branch, a tag or a full commit hash. `-m` takes a name the agent offers. In 0.9.2, Claude Code on a subscription login offers `default`, `sonnet`, `opus`, `haiku` and `claude-fable-5-1`, and any other name stops the trial before a model call:

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
```

On a provider run, the log says it skips the agent's own model setting: the model reaches Claude Code through the proxy's settings instead, and each request's model is in the trial's `trajectory/llm_trajectory.jsonl`. BenchFlow has no price for MiMo, so the footer gives its tokens and `—` for its cost.

Codex, OpenCode, Pi and other agents work the same way; [Agents and models](../evaluate/agents.md) lists them and how each logs in.

## See what happened

`bench results` takes the job folder a run printed on its `Results:` line. Ours was the oracle's, `jobs/2026-11-18__14-01-29/`; use the one your run printed:

```bash
bench results jobs/2026-11-18__14-01-29
```

```text
jobs/2026-11-18__14-01-29 (evaluation): 1/1 attempted passed (100%), 1/1 scored passed (100%), 0 unscored, 0 errored, only control runs, so they are counted
  Score by benchflow.score.v1: 100.0% (1 of 1 trials; 95% interval 20.7% to 100.0%, one task, its trials as the sample), 1 task
  k < 3: every task has 1 trial; with fewer than 3 tries per task the interval and pass@k say little about each task (run 3 or more trials per task)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ Task                        ┃ Reward ┃ Exec      ┃ Assess ┃ Control ┃ Agent  ┃ Model ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ econ-detrending-correlation │ 1      │ completed │ scored │ oracle  │ oracle │       │
└─────────────────────────────┴────────┴───────────┴────────┴─────────┴────────┴───────┘
```

`bench results jobs` with no job name lists every job in the folder.

Each run is a **job**: a folder under `jobs/`, holding one **trial** (one agent's attempt at one task) per task. In a trial's folder, `result.json` holds the reward (`rewards.reward`), `trajectory/acp_trajectory.jsonl` holds every message and tool call the agent made, and `verifier/test-stdout.txt` holds what the tests printed. `bench view jobs/<job>` opens the same in a browser.

## From Python

The same run from your own code, after `uv add benchflow` in your project ([Use it from Python](./install.md#use-it-from-python)):

```python
import benchflow as bf

task = bf.resolve_source(  # the task the commands above ran, at the same commit
    "benchflow-ai/skillsbench",
    path="tasks/econ-detrending-correlation",
    ref="f860ecf02ef53412a030f8661c83d644b739ff01",
)
result = bf.run_sync(
    bf.RolloutConfig(
        task_path=task,  # a task folder
        agent="oracle",
        environment="docker",
        jobs_dir="jobs/python",
    )
)
print(result.reward, result.rollout_dir)
print(bf.load_job(result.rollout_dir.parent))  # the job's summary
```

`bf.resolve_source` fetches the task's folder (into `.cache/` here) and returns its path; any task folder works in its place. The run prints its reward and trial folder, then the job's summary:

```text
1.0 jobs/python/2026-11-18__14-15-23/econ-detrending-correlation__a8ca52bd
**jobs/python/2026-11-18__14-15-23** (directory): 1 trial(s) of 1 task(s); oracle · no model (only control runs, so they are counted)
- Solve rate: 100.0% (1 of 1 trials, passed (reward = 1)), 95% interval 20.7% to 100.0% (wilson)
- Cost: none (control runs call no model)
```

[`quickstart.py`](../examples/python-sdk/quickstart.py) goes further in runnable cells: the oracle, a real agent and its trajectory, several runs at once, and reading and comparing jobs.

## Next

- Run a whole benchmark, or many tasks at once: [Run a benchmark](../evaluate/run-a-benchmark.md).
- Write your own task: [Write a task](../tasks/write-a-task.md).
- How the pieces fit: [How BenchFlow works](./how-it-works.md).
