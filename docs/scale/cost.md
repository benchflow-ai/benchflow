# Concurrency and budgets

`-j` sets how many trials run at once, and a budget stops a job before it spends more than you meant. Use both on any run larger than a handful of trials, and watch the run's own progress lines while it goes.

Runs here use MiMo (`-m xiaomi/mimo-v2.6-flash`): export its key and base URL first ([Quick start](../start/quick-start.md#on-another-provider)), or use `-m sonnet` with a Claude login.

## Concurrency

```bash
bench run tasks/ -a claude -m xiaomi/mimo-v2.6-flash -j 8
```

`-j` (or `--concurrency`) is the number of trials running at a time; the default is 4. On Docker, BenchFlow also prepares one sandbox ahead of its turn (`--warm-pool` changes how many). With `--trials`, the repeats run one after another, each `-j` at a time.

The right number is set by the slowest of: your machine (each trial is a container; check CPU, memory and disk), your sandbox provider's limits, and your model provider's rate limit. A rate limit shows as trials ending with `provider_rate_limit`; lower `-j`.

## Budgets

A budget caps a job. At the cap no new trial starts; whether running trials stop depends on the budget. Trials a budget kept from running are listed apart in `summary.json`, not counted as failures.

| Flag | Caps | Checked | At the cap, running trials |
|---|---|---|---|
| `--max-cost-usd` | dollars, over trials that report a cost, retries included | between starts, from finished trials | are cancelled |
| `--max-tokens` | total tokens of finished trials, retries included | between starts, from finished trials | stop as with `--max-cost-usd` |
| `--max-sandbox-seconds` | trial wall-clock seconds | as trials run, running ones included | are cancelled |
| `--max-rollouts` | trials started, retries included | at each start | finish |

A trial's cost is known only when it finishes, so a job can pass a cost or token cap: by about one trial's cost when trials run one at a time, by more when several finish together. A cap far below one trial's cost shows the overshoot:

```bash
bench run skillsbench/tasks \
  -i econ-detrending-correlation \
  -i crystallographic-wyckoff-position-analysis \
  -a opencode -m deepseek/deepseek-v4-flash -j 1 \
  --max-cost-usd 0.004
```

`skillsbench/` is a clone of SkillsBench at commit f860ecf0 ([how](../evaluate/run-a-benchmark.md#run-three-tasks-three-times-each)). The dollars below are 0.9.1's placeholder price for DeepSeek ([Usage and telemetry](./telemetry.md#cost-and-models-without-a-price)), so read them for how a cap behaves, not for what DeepSeek charges.

```text
budget reached: $0.0385 USD spent of a $0.0040 cap; stopping: no new rollouts, 0 running trial(s) cancelled
[PASS] crystallographic-wyckoff-position-analysis (reward=1.00, tools=35)
Job complete: 1/1 (100.0%) run, 1/2 (50.0%) of the plan, mean_reward=1.00, errors=0, idle_timeouts=0, time=3.9min

• Score: 1/1 run (100.0%), mean reward 1.00, errors=0; 1 of 2 planned tasks not run (budget), so 1/2 (50.0%) of the plan
Budget: budget reached: $0.0385 USD spent of a $0.0040 cap; 0 running trial(s) cancelled, 1 not started (not counted as 
failures; resume the job to run them)
Outcomes: 1 scored (1 passed, 0 failed), 0 unscored, 0 errored
Cost:      $0.04, 1,801,711 tokens
```

> **An unpriced model has no dollar cap.** `--max-cost-usd` needs a model BenchFlow has a price for; for one it doesn't, such as MiMo or a model newer than its price table, cap tokens instead.

The run's cost line counts the trials' own model use, and so do these caps as far as we checked: a rubric reviewer's tokens aren't in it ([Rubric review](../tasks/rubric-review.md#choose-the-reviewer) says where they're recorded), and neither are your sandbox provider's charges. With `--trials`, each repeat is a job with its own cap.

The cap counts everything the job has spent, across resumes. Rerunning the same command resumes the job, and with the same cap it stops at once, though the summary above says to resume:

```text
Resuming into existing job directory: 2026-11-04__18-18-39 (finished tasks are reused; pass a new job_name, or --fresh on the CLI, for a new run)
budget reached: $0.0385 USD spent of a $0.0040 cap; stopping: no new rollouts, 0 running trial(s) cancelled
...
1 of 2 planned tasks not run (budget), so 1/2 (50.0%) of the plan
```

To run what the budget skipped, start a new job with a cap that fits (`--fresh`), which runs every task again.

## Watch a long run

Each trial prints a progress line about every 45 seconds, with its tool calls so far and the last one:

```text
[econ-detrending-correlation]   … 2.3min, 15 tool calls (last: python3 - <<'EOF')
```

`bench view jobs/<job>` opens a job in the browser. What each trial used, in tokens, cost and time, is on [Usage and telemetry](./telemetry.md).

### How busy the run is (0.10)

> Needs BenchFlow `0.10.0rc1`, a release candidate that isn't published yet, so you can't install it from PyPI ([what you can do](../start/install.md#install-the-010-prerelease)).

The run says how busy it is two minutes after it starts and every five minutes after, and warns when slots sit idle while tasks wait or when the agents are rarely waiting on the model:

```text
Utilization, last 5 min: 1.0 of 1 slot running (0.0 in setup, 1.0 with the agent); 0 tasks waiting; a model request in flight 90% of the agents' time (0.9 requests at once on average)
```
