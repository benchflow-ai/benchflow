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

`skillsbench/` is a clone of SkillsBench at commit f860ecf0 ([how](../evaluate/run-a-benchmark.md#run-three-tasks-three-times-each)). The dollars below come from the placeholder price 0.9.1 and 0.9.2 have for DeepSeek ([Usage and telemetry](./telemetry.md#cost-and-models-without-a-price)): use them to see how a cap behaves, and DeepSeek's rate card for what a run costs.

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

### Per-trial budgets (0.10)

> Needs BenchFlow `0.10.0rc3`, a release candidate on PyPI that you install by its version ([Install](../start/install.md#install-the-010-prerelease)).

A per-trial budget caps each trial, not the job, and it is enforced while the trial runs:

| Flag | Run config | Task (`[agent]`) | Caps each trial's |
|---|---|---|---|
| `--token-budget` | `token_budget` | `token_budget` | tokens: every model call's input, cached input included, plus its output, as `total_tokens` counts them |
| `--turn-budget` | `turn_budget` | `turn_budget` | turns: one turn is one model call |

A measure the run sets replaces the task's. BenchFlow's model proxy enforces both: it sees every model call of the trial, a subagent's or a side call's (a session title) among them, and counts them all. Before each call it compares what the trial has used with its budget; once a measure is spent it refuses the call with HTTP 400 (`trial_budget_spent: ...`), and BenchFlow stops the agent within a second or two. A call's tokens are known only when it ends, so the call that spends the token budget completes, and so does a call already under way beside it: a trial can pass its token budget by those calls, never by a call that starts after. A turn budget is exact.

A budget below what the task takes shows the stop, here on the same SkillsBench task:

```bash
bench run skillsbench/tasks -i econ-detrending-correlation \
  -a opencode -m deepseek/deepseek-flash -j 1 --token-budget 60000
```

```text
  Per trial: budget 60,000 tokens (the run's), a turn being one model call; the model proxy counts every call of the trial, subagents' and side calls included, refuses the next once a budget is spent, and the agent is stopped and graded
...
[econ-detrending-correlation] trial budget reached: the token budget of 60,000 is spent (63,260 tokens, 6 turns used when the model proxy refused the next call); the agent was stopped and the verifier graded what it left
[FAIL] econ-detrending-correlation (reward=0.00, tools=7) (trial budget reached: the token budget of 60,000…)
Run card, after the run:
  Per trial: budget 60,000 tokens; 1 of 1 trials stopped at it (1 by tokens); tokens used: median 63,260, most 63,260; turns used: median 6, most 6
...
Outcomes: 1 scored (0 passed, 1 failed), 0 unscored, 0 errored
  1 stopped at their trial budget (1 by tokens; each counts 0 unless its tests passed): econ-detrending-correlation
      next: trial_budget in each trial's result.json has what it used; raise --token-budget or --turn-budget if the task needs more
Cost:      $0.01, 63,260 tokens
```

The six turns are five of the agent's calls and OpenCode's session-title call. The fifth agent call took the trial from 42,793 tokens to 63,260, and the proxy refused the call after it.

The verifier then grades what the agent left, as after a timeout: the trial counts 0 unless its tests passed, and it is never retried. Its error is `trial budget reached: ...` (`bench explain trial_budget`), and `result.json`'s `trial_budget` records it:

| Field | Meaning |
|---|---|
| `budget` | the trial's `tokens` and `turns` (null for a measure it had none of) |
| `source` | where each measure came from: `run` or `task` |
| `used` | what the trial used: its `tokens` (`total_tokens`) and `turns` (the model calls the proxy forwarded) |
| `subagents` | what its subagents used, from the agent's own session log (Claude Code marks a subagent's calls, Codex a sub-agent's thread): `turns`, `tokens`, and the log's whole `of_turns` and `of_tokens`; null without such a log. BenchFlow keeps the log for a trial with a budget |
| `stopped_by` | `tokens` or `turns`, the measure that stopped it; null when the trial ended within its budget |
| `stopped_at` | the proxy's count when it refused the next call |
| `refused_calls` | the calls the proxy refused for the budget |

The run card's `Per trial:` line gives the budget before the run and, after it, how many trials it stopped and what they used; `summary.json` counts the trials it stopped beside the score (`budget_stops`, `budget_stop_rate`) and totals their use in `run_checks.trial_budget`.

The proxy is the only place every call can be counted, so an agent whose calls go around it cannot be held to a budget: a subscription login, an endpoint and token of its own, or an agent that calls its own model. `bench run` refuses such a run before it makes a job, and a trial whose task sets a budget is refused before its agent starts, as an `agent integration failure [agent_budget]`; give the agent an API key instead.

## Watch a long run

Each trial prints a progress line about every 45 seconds, with its tool calls so far and the last one:

```text
[econ-detrending-correlation]   … 2.3min, 15 tool calls (last: python3 - <<'EOF')
```

`bench view jobs/<job>` opens a job in the browser. What each trial used, in tokens, cost and time, is on [Usage and telemetry](./telemetry.md).

### How busy the run is (0.10)

> Needs BenchFlow `0.10.0rc3`, a release candidate on PyPI that you install by its version ([Install](../start/install.md#install-the-010-prerelease)).

The run says how busy it is two minutes after it starts and every five minutes after, and warns when slots sit idle while tasks wait or when the agents are rarely waiting on the model:

```text
Utilization, last 5 min: 1.0 of 1 slot running (0.0 in setup, 1.0 with the agent); 0 tasks waiting; a model request in flight 90% of the agents' time (0.9 requests at once on average)
```
