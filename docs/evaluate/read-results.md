# Read the results

Every run leaves a job folder with a summary, and a folder per trial with its reward, what the agent did and what it cost. Read them from the shell with `bench results`, in the browser with `bench view`, or in Python with `bf.load_job`. The examples read the first repeat of the [Run a benchmark](./run-a-benchmark.md) run: Claude Code on three SkillsBench tasks.

## The summary

A run ends with its score, how each trial ended, the cost and where the files are. This one is the [Quick start](../start/quick-start.md#on-another-provider)'s single trial of Claude Code on MiMo; the rest of the page reads the three-task run:

```text
✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
Outcomes: 1 scored (1 passed, 0 failed), 0 unscored, 0 errored
Cost:      not reported in USD (subscription logins and unpriced models report tokens only), 511,746 tokens
Time:      6m 32s
Artifacts: jobs/2026-11-04__16-43-38
```

`Score` counts passed trials; `mean reward` averages the rewards, which differs when a task gives partial credit. `Outcomes` splits trials that were scored from those that weren't ([When a trial fails](./failed-trials.md)).

0.10.0rc2 counts every figure by one [scoring rule](../reference/scoring-rule.md), `benchflow.score.v1`: every finished trial is in n, and only a scored pass counts 1, so an error or a trial with no score counts 0. Under the `Score` they print its 95% interval, over tasks, and who ended the trials with no score (the model provider, or BenchFlow's time limits). They warn when a task has fewer than 3 trials, and when more than 5% of the trials errored, timed out or met a rate limit. `Outcomes` and `bench results` name the trials with no score in the same words: `1 unscored (1 verifier error), 2 errored (1 with no verdict)`.

A folder whose trials ran more than one setting gets no single score. The settings are the agent and its harness version, the model that served the calls, their route, the effort they carried, the task set and the BenchFlow version. Each setting gets its own line with its own score and interval, and a warning names what differs (`mixed settings: ...`). Runs that differ only in effort are never pooled.

## A job, trial by trial

```bash
bench results jobs/no-skills/claude-agent-acp/trial-01/2026-11-04__17-52-13
```

```text
jobs/no-skills/claude-agent-acp/trial-01/2026-11-04__17-52-13 (evaluation): 2/3 attempted passed (67%), 2/3 scored passed (67%), 0 verifier errors, 0 unscored, 0 control runs left out
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Task                                       ┃ Reward ┃ Exec      ┃ Assess ┃ Control ┃ Agent            ┃ Model                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ crystallographic-wyckoff-position-analysis │ 0      │ timed_out │ scored │         │ claude-agent-acp │ xiaomi/mimo-v2.6-flash │
│ econ-detrending-correlation                │ 1      │ completed │ scored │         │ claude-agent-acp │ xiaomi/mimo-v2.6-flash │
│ offer-letter-generator                     │ 1      │ completed │ scored │         │ claude-agent-acp │ xiaomi/mimo-v2.6-flash │
└────────────────────────────────────────────┴────────┴───────────┴────────┴─────────┴──────────────────┴────────────────────────┘
```

Two columns describe each trial. **Exec** says how the agent's turn ended: `completed`, `timed_out`, or an error. **Assess** says whether the verifier scored the result: `scored`; `error` when the verifier itself failed, as a verifier sandbox that never started; or `unscored` when no score came back for another reason. In 0.10.0rc1 the run's own summary counts both of the last two as unscored. In 0.10.0rc2 the first line counts the trials with no score the way the run's summary does, `0 unscored, 0 errored` here: unscored when the verifier gave none, errored when the run stopped before a verdict. **Control** marks the control runs, `oracle` and `empty` (the `nop` agent), which the totals leave out unless you pass `--include-controls`. The crystallographic trial ran out of time, and the verifier still scored what it left: 0. `bench results` on a folder of jobs lists them; on two jobs, it compares them task by task ([Compare runs](../audit/compare-runs.md)).

## Where the reward is

Each trial's `result.json` holds the reward under `rewards.reward`, with how the trial ended. The timed-out trial's, trimmed:

```json
{
  "task_name": "crystallographic-wyckoff-position-analysis",
  "rewards": {"reward": 0.0},
  "error": "Agent prompt exceeded wall-clock budget 900s",
  "error_category": "timeout",
  "agent_timeout_info": {"reason": "wall_clock_timeout", "timeout_sec": 900.0, "n_tool_calls": 28, ...},
  "n_tool_calls": 28
}
```

The job's `summary.json` holds the counts for the whole job, and `results.jsonl` one line per trial.

## What the agent did

| File in the trial folder | Holds |
|---|---|
| `trajectory/acp_trajectory.jsonl` | every message, thought and tool call the agent reported, one event per line |
| `trajectory/llm_trajectory.jsonl` | each request BenchFlow's model proxy sent and the response it got |
| `agent/` | the agent's own log |
| `verifier/test-stdout.txt` | what the tests printed |

`bench view <job or trial folder>` shows the same in a browser, at the `localhost` address it prints (the same page at `127.0.0.1` is refused): the conversation, each tool call with its output, and the verifier's result. [Trajectory and rollout formats](../train/export.md#trajectory-and-rollout-formats) describes each file.

## Tokens, cost and time

`result.json` has the trial's tokens under `final_metrics`:

```json
{"total_prompt_tokens": 612423, "total_completion_tokens": 31080, "total_cached_tokens": 564160, "total_cost_usd": null}
```

The same counts are in its `agent_result` block, with where each came from ([Usage and telemetry](../scale/telemetry.md#what-a-trial-records)).

The cost is `null` here because BenchFlow has no price for this model; for a priced one it's in dollars. `timing.json` has seconds per step of the trial. [Usage and telemetry](../scale/telemetry.md) explains where each number comes from.

## Everything in a trial's folder

The folder of the [Quick start](../start/quick-start.md#on-another-provider)'s Claude Code trial on MiMo, on 0.9.1; a trial on a subscription login has no `llm_trajectory.jsonl` or `provider_requests.jsonl`, since it doesn't go through BenchFlow's model proxy:

| File | Holds |
|---|---|
| `result.json` | the outcome: the reward, how the trial ended, tokens and cost, and where the task came from (its keys are below) |
| `config.json` | what the trial was asked to run: BenchFlow's version, the agent, model and effort, the sandbox and its user, the time limits, and the variables passed to the agent with their values, leaving out any that look secret (`agent_env`) |
| `timing.json` | seconds for each step, from `slot_wait` and `environment_setup` to `agent_execution`, `verifier` and `total`, with the time spent waiting on the provider (`provider_wait`) |
| `prompts.json` | the prompts the agent was sent |
| `agent/` | the agent's install log (`install-stdout.txt`), its own output and log, and its trajectory as it reported it |
| `trajectory/acp_trajectory.jsonl` | every message, thought and tool call, one event a line |
| `trajectory/llm_trajectory.jsonl` | each model request and response that went through BenchFlow's proxy, with its duration |
| `trajectory/provider_requests.jsonl` | the proxy's timing events for each request: its start, the upstream call's start and end, the response |
| `verifier/` | what the test printed (`test-stdout.txt`), the reward it wrote (`reward.txt`), and its own report when it writes one (`ctrf.json` here) |
| `rewards.jsonl` | each reward event; here one, the verifier's final reward |
| `results.jsonl` | the trial as one row of prompt and completion (the job folder's own `results.jsonl` has one line per trial) |
| `trainer/` | the trial for training: ADP, ATIF and verifiers-style rows ([Trajectory and rollout formats](../train/export.md#trajectory-and-rollout-formats)) |
| `artifacts/`, `artifacts-manifest.json` | the files the task declares as artifacts, collected after the agent's turn, and their manifest |
| `cloud-metadata.json`, `firewall-power.json`, `locked-system-path.json`, `kill-before-grading.json`, `trust-state.json` | the protections BenchFlow set up and what each found: the cloud metadata block, whether anything in the sandbox could take the firewall, whether the folders on `PATH` are root's alone, the processes it stopped before grading, and the trial's trust record |

`result.json`'s keys, by what they answer:

| Question | Keys |
|---|---|
| What ran | `task_name`, `rollout_name`, `agent`, `agent_name` (the name the agent gave itself), `model`, the skill settings (`skill_mode` and the keys with `skills` in their names), `scenes`, `loop` |
| The score | `rewards`; `rewards.reward` is the one counted |
| Work and usage | `n_tool_calls`, `n_skill_invocations`, `n_prompts`; `agent_result` (the same counts with tokens, cost, `usage_source` and `price_source`); `final_metrics` (token totals and cost); `trajectory_summary` (steps by kind); `usage_tracking` (how usage was captured) |
| How it ended | `error` and `error_category`; `verifier_error` and `verifier_error_category`; `export_error`; one `*_info` block per kind of ending, with its details (`agent_timeout_info`, `idle_timeout_info`, `api_error_info`, `suspected_api_error_info`, `usage_limit_info`, `integration_failure_info`, `sandbox_startup_info`, `transport_error_info`, `verifier_timeout_info`, `reward_out_of_range_info`); `partial_trajectory` and `trajectory_source`: `acp` from the agent protocol, `partial_acp` when the session timed out or crashed, `scraped` from files the agent itself could have written |
| When and where | `started_at`, `finished_at`, `timing` (as in `timing.json`), `source` (the task's repository, its path and the commit it resolved to), `task_digest`, `sandbox_id`, `verifier_isolation` (`shared` here: the verifier ran in the agent's sandbox) and where that came from (`verifier_isolation_source`), `cloud_metadata` |

A task run from a repository without a commit still records the one it resolved to, in `source.resolved_sha`: the Quick start's MiMo run resolved to `f860ecf0`.

## In Python

```python
import benchflow as bf

job = bf.load_job("jobs/no-skills/claude-agent-acp/trial-01/2026-11-04__17-52-13")
for row in job.to_records():
    print({k: row.get(k) for k in ("task_name", "reward", "execution", "assessment", "cost_usd")})
d = job.denominators()
print(d.pass_rate_scored, d.pass_rate_attempted)
```

```text
{'task_name': 'crystallographic-wyckoff-position-analysis', 'reward': 0.0, 'execution': 'timed_out', 'assessment': 'scored', 'cost_usd': None}
{'task_name': 'econ-detrending-correlation', 'reward': 1.0, 'execution': 'completed', 'assessment': 'scored', 'cost_usd': None}
{'task_name': 'offer-letter-generator', 'reward': 1.0, 'execution': 'completed', 'assessment': 'scored', 'cost_usd': None}
0.6666666666666666 0.6666666666666666
```

`job.to_records()` gives one dict per trial, ready for a data frame. `denominators()` gives the pass rate two ways: over scored trials and over every attempted trial. They match here; they differ when some trials ended unscored or errored, and [Compare runs](../audit/compare-runs.md#what-counts-in-the-denominator) says which to report.
