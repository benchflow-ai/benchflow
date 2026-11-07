# Run a benchmark

One `bench run` runs an agent on every task in a folder, several at a time, and repeats each task as many times as you ask. Use it for any run bigger than the [Quick start](../start/quick-start.md)'s single task. Before a run you'll publish, read [Before you run an eval](./before-you-run.md); it costs a minute and catches the settings that change scores.

Runs here use MiMo (`-m xiaomi/mimo-v2.6-flash`): export its key and base URL first ([Quick start](../start/quick-start.md#on-another-provider)), or use `-m sonnet` with a Claude login.

Words in the output: a **trial** is one rollout, one agent's attempt at one task; `--trials N` and the `trial-01` to `trial-NN` folders count **repeats**, passes over the whole task set; and `empty` in the Control column of `bench results` is the `nop` agent ([How BenchFlow works](../start/how-it-works.md#the-words)).

You need an agent with a login or key ([Agents and models](./agents.md)) and a sandbox; Docker is the default ([Sandboxes](../scale/sandboxes.md)).

## Run three tasks, three times each

Clone the benchmark at a fixed commit, then point `bench run` at its folder of tasks. Here Claude Code runs three SkillsBench tasks, each three times, three at a time:

```bash
git clone https://github.com/benchflow-ai/skillsbench.git
git -C skillsbench checkout f860ecf02ef53412a030f8661c83d644b739ff01
bench run skillsbench/tasks \
  -i econ-detrending-correlation \
  -i crystallographic-wyckoff-position-analysis \
  -i offer-letter-generator \
  -a claude -m xiaomi/mimo-v2.6-flash --trials 3 -j 3 \
  --save-outputs --jobs-dir jobs/no-skills
```

`--save-outputs` (also `--freeze-workspace`, the name error messages use) keeps each trial's final workspace, so the work can be [regraded](../scale/resume.md#regrade-saved-work) later. Each repeat prints its own summary. The end of the third, with the per-task log lines left out, and the line for all three:

```text
[PASS] econ-detrending-correlation (reward=1.00, tools=13)
[PASS] offer-letter-generator (reward=1.00, tools=14)
[FAIL] crystallographic-wyckoff-position-analysis (reward=0.00, tools=47) (Agent prompt exceeded wall-clock budget 900s)
1 tasks (33%) hit agent wall-clock timeout — error in result.json has the cause (agent_timeout_info has details when set)
Job complete: 2/3 (66.7%), mean_reward=0.67, errors=0, idle_timeouts=0, time=15.5min

• Score: 2/3 (66.7%), mean reward 0.67, errors=0
  ✗ crystallographic-wyckoff-position-analysis: reward 0.0
Outcomes: 3 scored (2 passed, 1 failed), 0 unscored, 0 errored
Cost:      not reported in USD (subscription logins and unpriced models report tokens only), 1,704,360 tokens
Time:      15m 29s
Artifacts: jobs/no-skills/claude-agent-acp/trial-03/2026-11-04__18-24-21
Matrix summary: jobs/no-skills/matrix-summary.json
Matrix: 3 trial job(s), 6/9 trials passed, errors=0 verifier-errors=0
```

Two tasks passed every time; the crystallographic task hit its 15-minute limit in all three repeats.

A registry dataset runs the same way with `-d name@version` instead of a folder; see [Find a benchmark](./find-a-benchmark.md).

## Choose the tasks

| Flag | Selects |
|---|---|
| `-i NAME` (repeatable) | only tasks whose names match: a name or a glob such as `'econ-*'` |
| `-x NAME` (repeatable) | skip tasks whose names match |
| `-n N` | the first N tasks by name |
| `-n N --sample-seed S` | a random sample of N tasks, the same sample for the same seed |

## Repeat each task

`--trials 3` runs the whole task set three times, one repeat after another. Each repeat is a job of its own, in `<jobs dir>/<agent>/trial-01`, `trial-02`, `trial-03`. The flag, the folders and the `Matrix:` line say trial where these pages say repeat: each `trial-NN` folder is one repeat, holding one trial per task, and `3 trial job(s)` counts the repeats. In 0.9.2, `bench run --help` doesn't list `--trials` (nor `--max-cost-usd`, `--max-tokens` and `--integrity`); `bench run --help-all` does, and so does [Commands](../reference/commands.md#bench-run). Repeats are what [pass@k and intervals](../audit/compare-runs.md) are computed from; one try per task gives a noisy number.

`--trials` needs a local folder or a registry dataset. Given a GitHub address, it stops with:

```text
--trials > 1 needs a task folder or --dataset; a --config file or --source-env run repeats nothing
  next: `bench run --help` lists the flags
```

## Run many trials at once

`-j 3` runs three trials at a time; the default is 4. How high to go depends on your machine, your sandbox provider and your model's rate limit: see [Concurrency and budgets](../scale/cost.md).

## Where results go

```text
jobs/no-skills/
└── claude-agent-acp/
    ├── trial-01/2026-11-04__17-52-13/     one job: a folder per trial, plus summary.json
    ├── trial-02/2026-11-04__18-08-58/
    └── trial-03/2026-11-04__18-24-21/
```

Running the same command again resumes the job where it stopped: trials that finished are kept, the rest run. `--fresh` starts a new job instead. [Read the results](./read-results.md) shows what's in each folder.

## From Python

The same run from Python, here with the oracle on a folder holding the half-life task ([Write a task](../tasks/write-a-task.md) shows how to copy it):

```python
import benchflow as bf

evaluation = bf.Evaluation(
    tasks_dir="tasks",
    jobs_dir="jobs/python",
    config=bf.EvaluationConfig(agent="oracle", environment="docker", concurrency=2),
)
result = evaluation.run_sync()
print(result.passed, result.total, result.score, result.job_dir)
```

```text
1 1 1.0 jobs/python/2026-11-04__18-36-24
```

`EvaluationConfig` takes the same settings as the flags (`agent`, `model`, `reasoning_effort`, `environment` for the sandbox, `concurrency` and more); [CLI and Python](../reference/cli-python.md) maps each flag to its argument. `bf.load_job` reads results back: see [Read the results](./read-results.md#in-python).
