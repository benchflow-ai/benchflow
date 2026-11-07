# Report a result

A score is worth publishing when another researcher can check it: what ran, how, how many times, how noisy the number is, and what was left out. Below: what a result needs, the BenchFlow files that hold each part, and how to share them.

Runs here use MiMo (`-m xiaomi/mimo-v2.6-flash`) with an API key: export its key and base URL first ([Quick start](../start/quick-start.md#on-another-provider)); any model with an API key works the same way.

## What a result needs

| Part | Where it comes from |
|---|---|
| The number, with its interval and how it was computed | [Compare runs](./compare-runs.md) |
| Tasks and tries: how many tasks, how many tries each | `summary.json` of each job; `--trials` |
| The benchmark's version | the Git commit or `name@version`; each trial's `task_digest` |
| The environment as built | the task image's digest and platform (`docker image inspect`): a task's digest covers its files, not what `apt-get` or `pip` installed when the image was built |
| The settings that change a score | the table in [Before you run an eval](../evaluate/before-you-run.md#what-to-write-down) |
| Controls: the oracle and nop results on the same tasks | [Check a benchmark](./check-a-benchmark.md) |
| The grader, when a model grades | the rubric file (it is part of the task's digest), the reviewer's agent and model (`--reviewer-agent`, `--reviewer-model`) and its effort; see [Rubric review](../tasks/rubric-review.md) |
| What was left out, and why | unscored and errored trials, tasks that failed their controls, budget stops |
| Tokens and cost | `result.json` per trial; `price_source` says whether cost is known |
| The trials themselves | the job folders, with trajectories |

A result missing its tries, its interval or its controls can't be told apart from noise or a broken task.

## Collect the evidence

Everything is in the job folders. Keep them whole: `summary.json`, `results.jsonl`, `evaluation.json`, and each trial's folder with `result.json`, `config.json`, `timing.json`, `trajectory/` and `verifier/`. For a machine-readable summary to publish next to them, write one with the run:

```bash
bench run tasks/ \
  -a claude -m xiaomi/mimo-v2.6-flash --trials 3 \
  --summary-out results/summary.json
```

`--summary-out` writes the run's counts, its job folders and the result of any gate, in the documented run-summary schema. Trial folders can hold anything the agent printed, so read a sample before you make them public.

## Write it up

Lead with the number and its uncertainty, then the conditions:

```text
Claude Code (claude-agent-acp 0.81.2, Claude Code 2.1.280) on xiaomi/mimo-v2.6-flash, the agent's default effort;
BenchFlow 0.9.1 on Docker; 3 SkillsBench tasks at commit f860ecf0, 3 tries each (9 trials), retries at the default (2).

6 of 9 trials passed: 67%. 95% interval over the 3 tasks, resampled: 0% to 100%;
35% to 88% if the 9 trials were independent (Wilson), which tries of one task aren't.
Per task: econ-detrending-correlation 3/3, offer-letter-generator 3/3,
crystallographic-wyckoff-position-analysis 0/3 (each try stopped at its 15-minute limit).
Controls on the same tasks: oracle 3/3, nop 0/3. No trial was unscored, errored or retried.
Tokens: 3,704,569 in all; MiMo has no price in BenchFlow, so no cost is given.
Not controlled: the provider's sampling defaults. Three tasks say nothing about SkillsBench as a whole.
```

The interval over tasks comes from the snippet in [Compare runs](./compare-runs.md#choose-the-metric). Name what you didn't control, such as the provider's own sampling defaults, and what you can't claim: three tasks say nothing about a whole benchmark.

## Share trajectories

`bench traj upload` sends a trial's trajectory to BenchFlow's public trajectory dataset on Hugging Face, after showing you the redacted steps and asking you to confirm:

```bash
bench traj upload jobs/<job>/<trial>
```

It also asks for your GitHub username and email (`--github-id`, `--email`), which go with the upload to credit you. It redacts secrets before upload; read the preview anyway.

## The report file (0.10)

> Needs BenchFlow `0.10.0rc2`, a release candidate on PyPI that you install by its version ([Install](../start/install.md#install-the-010-prerelease)).

0.10 writes `report.md` into every job, next to `run-card.json`. One from an OpenCode trial on the repair-café task, written by 0.10.0rc1:

```markdown
# Report: 2026-11-04__18-15-46

1 trials of 1 tasks, opencode on deepseek/deepseek-v4-flash: 1 passed (100%, 95% interval 21% to 100%, counting trials as independent), mean reward 1.00, 0 with an error.

| Task | Trials | Passed | Pass rate | 95% interval | Rewards | |
|---|---|---|---|---|---|---|
| repair-cafe | 1 | 1 | 100% | 21% to 100% | 1 | ██████████ |

Tokens 45,342; cost $0.00; trial time 29 to 29 s, 0 trial-minutes in all.

Sampling the model requests carried (the first 20 of each trial, with how many sent each value): max_tokens 32000 (6); no temperature or top_p sent, so the model server's default applies.
```

The last line is the part people usually leave out: what the requests actually carried, read from the recorded calls rather than from what was asked for. With one trial, the interval runs from 21% to 100%; that is what one try can tell you.

0.10.0rc2 counts the first line by the [scoring rule](../reference/scoring-rule.md): the interval is over tasks, not trials, the line after it names the rule, and a warning says when a task has fewer than 3 trials. A report over trials that ran several settings (agent and harness version, served model, route, effort sent, task set, BenchFlow version) gives one line per setting instead of one score, with a `mixed settings` warning. A `Seeds:` line says which seeds the numbers depend on, and a table puts each trial's model outcome beside its verifier reward. Report a result per setting, the way the report gives it.
