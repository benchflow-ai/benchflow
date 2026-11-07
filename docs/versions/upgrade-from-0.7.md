# Upgrade from 0.7

0.9 renamed the main commands, changed what a run does when you don't name an agent, and reads 0.7 job folders. Your 0.7 commands still run. This page moves an install and a project from 0.7.x to 0.9.1, and back if you need to.

## Before you upgrade

Write down what you have, so you can go back to exactly it:

```bash
bench --version
uv tool list --show-paths | grep benchflow
```

Note the extras you installed (`benchflow[sandbox-daytona]` and so on); the upgrade has to name them again. Finished jobs stay readable; a job that was still running under 0.7 is better finished with 0.7.

## Upgrade the install

Install 0.9.1 over 0.7.x with the same command shape as a fresh install; uv replaces the old version:

```bash
uv tool install --python 3.12 'benchflow==0.9.1' \
  --constraints https://raw.githubusercontent.com/benchflow-ai/benchflow/v0.9.1/constraints.txt
bench --version
```

```text
benchflow 0.9.1
```

Add your extras inside the quotes: `'benchflow[sandbox-daytona]==0.9.1'`.

## Old jobs and resume

A plain rerun resumes the latest job in its jobs folder, including one made by 0.7: it reuses the old trials, runs nothing new and prints the old score. A job made with 0.7.11, then the same task and folder with 0.9.1:

```bash
bench eval run --tasks-dir one --agent oracle --jobs-dir jobs/upgrade   # 0.7.11
bench run one -a oracle --jobs-dir jobs/upgrade                         # 0.9.1
```

```text
Resuming into existing job directory: 2026-11-04__18-57-31 (finished tasks are reused; pass a new job_name, or --fresh on the CLI, for a new run)
Job 2026-11-04__18-57-31 was made by BenchFlow 0.7 or older (it has no evaluation.json): this run reuses its finished trials and rewrites its summary.json and results files. Its own files are kept in jobs/upgrade/2026-11-04__18-57-31/.pre-0.8. Pass --fresh (or another --job) to leave it as it was.
Job: 1 task, 1 done, 0 to run (concurrency=4, warm pool: up to 1 prepared sandbox)
...
Resumed job jobs/upgrade/2026-11-04__18-57-31: all 1 task(s) were already finished, so nothing ran and these are the earlier results. Pass --fresh (or a new --job-name) for a new run.
```

So pass `--fresh` the first time you run after the upgrade, or point `--jobs-dir` at a new folder. With `--fresh`, the same command starts a new job and runs the task again: `Job: 1 task, 0 done, 1 to run`.

The rewrite is what an old job gains: it gets the `evaluation.json` that 0.9.1 keeps for every job, so `bench eval resume` and `bench results` treat it like a new one. To only read an old job's rewards, `bench results <job folder>` prints them and runs nothing.

`bench view` serves a job in the browser at the `localhost` address it prints. Open it as printed: the same page at `127.0.0.1` is refused.

## Scores count every trial

The headline numbers count differently since 0.8, and 0.9.1 keeps it: `mean_reward` and the pass rate are over every trial, with an unscored or errored trial as 0, where 0.7 averaged over scored trials only. 0.7's figure is still there, as `mean_reward_scored_only` in `summary.json`, and when 0.9.1 rewrites an old job's summary and the two differ, it says so (this is the message as the code has it; `…` marks the numbers):

```text
This job's BenchFlow 0.7 summary said mean_reward=…, the mean over scored trials only. BenchFlow 0.8's mean_reward counts every trial, an unscored one as 0: …; its mean_reward_scored_only is 0.7's figure: ….
```

Compare a 0.7 score with `mean_reward_scored_only`, or rerun both sides on 0.9.1; [Compare runs](../audit/compare-runs.md#what-counts-in-the-denominator) says which number to report.

## Changed commands

The 0.7 spellings still work and print the new one:

```bash
bench eval list jobs/opencode
```

```text
note: `bench eval list` is now `bench results`, with the same arguments
...
```

The new names, side by side:

| 0.7 | 0.9.1 |
|---|---|
| `bench eval run --tasks-dir T --agent A --model M` | `bench run T -a A -m M` |
| `bench tasks init NAME` | `bench new NAME` |
| `bench tasks check T` | `bench check T` |
| `bench tasks migrate T` | `bench import T` |
| `bench eval list`, `bench eval inspect JOB`, `bench eval compare A B` | `bench results`, `bench results JOB`, `bench results A B`; `bench eval compare A B` still runs, and only it takes `--vary` and `--on-mismatch` |
| `bench eval view` | `bench view` |
| `bench eval regrade` | `bench regrade` |
| `bench agent list`, `bench agent show NAME` | `bench agents`, `bench agents NAME` |
| `bench tasks export` | `bench export` |

One behaviour changed with the name: `bench eval run` without `--agent` runs Claude Code, while `bench run` without `-a` runs the oracle. Name the agent in scripts you move to `bench run`.

`bench eval branch` is the same command as `bench branch`. The other `bench eval` commands have no new name and stay current: `metrics` (pass@k and the rest), `resume`, `score`, `compare` (for `--vary`), `smoke`, `adopt`, `compare-lift`, `continue` and `branches`.

## Changed tasks

`task.md` calls the sandbox section `sandbox:`. A task 0.7.11 runs already says so, since 0.7.11 refuses `environment:` too; older files, such as SkillsBench 1.1's 0.6-era tasks, still say `environment:` and are refused with:

```text
the 'environment' key was renamed to 'sandbox' — rename 'environment:' to 'sandbox:' (legacy task.toml imports convert '[environment]' automatically)
```

Rename the key. `task.toml` tasks keep their `[environment]` table: BenchFlow converts it on the way in. Run `bench check` over your tasks after the upgrade; it lists every such problem without starting a sandbox.

## Go back

Run the install line with the old version; uv replaces 0.9.1. 0.7.11 has no constraints file, so it installs with today's versions of its dependencies:

```bash
uv tool install --python 3.12 'benchflow==0.7.11'
bench --version
```

```text
benchflow 0.7.11
```
