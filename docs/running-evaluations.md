# Running evaluations

`bench eval run` is the main command for both one task and a batch. It accepts
a local directory, a remote Git repository, a YAML run config, or a pinned
dataset version. All of these use Docker by default.

Start with [Getting started](./getting-started.md) if you have not completed a
local run yet.

## One task from a remote repository

```bash
bench eval run \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks/citation-check \
  --agent codex \
  --model gpt-5.5 \
  --sandbox docker
```

BenchFlow clones and caches the source under `.cache/datasets/`. Pin a branch,
tag, or commit with `--source-ref` when reproducibility matters.

## One local task

```bash
bench eval run \
  --tasks-dir tasks/my-task \
  --agent codex \
  --model gpt-5.5
```

The omitted `--sandbox` defaults to `docker`. A task directory contains a
native `task.md` plus its `environment/` and `verifier/` directories. BenchFlow
can still read the retired split layout for compatibility.

## A local batch

Point `--tasks-dir` at the parent directory and choose a conservative local
concurrency:

```bash
bench eval run \
  --tasks-dir tasks \
  --agent codex \
  --model gpt-5.5 \
  --sandbox docker \
  --concurrency 2
```

Use repeatable filters to select task names:

```bash
bench eval run \
  --tasks-dir tasks \
  --include citation-check \
  --include weighted-gdp-calc \
  --agent codex \
  --model gpt-5.5
```

Local concurrency is limited by your Docker daemon, CPU, memory, and model
rate limits. Increase it gradually. A cloud sandbox becomes useful when you
need more isolation or more parallel machines, not because BenchFlow requires
one; see [Sandboxes](./sandboxes.md).

## YAML run configs

Use a config when the same run should be repeated or reviewed:

```yaml
source:
  repo: benchflow-ai/skillsbench
  path: tasks
agent: codex
model: gpt-5.5
environment: docker
concurrency: 2
include:
  - citation-check
```

```bash
bench eval run --config run.yaml
```

Check the [CLI reference](./reference/cli.md#bench-eval-run) for the full run
schema and flags.

## Compare a task with and without skills

For a task that already contains its skill payload, run the two modes into
separate job directories:

```bash
bench eval run \
  --tasks-dir tasks/my-task \
  --agent codex --model gpt-5.5 \
  --skill-mode no-skill \
  --jobs-dir jobs/my-task-no-skill

bench eval run \
  --tasks-dir tasks/my-task \
  --agent codex --model gpt-5.5 \
  --skill-mode with-skill \
  --jobs-dir jobs/my-task-with-skill
```

Use `--skills-dir <directory>` when the skills live outside the task package.
For a structured lift experiment backed by `evals/evals.json`, use
`bench skills eval`; see [Skill evals](./skill-eval.md).

## Results and exit status

By default, artifacts land in `jobs/<timestamp>/`. Summarize them with:

```bash
bench eval list jobs/
bench eval metrics jobs/
```

Exit code 0 means the evaluation pipeline completed. It does not mean every
task passed. Read each rollout's reward or the printed `[PASS]` / `[FAIL]`
status to assess model performance. Configuration, agent, or verifier errors
produce a non-zero exit.

Use a new `--jobs-dir` for an independent rerun. Reusing one intentionally
resumes it and skips completed rollouts.

## Reproducible published runs

For leaderboard, paper, or release evidence, prefer a pinned registry dataset:

```bash
bench eval run \
  --dataset skillsbench@1.1 \
  --agent codex \
  --model gpt-5.5
```

Dataset runs verify the pinned commit and per-task content digests. Ad-hoc
`--tasks-dir` and floating repository runs are better suited to development.
