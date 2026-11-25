# BenchFlow

BenchFlow is a command-line tool, `bench`, and a Python package, `benchflow` (these pages cover 0.10.3), for evaluating AI agents. It runs an agent on each task in a fresh sandbox, scores what the agent left with the task's own checks, and keeps a record of each attempt that can explain its score: what the agent did, each model call through BenchFlow's proxy with the settings it carried, and the workspace it left. SkillsBench and Robo Use run on it.

## What do you want to do?

- Run my first eval: [Quick start](./start/quick-start.md)
- See what that record lets you decide, where BenchFlow stops, and how its parts fit: [Why BenchFlow](./start/why-benchflow.md), [How BenchFlow works](./start/how-it-works.md)
- Measure an agent or a model on a benchmark: [Find a benchmark](./evaluate/find-a-benchmark.md), then [Run a benchmark](./evaluate/run-a-benchmark.md)
- Check a benchmark, or a published score: [Check a benchmark](./audit/check-a-benchmark.md), [Reproduce a published score](./audit/reproduce.md)
- Publish a score others can check: [Report a result](./audit/report.md)
- Build tasks, or a whole benchmark: [Write a task](./tasks/write-a-task.md)
- Grade design work by blind pairwise votes, from people or a model (0.10): [Human preferences](./tasks/human-preferences.md)
- Score an agent I wrote: [Run your own agent](./evaluate/your-own-agent.md)
- Run simulated robot tasks from several seeds: [Embodied tasks](./evaluate/embodied.md)
- Train a model on scored attempts: [BenchFlow as an RL environment](./train/environment.md)
- Run thousands of trials in the cloud: [Sandboxes](./scale/sandboxes.md)
- Bring tasks, results or trajectories from another tool: [Bring what you have](./evaluate/bring-what-you-have.md)
- Move a project from 0.7: [Upgrade from 0.7](./versions/upgrade-from-0.7.md)

## Try it

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and Docker running. Install BenchFlow:

```bash
uv tool install --python 3.12 'benchflow==0.10.3' \
  --constraints https://raw.githubusercontent.com/benchflow-ai/benchflow/v0.10.3/constraints.txt
```

The constraints file pins every dependency to the version the release was tested with ([Install](./start/install.md)).

Score a public task with its reference solution, the `oracle` agent, which needs no model and no key:

```bash
bench run benchflow-ai/skillsbench/tasks/econ-detrending-correlation -a oracle
```

It ends with the score, `1/1 passed`, when the solution passes the task's tests. The [Quick start](./start/quick-start.md) then runs Claude Code on the same task and shows how to read what it did.

## Not in these guides

Hosted environments have only their reference here ([`bench hub`](./reference/commands.md#bench-hub) and `bench run --source-env`). Branching a trial from a saved sandbox snapshot has a guide, [Branch a rollout at a checkpoint](./train/environment.md#branch-a-rollout-at-a-checkpoint), and its reference is [`bench branch`](./reference/commands.md#bench-branch).

## Versions

These pages document BenchFlow 0.10.3, the current release; 0.10.1 is the one before it (0.10.2 was tagged but never published), and 0.9.3 the last 0.9 release ([Changelog](./versions/changelog.md)). Outputs from runs on an earlier version say which, and [What's in 0.10.3](./versions/0.10.md#changes-that-can-move-a-score-or-stop-a-task) lists the changes that can move a score. Commands that download BenchFlow's example files name the repository's `main` branch in this Markdown. On the published site they fetch from `v0.10.3` instead, the 0.10.3 release tag, which holds these pages and their example files, so a page always runs against the files it was written with. The repository's Markdown and preview builds keep `main`. A page new in 0.10 says so in its first line, and the sidebar tags it `0.10`; a section new in 0.10 ends its heading with "(0.10)". [What's in 0.10.3](./versions/0.10.md) lists them all.

## For coding agents

Every page is also Markdown: add `.md` to its URL, or send `Accept: text/markdown`. [`/docs/llms.txt`](https://www.benchflow.ai/docs/llms.txt) lists every page with its sections, and links one file per section with that section's pages in full.
