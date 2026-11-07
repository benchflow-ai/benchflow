# BenchFlow

BenchFlow is a command-line tool, `bench`, and a Python package, `benchflow` (these pages cover 0.9.2). It runs AI agents on tasks in isolated sandboxes, scores every attempt with the task's own checks, and keeps everything the agent did.

## What do you want to do?

- Run my first eval: [Quick start](./start/quick-start.md)
- See what an agent eval needs, and how BenchFlow's parts fit: [Why BenchFlow](./start/why-benchflow.md), [How BenchFlow works](./start/how-it-works.md)
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
uv tool install --python 3.12 'benchflow==0.9.2' \
  --constraints https://raw.githubusercontent.com/benchflow-ai/benchflow/v0.9.2/constraints.txt
```

The constraints file pins every dependency to the version the release was tested with ([Install](./start/install.md)).

Score a public task with its reference solution, the `oracle` agent, which needs no model and no key:

```bash
bench run benchflow-ai/skillsbench/tasks/econ-detrending-correlation -a oracle
```

It prints `Score: 1/1` when the solution passes the task's tests. The [Quick start](./start/quick-start.md) then runs Claude Code on the same task and shows how to read what it did.

## Not in these guides

Two features of 0.9.2 have only their reference here: branching a trial from a saved sandbox snapshot ([`bench branch`](./reference/commands.md#bench-branch) and `bench run --checkpoints`), and hosted environments ([`bench hub`](./reference/commands.md#bench-hub) and `bench run --source-env`).

## Versions

These pages document BenchFlow 0.9.2, the current release. 0.9.2 is 0.9.1 with five grading-integrity fixes and nothing else ([Changelog](./versions/changelog.md)). Outputs from runs on 0.9.1 say so, and hold for 0.9.2 unless a task's grading meets one of those fixes. Commands that download BenchFlow's example files name the repository's `main` branch in this Markdown. On the published site they fetch from `docs-v0.9.2` instead, a tag of the 0.9.2 release with these pages and their example files added, so a page always runs against the files it was written with. The repository's Markdown and preview builds keep `main`. A page that needs 0.10.0rc2 says so in its first line, and the sidebar tags it `0.10`; it's on PyPI as a pre-release ([Install](./start/install.md#install-the-010-prerelease)). A section that needs it ends its heading with "(0.10)". [What's in 0.10.0rc2](./versions/0.10.md) lists them all.

## For coding agents

Every page is also Markdown: add `.md` to its URL, or send `Accept: text/markdown`. [`/docs/llms.txt`](https://www.benchflow.ai/docs/llms.txt) lists every page with its sections, and links one file per section with that section's pages in full.
