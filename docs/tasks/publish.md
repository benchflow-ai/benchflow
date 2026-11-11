# Publish a benchmark

A benchmark is a folder of tasks other people can run and get the same tasks. Share it through Git, pin versions so results can be compared, and, for a widely used set, add it to the registry. Before you publish, run the checks in [Check a benchmark](../audit/check-a-benchmark.md): every task should pass its oracle and fail `nop`.

Runs here use MiMo (`-m xiaomi/mimo-v2.6-flash`): export its key and base URL first ([Quick start](../start/quick-start.md#on-another-provider)), or use `-m sonnet` with a Claude login.

## Share through Git

Push the folder of tasks to a Git repository. Anyone can run it by `org/repo/path`, with `@ref` for a tag or commit:

```bash
bench run your-org/your-benchmark/tasks@v1.0 -n 5
bench run your-org/your-benchmark/tasks/half-life@v1.0 \
  -a claude -m xiaomi/mimo-v2.6-flash
```

BenchFlow downloads only the folder named. To repeat each task with `--trials`, people clone the repository first: `--trials` needs a local folder or a registry dataset, and refuses a Git address with `--trials > 1 needs a task folder or --dataset`.

## Version the tasks

Every task has a content digest: one hash over its files. Two runs with the same digest ran the same task.

```bash
bench tasks digest tasks/half-life
```

```text
sha256:039d8e9d3c9e722203651d37033dcf839f1f63318e08fcfd5303e666a48300b2
```

Each trial records the digest of the task it ran in `config.json` and `result.json` (`task_digest`). When you fix a task, its digest changes; give the release a new tag, and say which tasks changed, so nobody compares scores across the two versions by accident.

## Add it to the registry

The registry is what `bench run -d name@version` and `bench hub list` read. An entry pins each task to a Git commit and a digest. SkillsBench 1.1's entry, with its description and notes left out and one of its 87 tasks:

```json
{
  "name": "skillsbench",
  "version": "1.1",
  "git_tag": "v1.1",
  "bench_version": ">=0.6.3,<0.7",
  "tasks": [
    {
      "name": "econ-detrending-correlation",
      "git_url": "https://github.com/benchflow-ai/skillsbench.git",
      "git_commit_id": "b63b7b2850226b6aa4fb5929a8c1ac7bc4d9a6af",
      "path": "tasks/econ-detrending-correlation",
      "digest": "sha256:9b238639d4d2c03792ec2bb0ccce3f43b63a91a030a614dadfc755be8ca747b9"
    }
  ]
}
```

`bench_version` names the BenchFlow versions the entry was validated with; a run outside that range is refused unless it passes `--ignore-bench-version`, as [Find a benchmark](../evaluate/find-a-benchmark.md#skillsbenchs-registry-entries-need-benchflow-06) shows for this entry on 0.9.2. `bench hub check` checks a sample of a dataset's tasks. The default registry is [`registry.json`](https://github.com/benchflow-ai/skillsbench/blob/main/registry.json) in the SkillsBench repository; to add a benchmark to it, open a pull request there with your entry. To run datasets from your own registry file instead, pass its path or URL with `--registry`.

## Publish in the task.toml layout too

Some runners read only the `task.toml` layout. `bench export` writes each task in it, with a report of anything the layout can't hold, and `bench import` converts the other way. [Bring what you have](../evaluate/bring-what-you-have.md#take-your-tasks-elsewhere) shows both, with their output.

## Declare the setup runs should match (0.10)

> Needs BenchFlow `0.10.0rc3`, a release candidate on PyPI that you install by its version ([Install](../start/install.md#install-the-010-prerelease)).

If your benchmark's published numbers used a particular setup, write it in `run-reference.toml` next to the tasks. Every run's [run card](../evaluate/before-you-run.md#the-run-card-010) compares itself with it:

```toml
# The setup this task set's published numbers used.
[reference]
source = "the half-life task's example reference"
agent = "claude-agent-acp"
trials = 5
agent_timeout_sec = 900
strict = true
```

With `strict = true`, a run that differs is refused:

```bash
bench run tasks-ref -a claude -m xiaomi/mimo-v2.6-flash --trials 3 --dry-run
```

```text
  Reference: the half-life task's example reference: 1 setting differs, 0 to check yourself (strict)
The run differs from its reference setup:
  differs: trials 3 (reference 5)
This run differs from the setup in .../tasks-ref/run-reference.toml (the half-life task's example reference), which asks every run to match it or say why: trials 3 (reference 5)
  next: match the reference, or pass --accept-deviations '<why>' (accept_deviations: in a run config); the note is kept in run-card.json
```

`--accept-deviations "three trials for the docs example"` lets the run go ahead, and the card records the reason:

```text
  differs: trials 3 (reference 5)
  accepted: three trials for the docs example
```

Without `strict`, the differences are printed as a warning and the run goes on.
