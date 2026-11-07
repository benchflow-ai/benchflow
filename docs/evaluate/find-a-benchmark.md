# Find a benchmark

A benchmark is a set of tasks. BenchFlow runs benchmarks from three places: its registry, a folder in any Git repository, and a folder on your machine.

## Browse the registry

```bash
bench hub list
```

```text
                                                  BenchFlow Registry
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Dataset        ┃ Version ┃ Tasks ┃ Description                                                                      ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ skillsbench    │ 1.0     │    87 │ SkillsBench v1.0 — 87-task benchmark set (git tag v1.0). Superseded by 1.1 (sam… │
│ skillsbench    │ 1.1     │    87 │ SkillsBench v1.1 — 87-task benchmark set (same roster as 1.0) republished as na… │
│ terminal-bench │ 2.0     │    89 │ Terminal-Bench 2.0: 89 tasks in a terminal, pinned by BenchFlow to commit 69671… │
└────────────────┴─────────┴───────┴──────────────────────────────────────────────────────────────────────────────────┘
Showing 3 dataset(s). Refine with --search, raise --limit, or use --json for the full registry.
```

Each registry dataset is pinned to a version, so two people who run `terminal-bench@2.0` run the same tasks. `bench hub show terminal-bench@2.0` lists its tasks. Run one task of it, or all of them, with `-d`:

```bash
bench run -d terminal-bench@2.0 -i fix-git
bench run -d terminal-bench@2.0
```

```text
[fix-git] Oracle mode: running oracle solve.sh
[fix-git] Running verifier...
[fix-git] Rewards: {'reward': 1.0}
[PASS] fix-git (reward=1.00, tools=0)
...
✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
```

The version is required. `bench run -d skillsbench` stops with:

```text
Invalid dataset spec 'skillsbench' — expected <name>@<version> (e.g. skillsbench@1.1)
```

### SkillsBench's registry entries need BenchFlow 0.6

An entry names the BenchFlow versions it was checked with, and the two SkillsBench entries name `>=0.6.3,<0.7`. BenchFlow 0.9.2 refuses them:

```bash
bench run -d skillsbench@1.1 -i econ-detrending-correlation
```

```text
bench 0.9.2 is outside the range this dataset was validated against (>=0.6.3,<0.7) — results may not be comparable with published runs
  next: pick a dataset version validated for this version of BenchFlow, or re-run with --ignore-bench-version to proceed anyway
```

`--ignore-bench-version` doesn't help: the tasks at tag v1.1 use a `task.md` key 0.9.1 renamed, so none of them load (`the 'environment' key was renamed to 'sandbox'`). Run SkillsBench from its repository instead, at a commit, as below; the [Quick start](../start/quick-start.md)'s outputs came from commit f860ecf0.

### Terminal-Bench 2.0 on 0.9.1

Terminal-Bench 2.0 runs on 0.9.1, but 12 of its 89 tasks fail a control there, for causes in the task, the servers it downloads from, or BenchFlow; no agent can pass four of them. [Check a benchmark](../audit/check-a-benchmark.md#terminal-bench-20-on-091) lists the twelve with their causes and shows how to run the rest. Its tasks were also written for an agent that runs as root, and BenchFlow runs agents as a user of their own, so five tasks' tests read files only root could write ([the pitfall](../audit/pitfalls.md#terminal-bench-20-tasks-fail-that-pass-in-published-runs)). On Docker, a full run's task images take over 100 GB of disk.

## Run a benchmark from a Git repository

Any folder of tasks in a GitHub repository runs without the registry. Name it as `org/repo/path`, and pin it with `@ref` so the run can be repeated: a branch, a tag or a full commit hash (a short hash isn't found):

```bash
c=f860ecf02ef53412a030f8661c83d644b739ff01
bench run benchflow-ai/skillsbench/tasks/econ-detrending-correlation@$c   # one task
bench run benchflow-ai/skillsbench/tasks@$c -n 5                          # its first five tasks by name
```

BenchFlow downloads only that folder, into `.cache/datasets-v2/<org>/<repo>/` under the top of the Git repository you run it from (or under the current folder, outside a repository), and a rerun reuses it.

## Your own folder of tasks

A folder with one task in it, or a folder of task folders, runs the same way:

```bash
bench run tasks/half-life
bench run tasks/ -n 3
```

[Write a task](../tasks/write-a-task.md) builds one. Tasks in another layout, such as Terminal-Bench's `task.toml` folders, are on [Bring what you have](./bring-what-you-have.md), with what to check before you trust a benchmark you didn't write.
