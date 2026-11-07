# Bring what you have

BenchFlow runs the tasks, agents and models you already have, and writes its results in formats other tools read. Find what you have in the table, then follow its link.

| You have | What BenchFlow does with it | See |
|---|---|---|
| Tasks in the `task.toml` layout, Terminal-Bench's | runs them as they are, or converts them with `bench import` | [below](#tasks-in-the-tasktoml-format) |
| Terminal-Bench 4.1 | can't score it yet on these releases | [below](#terminal-bench-41) |
| A benchmark someone else wrote | check it with controls before you trust its numbers | [Check a benchmark](../audit/check-a-benchmark.md) |
| Your own agent | runs it over ACP, or as a script | [Your own agent](./your-own-agent.md) |
| A model on your own server | calls any OpenAI-compatible endpoint | [A model you serve](./agents.md#a-model-you-serve) |
| Jobs from BenchFlow 0.7 or 0.8 | reads them as they are | [below](#old-results) |
| A tool that reads ATIF, or a trainer | writes ATIF and training rows for every trial | [below](#trajectories-and-training-rows) |
| An agent's OpenTelemetry | doesn't need it: model calls are recorded at BenchFlow's proxy | [below](#model-calls-and-opentelemetry) |

## Tasks in the task.toml format

A `task.toml` folder holds `task.toml` (the settings), `instruction.md`, `environment/`, `tests/` and `solution/`. BenchFlow runs it as it is, renaming the settings it calls differently on the way in. Terminal-Bench 2.0's `fix-git` task, from the registry, scored by its reference solution:

```bash
bench run -d terminal-bench@2.0 -i fix-git
```

```text
[fix-git] Running verifier...
[fix-git] Rewards: {'reward': 1.0}
[PASS] fix-git (reward=1.00, tools=0)
Job complete: 1/1 (100.0%), mean_reward=1.00, errors=0, idle_timeouts=0, time=0.2min

✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
```

A local folder of tasks in the `task.toml` format runs the same way: `bench run path/to/tasks`. To move a task to BenchFlow's own `task.md` format ([Task format](../tasks/task-format.md)), convert it in place:

```bash
bench import half-life
```

```text
Created: half-life/task.md
  kept task.toml and instruction.md
```

`--remove-legacy` deletes the split files once the new `task.md` checks out. Converting is optional: both layouts run.

Ten published Terminal-Bench 2.0 tasks, `hf-model-inference` and `pypi-server` among them, keep services the agent leaves running (`x-keep-services`). A `--sandbox daytona` sandbox can't hold those for an agent that runs as a non-root user, so BenchFlow refuses such a trial before it starts, with `KeptServiceFreezerUnavailable` ([Error messages](../reference/errors.md)). Run those tasks with `--sandbox docker` or `--sandbox remote-docker`.

> **Known issue in 0.9.1:** before grading, it removes every `conftest.py` in the workspace, the repository's own included, so on repository-based benchmarks such as SWE-bench a correct answer whose tests need its fixtures scores 0. 0.10.0rc1 keeps a repository's unchanged `conftest.py`. Run the oracle over such a benchmark before you trust its scores ([Known pitfalls](../audit/pitfalls.md#scores-that-count-the-wrong-thing)).

## Terminal-Bench 4.1

Terminal-Bench 4.1 ([tbench.ai](https://www.tbench.ai), released 2026-11-03, 66 tasks) is the current Terminal-Bench. It isn't in the registry: its tasks are in the Git repository tbench.ai links to, under `tasks/` at the tag `v4.1.0`, and run by address like any [folder of tasks in a repository](./find-a-benchmark.md#run-a-benchmark-from-a-git-repository), as `<org>/<repo>/tasks/interleaved-vigenere@v4.1.0`. They don't score yet: none of the four we ran with their reference solutions on 0.9.1 got a score, nor the one we ran on 0.10.0rc1. Three failed where the tests' separate sandbox receives the files the task lists as artifacts; the fourth uses a newer `task.toml` schema (2.0) that 0.9.1 refuses. On `interleaved-vigenere`:

```text
[ERR] interleaved-vigenere (tools=0) (separate verifier transfer failed: artifact…)
...
Outcomes: 0 scored (0 passed, 0 failed), 1 unscored, 0 errored
  1 unscored: the verifier's sandbox failed (infrastructure problem): interleaved-vigenere
```

and `result.json` records `separate verifier transfer failed: artifact collection of /app/cracker.py error: Workspace metadata could not be captured completely`. On `session-window-debug` the task is refused with `Unsupported schema_version major 2 (from '2.0'); supported major version(s): 1`. The run then suggests `bench eval resume`, which fails the same way, and `bench results` lists the trial as a verifier error rather than unscored; neither is a problem with your task or your machine. Don't report Terminal-Bench 4.1 scores from these releases. The registry's `terminal-bench@2.0` runs on 0.9.1: its `fix-git` task, above, scores 1 with its reference solution, and [Check a benchmark](../audit/check-a-benchmark.md#terminal-bench-20-on-091) has the oracle and nop on all 89 of its tasks, with the 12 that fail a control. Name the version next to any Terminal-Bench score you report.

## Old results

`bench results` and `bench eval metrics` read jobs from BenchFlow 0.7 and later as they are. A job made by 0.7.11, read by 0.9.1:

```bash
bench eval metrics jobs/old07
```

```text
│ Total          │ 1                                │
│ Passed         │ 1                                │
...
│ Score          │ 100.0% (1 of 1 trials)           │
│ Mean reward    │ 1.000 (over 1 trials)            │
```

Don't run into an old job's folder unless you mean to: a plain rerun resumes the old job and rewrites its summary ([Upgrade from 0.7](../versions/upgrade-from-0.7.md#old-jobs-and-resume)). Results written by other runners aren't read correctly yet; in our test on 0.10.0rc1, a passing trial from another `task.toml` runner showed as unscored. To compare with such results, run the same tasks here.

## Trajectories and training rows

Every trial writes `trainer/atif.json`, its trajectory in ATIF (the Agent Trajectory Interchange Format), version `ATIF-v1.7`, for any tool that reads ATIF. BenchFlow writes ATIF but doesn't read it: `bench traj` uploads and checks trajectories, and has no import.

For a trainer, `bench train convert` writes rows in the `prime-sft`, `trl-sft` or `branch-tree` layout, and every JSON file BenchFlow writes for other programs has a versioned schema. [Trajectory and rollout formats](../train/formats.md) lists the files and schemas; [Export training data](../train/export.md) shows a row.

## Model calls and OpenTelemetry

BenchFlow records each model call where it passes through BenchFlow's model proxy, with its tokens, timing and the settings it carried, so it doesn't depend on the agent's own telemetry. That matters because agents don't all export OpenTelemetry, and Claude Code often doesn't. BenchFlow 0.9.1 and 0.10.0rc1 neither emit nor read OpenTelemetry. The gap is a subscription login, whose calls skip the proxy: [Usage and telemetry](../scale/telemetry.md) says what is recorded then.

## Take your tasks elsewhere

`bench export` writes a task in the `task.toml` layout, with a report of anything that layout can't hold:

```bash
bench export tasks/half-life half-life-toml
```

```text
Exported: half-life-toml
  target: task-toml
  status: lossless
  losses: 0
  report: compatibility/export-report.json
```

The exported task runs too: the oracle scored it 1.0. Its `task.toml` spells out every default setting, so it's longer than the `task.md` it came from. `--report-only` prints the report without writing files.
