# Configs and variables

A run config is a YAML file that holds a whole run, so you can repeat it, review it and keep it in version control. Write one when a run has more settings than you want to retype; flags still override it, and each environment variable goes only where you send it.

## A run in a file

```yaml
# run.yaml
tasks_dir: tasks
agent: oracle
sandbox: docker
concurrency: 2
jobs_dir: jobs/config
```

```bash
bench run --config run.yaml
```

```text
Job complete: 1/1 (100.0%), mean_reward=1.00, errors=0, idle_timeouts=0, time=0.2min

✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
Outcomes: 1 scored (1 passed, 0 failed), 0 unscored, 0 errored
Cost:      none: no model tokens were used
Time:      10s
Artifacts: jobs/config/2026-11-04__18-11-09
```

A config needs `tasks_dir:` (a folder) or `source:` (a GitHub folder: `repo`, `path`, `ref`). The other keys mirror the flags: `agent`, `model`, `reasoning_effort`, `harness`, `include`, `exclude`, `n_tasks`, `concurrency`, `agent_env` and more; [Files BenchFlow reads and writes](../reference/files.md) lists them all.

- A flag on the command line wins over the file, and the file wins over the default: `bench run --config run.yaml -a nop` runs `nop`.
- Leave `agent` out and a config runs `claude-agent-acp`, which needs a model and a login, while `bench run` with no `-a` runs the oracle ([Files BenchFlow reads and writes](../reference/files.md#run-config) lists each key's default).
- A key the loader doesn't know stops the run before anything starts, and names the line and the nearest key.
- `--dry-run` prints the plan the config makes.
- A config runs each task once. `--trials 3` beside it is refused with `--trials > 1 needs a task folder or --dataset; a --config file or --source-env run repeats nothing`; to repeat, name the task folder on the command line with `--trials` instead ([Run a benchmark](../evaluate/run-a-benchmark.md#repeat-each-task)).
- A different format, the YAML job config (`agents:` with `datasets:` or a `tasks:` list), runs every agent on every dataset `n_attempts` times and never retries a trial; its keys are in [Files BenchFlow reads and writes](../reference/files.md#yaml-job-config).

## Environment variables

A sandbox gets only the variables you hand it on purpose: `--agent-env` (or `agent_env:` in a config) for the agent, the task's `sandbox: env:`, `verifier: env:` and `oracle: env:` for those steps, and `--reviewer-agent-env` for a rubric reviewer. [Environment variables](../reference/environment-variables.md#where-a-variable-reaches) has the table, and what a task can read from your shell. Don't pass a sandbox provider's key, such as `DAYTONA_API_KEY`, with `--agent-env`: that puts it inside the sandbox, where the agent can read it.
