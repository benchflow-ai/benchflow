# Task formats

A **task format** lets a benchmark keep its tasks in a compact source form and still run them as first-class BenchFlow tasks, with no export step. BenchFlow detects a folder in a registered format when it loads tasks, calls the format to write an ordinary native package (`task.md`, `environment/`, `verifier/`, `oracle/`), and runs that package like any other: same sandboxes, agents, verifier, trial layout and job tooling.

Use it when many tasks share one generated environment. For example, a [robouse](https://github.com/benchflow-ai/robouse) task is a `task.md` with a `robouse:` block. Every task gets the same agent image (a `robo` client only), the same trusted simulator service and the same verifier, all built from the installed `robouse` package. Writing those files into each of hundreds of task folders would duplicate them, and they would drift from the package.

## Where formats apply

Every path that loads a task passes the folder through `benchflow.task.formats.materialize_task_dir` first:

| Entry point | Behaviour |
|---|---|
| `bench eval run --tasks-dir DIR` (and `tasks_dir:` in a YAML config) | `DIR` itself, or each child folder, is materialized if a format claims it; native children are used as they are |
| `bench tasks check DIR` | checks the materialized package and prints its path |
| `RolloutConfig(task_path=...)`, `bf.run(...)`, `SDK().run(...)`, `Evaluation(...)` | `task_path` / `tasks_dir` is materialized the same way |

Trial names, `result.json` `task_name` and `--include` / `--exclude` use the name of the materialized folder. Formats should name it after the task id, so the name matches the source folder.

## Registering a format

A format is any object with a `name`, a cheap `detect(task_dir) -> bool` that never raises, and `materialize(task_dir, out_root) -> Path`. Ship it as an entry point in the package that owns the tasks:

```toml
[project.entry-points."benchflow.task_formats"]
robouse = "robouse.engine.format:RobouseTaskFormat"
```

The entry point may name an instance or a class that takes no arguments. Code running in the same process can also call `benchflow.task.formats.register_task_format(obj)`. A format whose entry point fails to import is logged and skipped, so native tasks still load.

`materialize` writes the native package under `out_root`, which is `$BENCHFLOW_TASK_FORMAT_CACHE/<name>` (default `~/.cache/benchflow/task-formats/<name>`), and returns its folder. It must be deterministic and safe to call concurrently. Content-address the output (hash the source task and the generator), write it to a temporary folder and rename it into place. BenchFlow refuses a returned folder that some format still claims, so a format cannot hand back its own input.

## Seeded variants

A format may also implement `materialize_variant(task_dir, out_root, *, seed)`, which writes the package of one seed. `bench eval run --seeds 0-4` (or `0,3,7`) then loads every claimed task once per seed, and `summary.json` gets a `seeded` report (pass@k, mean, standard deviation and reset reproducibility per task). The seeded package's folder name must differ per seed; `benchflow.embodied.sidecar.EmbodiedTaskFormat` names it `<task>--seed-<n>`. `--seeds` refuses native task packages, since BenchFlow cannot pass them a seed.

## Simulator tasks

Formats whose tasks run a simulator next to the agent can subclass `benchflow.embodied.sidecar.EmbodiedTaskFormat`, which writes the agent image, the compose topology with a trusted simulator service, the physical verifier and the seeded and noop variants, and leaves only the simulator image, the episode factory and the reference solutions to the benchmark. See [embodied.md](./embodied.md).

## What a format may generate

Anything a native package may contain. The robouse format uses:

- `environment/docker-compose.yaml` with a `main` service (the agent container) and a trusted sidecar service. The build context of the sidecar may lie outside the package (a shared runtime folder under `out_root`). Keep it an absolute path in the cache: BenchFlow sometimes stages a copy of the task elsewhere, for example when it injects skills.
- `verifier.service: <sidecar>` in `task.md`, so the verifier runs next to the trusted state and its `/logs/verifier` is downloaded from there (Docker sandbox only).
- `oracle/solve.sh`, which BenchFlow uploads to `/oracle` and hides from agents.

Unknown top-level `task.md` keys are rejected by the native schema, so the materialized `task.md` must move any format-specific block elsewhere (robouse moves `robouse:` to `metadata.robouse`).
