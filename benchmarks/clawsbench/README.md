# ClawsBench — Environment-plane manifest

ClawsBench is BenchFlow's internal-dogfood **stateful multi-service benchmark**:
mock Gmail / Slack / Calendar / Docs / Drive services (the `claw-*` servers,
FastAPI + SQLite), sourced from the `smolclaws` repo.

`environment.toml` is the **manifest** — the entire framework-integration
surface. BenchFlow's `ManifestEnvironment` adapter reads it; the benchmark
image is otherwise untouched. The manifest is the declarative replacement
for the hard-coded `SERVICES` dict in `benchflow/sandbox/services.py`.

## How it maps to smolclaws

| Manifest field | Why |
|---|---|
| `base_image` | smolclaws ships one shared base; tasks are per-task images built `FROM` it |
| `owns_lifecycle = false` | the smolclaws images have no service-starting entrypoint — the framework starts the `claw-*` servers |
| `task_selection.mechanism = "image"` | each task's seed data is baked into its own image at build time |
| `[[services]]` catalog | the benchmark-wide service set; `ManifestEnvironment` starts only the binaries present in a given per-task image |

## Run it

```bash
bench eval run --tasks-dir benchmarks/clawsbench/tasks/<task> \
  --environment-manifest benchmarks/clawsbench/environment.toml \
  --agent claude-agent-acp --model claude-haiku-4-5
```

## Not yet implemented

`snapshot` / `restore` (environment-state branching) are the platform layer —
`ManifestEnvironment` raises `NotImplementedError` for them.
