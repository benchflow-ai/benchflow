# The Environment plane

The **Environment plane** is the stateful world the agent acts in — Han's
"S" in `E = {T, H, V, S, C}`. It is one of BenchFlow's four swappable planes
(Sandbox, Agent, Environment, Reward). See [`architecture.md`](./architecture.md),
"The Environment plane & the manifest".

A benchmark author never subclasses the framework. They write one file — an
**`environment.toml` manifest** — and the default adapter (`ManifestEnvironment`)
runs it on any Sandbox provider. The manifest is the entire integration
surface.

## The manifest schema

The manifest's keys live under an `[environment]` table.

### `[environment]`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | — (required) | Environment / benchmark name. |
| `image` | str | `None` | A ready-to-run image. Set this **or** `base_image`. |
| `base_image` | str | `None` | Image that per-task images build `FROM` (smolclaws-style). |
| `ports` | list[int] | `[]` | Ports the environment exposes (in addition to service ports). |
| `owns_lifecycle` | bool | `true` | `true` — the image entrypoint starts the services. `false` — the framework starts the `[[services]]`. |
| `keep_alive` | bool | `true` | Keep the environment up for the whole rollout. |
| `isolation` | `"per_task"` \| `"persistent"` | `"per_task"` | `per_task` — a fresh environment per episode. `persistent` — cross-episode state. |

Exactly one of `image` / `base_image` must be set. When `owns_lifecycle` is
`false` the manifest must declare `[[environment.services]]`; when it is
`true` it must not.

### `[environment.task_selection]`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `mechanism` | `"image"` \| `"env_var"` | `"env_var"` | `image` — the task's seed data is baked into a per-task image. `env_var` — one image, the task id passed at runtime. |
| `key` | str | `"BENCHFLOW_TASK_ID"` | Env var name (when `mechanism = "env_var"`). |
| `inject_into` | `"entrypoint"` \| `"exec"` | `"entrypoint"` | `entrypoint` reaches PID 1; `exec` does not. |

### `[[environment.services]]`

An array — one table per service the framework starts (only when
`owns_lifecycle = false`). It is the declarative replacement for the
hard-coded `SERVICES` dict in `benchflow/sandbox/services.py`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | — (required) | Service name. |
| `command` | str | — (required) | Full start command. |
| `port` | int | — (required) | Port the service listens on. |
| `health_path` | str | `"/health"` | HTTP path probed for readiness. |

### `[environment.readiness]`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `http` | list[str] | `[]` | Explicit HTTP probes. When empty, derived from the services. |
| `tcp` | list[int] | `[]` | TCP-connect probes. |
| `timeout_sec` | int | `120` | How long to wait for readiness before failing the rollout. |

### `[environment.forward_env]`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `keys` | list[str] | `[]` | Host env vars forwarded into the environment container. |

### `[environment.state]`

Present only for an environment that supports **roll-back** — `snapshot` /
`restore`. Absent this table, the environment is treated as stateless and
`snapshot`/`restore` raise `RuntimeError`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `kind` | `"sqlite"` | `"sqlite"` | State backend. Only SQLite is supported today. |
| `paths` | list[str] | `[]` | The database files to capture and restore (one snapshot covers all of them). |

## Worked example — ClawsBench

`benchmarks/clawsbench/environment.toml` — the internal-dogfood stateful
multi-service benchmark (mock Gmail / Slack / Calendar / Docs / Drive):

```toml
[environment]
name           = "clawsbench"
base_image     = "kywch/smolclaws-base:latest"
owns_lifecycle = false
isolation      = "per_task"

[environment.task_selection]
mechanism = "image"

[environment.readiness]
timeout_sec = 60

[environment.forward_env]
keys = ["ANTHROPIC_API_KEY"]

[[environment.services]]
name    = "gmail"
command = "claw-gmail --db /data/gmail.db serve --host 0.0.0.0 --port 9001 --no-mcp"
port    = 9001

# ... slack (9002), gcal (9003), gdoc (9004), gdrive (9005)
```

One manifest serves the whole benchmark even though smolclaws builds a
per-task image carrying only a subset of the services: `ManifestEnvironment`
probes each service's entry point with `--help` and starts only the services
whose package is actually installed in this per-task image.

## Worked example — chi-bench

`benchmarks/chi-bench/environment.toml` — the *other* topology, and the
external proof that a heavy environment onboards untouched. chi-bench is a
~25k-LOC healthcare simulator that ships **one** ready-to-run image whose
entrypoint starts its own services, so the manifest declares no
`[[services]]`:

```toml
[environment]
name           = "chi-bench"
image          = "chi-bench:latest"
owns_lifecycle = true
isolation      = "per_task"
ports          = [8020, 8023, 8100, 8200]

[environment.task_selection]
mechanism   = "env_var"
key         = "CHI_BENCH_TASK_ID"
inject_into = "entrypoint"

[environment.readiness]
http        = ["http://localhost:8023/health"]
timeout_sec = 120

[environment.forward_env]
keys = ["ANTHROPIC_API_KEY"]
```

This ~25-line manifest is the *entire* framework-integration surface:
chi-bench's image, Dockerfile, and entrypoint are unmodified, and the ~920
LOC of external-framework coupling it previously carried collapses into the manifest.
ClawsBench (`base_image` + framework-started `[[services]]`) and chi-bench
(`image` + `owns_lifecycle = true`) are the two topologies behind one
contract. See [`benchmarks/chi-bench/README.md`](../benchmarks/chi-bench/README.md)
for the field-by-field mapping.

## How it runs

`ManifestEnvironment` runs the **in-sandbox topology** (the architecture's
core): the services run inside the rollout's own sandbox, so the agent
reaches them on `localhost`. During a rollout:

1. `Rollout.start()` provisions the environment — starts the declared
   services inside the sandbox.
2. It gates on `readiness()` — the agent never runs before the environment
   is healthy.
3. `Rollout.cleanup()` tears the environment down.

Run one task or a task directory against an environment manifest with
`bench eval create --tasks-dir ...`. `--environment-manifest` applies the
Environment-plane manifest to every rollout in the Job pipeline.

```bash
# one task
bench eval create --tasks-dir benchmarks/clawsbench/tasks/<task> \
  --environment-manifest benchmarks/clawsbench/environment.toml \
  --agent claude-agent-acp --model claude-haiku-4-5

# task directory
bench eval create --tasks-dir benchmarks/clawsbench/tasks \
  --environment-manifest benchmarks/clawsbench/environment.toml \
  --agent claude-agent-acp --model claude-haiku-4-5
```

YAML configs may declare the same seam with ``environment_manifest:
<path>`` at the top level so the batch run is reproducible from disk.

`--environment-manifest` is distinct from `--sandbox`: the sandbox is *where*
it runs (the Sandbox plane); the environment manifest is *the world* (the
Environment plane).

## Exporting for training

A scored rollout's trajectory exports to the Verifiers / ORS dataset format
that prime-rl ingests — `benchflow.trajectories.export`:

```python
from benchflow.trajectories.export import (
    trajectory_to_verifiers_record,
    export_trajectories_to_jsonl,
)

record = trajectory_to_verifiers_record(
    task_id="clawsbench/archive-alice",
    messages=trajectory_messages,
    verify_result=verify_result,
    model="claude-haiku-4-5",
    environment="clawsbench",
)
export_trajectories_to_jsonl([record], "dataset.jsonl")
```

Each line is one record: `prompt`, `completion`, `reward`, `metrics`,
`is_completed`, `is_truncated`, `example_id`, `info` — the shape pinned
against the Verifiers `RolloutOutput` type.

## Roll-back — `snapshot` / `restore`

`snapshot` / `restore` are **real**. For an environment that declares an
`[environment.state]` table, `snapshot()` copies each declared SQLite file
with `sqlite3 .backup` (a consistent online backup) into a per-snapshot
directory inside the sandbox, and `restore(snap)` copies the captured files
back over the live paths. This is the substrate `Rollout.branch()` runs on:
a branch quiesces the agent and services, restores a snapshot, and explores
an alternative continuation. An environment with no `[environment.state]`
table is stateless — `snapshot`/`restore` raise `RuntimeError`.

## Reset — `reset`

`reset` returns the environment to the per-task baseline so it can be reused
for a fresh episode without tearing down the sandbox (distinct from
`restore`, which rolls back to an arbitrary snapshot). For an environment
that declares an `[environment.state]` table, `provision` captures a
baseline; `reset` then stops the framework-started services, restores the
baseline, and restarts the services. For an `owns_lifecycle = true` manifest
the framework cannot restart entrypoint-owned services; `reset` is then a
no-op (and the host must recycle the container for a hard reset).

## Not yet implemented

`ManifestEnvironment` does not exercise:

- **Sidecar / shared-fleet topology** — host-exposed ports, `AccountBroker`.
