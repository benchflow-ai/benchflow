# Committed environment registry

A git-tracked [environment registry](../../src/benchflow/_utils/env_registry.py)
so the env-axis pins from PR #790 resolve from the repo instead of a `/tmp` dir.

Each `<name>@<version>.toml` **or** `<name>@<version>.yaml` is an Environment-plane
manifest. YAML is canonical for new manifests (consistent with the task / run /
job configs); TOML stays supported for back-compat. Bind one at the command line,
decoupled from the task:

```bash
export BENCHFLOW_ENV_REGISTRY=benchmarks/_environments
bench eval create --tasks-dir <tasks> --environment-manifest env0@prod  --sandbox daytona ...
bench eval create --tasks-dir <tasks> --environment-manifest env0@outage --sandbox daytona ...
```

`$BENCHFLOW_ENV_REGISTRY` can be **any** local directory of
`name@version.toml` files — a pip install has no repo checkout, so download
the pinned manifest instead (e.g. from
`https://raw.githubusercontent.com/benchflow-ai/benchflow/main/benchmarks/_environments/env0@prod.toml`
into `./env-registry/`, then `export BENCHFLOW_ENV_REGISTRY=$PWD/env-registry`).

`resolve_environment` parses `name@version`, looks it up here, and content-
addresses it (`sha256:…`) so every run records exactly which environment it bound.

| entry | what it is |
|-------|------------|
| `env0@prod`   | env-0 — 8 services (mock-auth/-gmail/-gcal/-gdrive/-gdoc/-slack/-discord/-stripe), mirroring upstream `tasks/_manifests/env-0.toml` verbatim. The pinned production environment. |
| `env0@outage` | The "Same state, tool outage" perturbation variant: `env0@prod` minus the `mock-gmail` and `mock-slack` services (6 of the 8 declared). Bind the same task to both pins to attribute the reward delta to the environment. |

## Running env0 tasks

The env0 tasks are public in
[`benchflow-ai/env0`](https://github.com/benchflow-ai/env0) under `tasks/` —
the 60-task Standard60 snapshot (exact list in its
`tasks/STANDARD60_MANIFEST.txt`).

env0 per-task images build `FROM ghcr.io/benchflow-ai/env0:0.2.0`, which is
**amd64-only** — run env0 on **Daytona** (x86_64), not local Docker on Apple
Silicon.

env0 tasks author their Dockerfiles with a repo-root build context
(`COPY tasks/<name>/data …`). benchflow builds from each task's
`environment/` directory, so stage them first with the bundled adapter:

```bash
git clone https://github.com/benchflow-ai/env0
python -m benchflow._utils.build_context_stage env0/tasks /tmp/env0-staged
bench eval create --tasks-dir /tmp/env0-staged --environment-manifest env0@prod --sandbox daytona ...
```

env0's `task.md` frontmatter also pins a manifest
(`benchflow.environment.manifest: ../_manifests/env-0.toml`); an explicit
`--environment-manifest` (or `--state`) always overrides that pin — the
frontmatter applies only when neither flag is given.
