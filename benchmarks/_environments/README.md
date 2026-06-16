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

`resolve_environment` parses `name@version`, looks it up here, and content-
addresses it (`sha256:…`) so every run records exactly which environment it bound.

| entry | what it is |
|-------|------------|
| `env0@prod`   | env-0 — 7 services (auth/gmail/slack/gcal/gdoc/gdrive/stripe). The pinned production environment. |
| `env0@outage` | env-0 with gmail + slack removed — the "Same state, tool outage" perturbation variant. |

## Running env0 tasks

env0 per-task images build `FROM xdotli/env0-base:latest`, which is **amd64-only**
— run env0 on **Daytona** (x86_64), not local Docker on Apple Silicon.

env0 tasks (in `benchflow-ai/smolclaws`) author their Dockerfiles with a
repo-root build context (`COPY tasks/<name>/data …`). benchflow builds from each
task's `environment/` directory, so stage them first with the bundled adapter:

```bash
python -m benchflow._utils.build_context_stage <smolclaws>/tasks /tmp/env0-staged
bench eval create --tasks-dir /tmp/env0-staged --environment-manifest env0@prod --sandbox daytona ...
```
