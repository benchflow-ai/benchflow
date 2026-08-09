# Environment registry — moved into the package

The committed environment pins that used to live here (`env0@prod.toml`,
`env0@outage.toml`) are now **built-in registry** entries shipped inside the
`benchflow` wheel:

    src/benchflow/environment/_registry/

That directory is the single source of truth — this one intentionally keeps no
copies, so the pins cannot drift. The move fixes the pip-install gap: a bare
`pip install benchflow` used to have no checkout to point
`$BENCHFLOW_ENV_REGISTRY` at and had to hand-download the pins; now the specs
resolve with **no env vars and no checkout**:

```bash
bench eval create --tasks-dir <tasks> --environment-manifest env0@prod  --sandbox daytona ...
bench eval create --tasks-dir <tasks> --environment-manifest env0@outage --sandbox daytona ...
```

`$BENCHFLOW_ENV_REGISTRY` still works and, when set, wins entirely: point it at
any local directory of `name@version.toml` (or `.yaml`) files to resolve your
own pins instead of the built-ins. In a checkout that is simply:

```bash
export BENCHFLOW_ENV_REGISTRY=src/benchflow/environment/_registry
```

Resolution stays content-addressed (`env_hash = sha256(manifest bytes)`) so
every run records exactly which environment it bound. See
[`docs/environment-plane.md`](../../docs/environment-plane.md#registry-nameversion)
for the full registry contract and
[`src/benchflow/_utils/env_registry.py`](../../src/benchflow/_utils/env_registry.py)
for the resolver.

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
