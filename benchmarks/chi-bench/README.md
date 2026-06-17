# chi-bench — Environment-plane manifest

chi-bench is a ~25k-LOC **healthcare simulator** (prior-authorization and
care-management worlds) fronted by a thin MCP transport. It is the **external
proof** of the Stateful Multi-Service Benchmark (SMSB) archetype: structurally
the same machine as ClawsBench, onboarded here by a single ~25-line manifest
with its environment **untouched** — the architecture's zero-modification
conformance bar.

`environment.toml` is the **manifest** — the entire framework-integration
surface. BenchFlow's `ManifestEnvironment` adapter reads it; the chi-bench
image, Dockerfile, and entrypoint are otherwise unmodified. The ~920 LOC of
Harbor coupling chi-bench previously carried collapses into this manifest.

## How it maps to chi-bench

| Manifest field | Why |
|---|---|
| `image = "chi-bench:latest"` | chi-bench ships **one** self-contained, ready-to-run image (`docker/Dockerfile`, `runtime` target); it bakes the server runtime plus every task's fixtures/worlds, so a trial runs by image pull alone |
| `owns_lifecycle = true` | the image's `tini` entrypoint (`docker/entrypoint.sh`) bootstraps the SQLite stores and starts `cb serve` itself — the framework starts nothing |
| `task_selection.mechanism = "env_var"` | one image serves every task; the task id is chosen at runtime, not by a per-task image |
| `task_selection.key = "CHI_BENCH_TASK_ID"` | the exact env var `docker/entrypoint.sh` reads to resolve the per-trial fixtures directory (`exit 64` if unset) |
| `task_selection.inject_into = "entrypoint"` | the var must reach PID 1 — the entrypoint reads it before `exec`-ing the agent, so a per-`docker exec` injection would be invisible |
| `readiness.http` | the unified FastAPI server exposes `GET /health` → `{"status": "ok"}` on `:8023` (`server/routers/common.py`); the framework gates the agent on it, mirroring the entrypoint's own startup probe |
| `ports` | the image `EXPOSE`s `8020 8023 8100 8200` — unified HTTP `:8023` plus three FastMCP streamable-http servers: provider `:8020`, payer `:8100`, care-management `:8200` |
| `forward_env` | the verifier's `WorkspaceJudge` shells out to the Claude Code CLI, which needs `ANTHROPIC_API_KEY` forwarded from the host |

## Lifecycle

The `chi-bench` image **owns its own lifecycle**: `docker/entrypoint.sh`
resolves `CHI_BENCH_TASK_ID`, wires `CHI_BENCH_FIXTURES_DIR`, runs
`python -m chi_bench.bootstrap`, starts `cb serve --host 0.0.0.0 --port 8023`,
then blocks until `:8023/health` answers and each of `:8020 / :8100 / :8200`
accepts a real MCP `initialize` — only then does it `exec` the agent. The
framework's readiness gate probes the same `:8023/health` endpoint.

## Run it

```bash
bench eval run --tasks-dir benchmarks/chi-bench/tasks/<task> \
  --environment-manifest benchmarks/chi-bench/environment.toml \
  --agent claude-agent-acp --model claude-haiku-4-5
```

## Not yet implemented

`snapshot` / `restore` (environment-state branching) are the platform layer —
`ManifestEnvironment` raises `NotImplementedError` for them. chi-bench's state
is SQLite, so this is the natural place a future `[environment.state]` table
would land.
