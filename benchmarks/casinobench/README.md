# CasinoBench — in-sandbox env-0 casino benchmark

CasinoBench is a stateful, single-service BenchFlow benchmark. A casino
**mock-service** (FastAPI/uvicorn) runs *inside the rollout's own sandbox* — the
env-0 / ClawsBench pattern — and the agent plays through the `casino` seven-tool
CLI over `localhost:9001`. Each game is one task; the game is selected per task
by the `CASINOBENCH_GAME` env var. The service is the chip authority, so the
verifier just reads the live final standing and scores net chips. There is **no
host-subprocess wiring** — the service is declared in the manifest and started
by BenchFlow's Environment plane in-sandbox.

```
benchmarks/casinobench/
├── environment.toml                 # the in-sandbox manifest (the whole framework seam)
├── docker/casinobench-base.Dockerfile  # casinobench-base: working service + ACP backends + casino CLI
├── verifier/test.sh                 # shared scorer: curl /_admin/state -> net chips -> reward
└── tasks/
    └── blackjack/
        ├── task.md                  # CASINOBENCH_GAME=six-deck-blackjack-s17, manifest -> ../../environment.toml
        └── tests/test.sh            # symlink -> ../../../verifier/test.sh
```

## The base image

`casino-service` (the env-0 mock-service) and the ACP seats live in **one**
image so competing agents can share a sandbox. The image is built in two steps:

```bash
CB=~/casinobench   # your casinobench engine checkout (proprietary, separate repo)

# 1. assemble the agent-seat build context (gitignored — see agent_env/.gitignore)
cp src/benchflow/agents/deepagents_acp_shim.py examples/casino/agent_env/deepagents-acp-shim
cp -r "$CB/packages/environments/casino"        examples/casino/agent_env/casino-pkg
cp -r "$CB"                                      examples/casino/agent_env/casinobench-engine

# 2. the seat image (ACP backends + casino CLI; service still a --no-deps stub)
docker build -t casino-agent-seat:latest examples/casino/agent_env \
  -f examples/casino/agent_env/Dockerfile

# 3. casinobench-base: extends the seat image with the engine + service extra so
#    `casino-service` actually runs. This is the manifest's run `image`.
docker build -t env0acrdd8632.azurecr.io/casinobench-base:2.0.1 examples/casino/agent_env \
  -f benchmarks/casinobench/docker/casinobench-base.Dockerfile
```

Verify the service is real (not the broken `--no-deps` stub):

```bash
docker run --rm env0acrdd8632.azurecr.io/casinobench-base:2.0.1 casino-service --help   # exits 0
```

## Run it

```bash
bench eval run \
  --tasks-dir benchmarks/casinobench/tasks/blackjack \
  --environment-manifest benchmarks/casinobench/environment.toml \
  --agents roster.yaml
```

`roster.yaml` lists the seats (each an ACP backend baked into the base image):

```yaml
agents:
  - { name: claude, agent: claude-agent-acp, model: claude-haiku-4-5 }
  - { name: codex,  agent: codex-acp,        model: gpt-5.5 }
```

Single-agent runs work too — swap `--agents roster.yaml` for `--agent
claude-agent-acp --model claude-haiku-4-5`.

## How a trial flows

1. The Environment plane reads `environment.toml`, runs `env0acrdd8632.azurecr.io/casinobench-base:2.0.1`,
   forwards `CASINOBENCH_GAME` / `CASINOBENCH_HANDS` / `BENCHFLOW_SEED`, starts
   `casino-service` on `:9001`, and health-gates it on `/health`.
2. The agent plays via the `casino` CLI (`lobby` → `join` → `observe`/`act` …).
   The CLI is a thin HTTP client of `$CASINO_URL` (default `http://localhost:9001`).
3. `tests/test.sh` (the shared `verifier/test.sh`) curls `/_admin/state` for the
   final chips and writes the net-chips reward to `/logs/verifier/reward.txt`
   (and `reward.json`). A failed read aborts with no reward file so a verifier
   error is recorded rather than a fabricated `0`.

## Adding another game

Copy `tasks/blackjack/` to `tasks/<game>/`, set `CASINOBENCH_GAME` to a
registered game id (`casino lobby` lists them — e.g. `european-roulette`,
`punto-banco-baccarat`, `jacks-or-better-video-poker`, `infinite-deck-blackjack`),
and keep the `tests/test.sh` symlink to the shared verifier.
