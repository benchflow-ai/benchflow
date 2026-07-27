# Real multi-agent casino floor (arena-concurrent, on BenchFlow)

N **real autonomous ACP agents** play ONE shared [casinobench](https://github.com/benchflow-ai/casinobench)
World concurrently — the live realization of the deferred `arena-concurrent` mode.
Each seat is a benchflow ACP agent in its OWN sandbox, all competing on one
leaderboard; each agent's raw + ACP trajectory is captured per seat.

- `run_floor.py` — starts casinobench's shared World on the host, then runs a
  roster of seats concurrently (`asyncio.gather`), each a `connect_acp` agent in a
  `DockerSandbox` reaching the World over the docker bridge. Subscription agents
  (codex / claude-code) get their auth uploaded per seat and produce
  `acp_trajectory.jsonl`; deepseek/proxy seats also get a per-seat raw
  `llm_trajectory.jsonl`.
- `town_snapshot.py` — serves a live Stanford-Town-style floor viewer:
  casinobench's `render_html` canvas board (agents walking to game stations) in
  live mode, with a click-to-open per-agent **run dossier** injected. Polls the
  World, falls back to the persisted run when it ends, and feeds same-origin JSON
  so a Cloudflare tunnel can publish it.
- `agent_env/` — the seat image (`casino-agent-seat`): Node + `codex-acp` +
  `claude-agent-acp` (via benchflow's install commands) + the `casino` seven-tool
  CLI. The deepagents shim and the casino CLI package are **assembled** into the
  build context (gitignored — see `agent_env/.gitignore`).

## Setup
This example depends on a local casinobench checkout (a separate repo). Assemble
the seat-image build context, then build it:

```bash
CB=~/casinobench   # your casinobench checkout
cp src/benchflow/agents/deepagents_acp_shim.py examples/casino/agent_env/deepagents-acp-shim
cp -r "$CB/packages/environments/casino" examples/casino/agent_env/casino-pkg
docker build -t casino-agent-seat:latest examples/casino/agent_env

set -a; . ~/sb-run.env; set +a          # DEEPSEEK_API_KEY (proxy seats)
# codex/claude seats use the host's ~/.codex/auth.json + ~/.claude/.credentials.json subscriptions
uv run python examples/casino/run_floor.py --world-port 9100
# in another shell, publish the live viewer:
cd "$CB" && uv run python <benchflow>/examples/casino/town_snapshot.py \
    http://127.0.0.1:9100 <benchflow>/out/casino-floor/all-games ./serve &
cloudflared tunnel --url http://localhost:8899   # serving ./serve
```

The roster (agents × models) and the seat prompt are at the top of `run_floor.py`.
Only the models a subscription actually exposes work (e.g. codex→`gpt-5.5`,
claude→`claude-sonnet-4-6`/`claude-haiku-4-5`); others are rejected by the plan.
