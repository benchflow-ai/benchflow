# Native concurrent multi-agent floor (`benchflow arena run --agents`)

Run **N agents on ONE shared task + its ONE service, concurrently, in ONE shared
sandbox** — each agent in its own `/work/<seat>` folder, each with its own ACP
trajectory and (proxy seats only) a separate raw `llm_trajectory.jsonl`.

```bash
set -a; . ~/sb-run.env; set +a          # API keys for proxy seats
uv run benchflow arena run --agents examples/casino/agents.yaml
```

## `agents.yaml`

One shared task, N seats. Each seat names a **prebuilt** benchflow agent
(`agent:`) **or** a **BYOA** agent manifest (`manifest:`) — exactly one — plus its
model and an optional per-agent instruction file. `count` fans a seat out into
`name-0..name-(n-1)`.

```yaml
sandbox: { image_dir: agent_env }        # the shared image (Dockerfile dir, relative to this file)
services:
  url_env: CASINO_URL                    # env var the in-sandbox task CLI reads for the service URL
  command: "uv run casino-service --host 0.0.0.0 --port {port}"   # host service, reached over the bridge
  cwd: ~/casinobench
  standings_path: /_admin/standings      # optional → per-seat reward vector in floor.json
out: out/native-floor/casino
drive: auto-loop                         # auto-loop | service-rounds
prompt: "Play through the casino CLI. Begin with `casino lobby`."
agents:
  - { name: codex,    agent: codex-acp,         model: gpt-5.5,            count: 2, instructions: prompts/aggressive.md }
  - { name: claude,   agent: claude-agent-acp,  model: claude-sonnet-4-6,  count: 2, instructions: prompts/cautious.md }
  - { name: deepseek, agent: deepagents,        model: deepseek-v4-pro,    instructions: prompts/aggressive.md }  # proxy → raw+acp
  - { name: mine,     manifest: agents/my.toml, model: gpt-5.5 }            # BYOA (ACP manifest contract)
```

## What you get

```
out/native-floor/casino/
├── roster.json                       # seat → agent / model / protocol / byoa
├── floor.json                        # per-seat status + (opt) standings + reward vector
└── <seat>/trajectory/
    ├── acp_trajectory.jsonl          # every seat — the agent's tool calls + thinking
    └── llm_trajectory.jsonl          # PROXY seats only — the raw LiteLLM exchanges
```

### Instruction files (per agent family)

The runner writes each seat's `instructions:` body into `/work/<seat>/<file>`
**before** launch — `CLAUDE.md` for claude-agent-acp, `GEMINI.md` for gemini,
`AGENTS.md` for everything else (`AgentConfig.instruction_filename`).

### Raw-LLM coverage is partial by design

Subscription seats (codex / claude **oauth**) call their provider directly, so
they produce an **ACP trajectory only** (`raw=false` in `floor.json`). Only
**proxy-routed** seats (an API key fronted by that seat's own LiteLLM proxy —
e.g. `deepagents` on deepseek) also get a separate raw `llm_trajectory.jsonl`.

### Drive modes

- `auto-loop` (default, verified) — one prompt; the agent runs its own
  observe→act loop via the in-sandbox CLI. Multi-round happens inside the prompt.
- `service-rounds` (structural) — **the mock service drives the rounds**: the
  runner polls the shared service per seat and re-prompts (nudges) the seat only
  on `YOUR_TURN`, until `DONE`/deadline (re-entrant per round). The service
  controls pacing; the agent acts once per nudge through its own tools.

## Agent paths

All three benchflow-ai/agents families collapse to one `AgentConfig`, so the
runner never branches on "which path":

- **raw ACP** + **ai-sdk** → `protocol=acp` → `connect_acp` (the verified path).
- **omnigent** → `protocol=session-factory` → `Agent.connect`/`Session.prompt`
  (structural; no session-factory agent is registered in this repo yet).

**BYOA** = a `manifest.toml` following the data-only agent contract (ACP-only,
strict / no unknown fields); it is schema-validated then `register_agent`-ed,
indistinguishable downstream from a prebuilt.
