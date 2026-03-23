# benchflow

Multi-turn agent benchmarking framework. Superset of Harbor.

## Architecture

benchflow = Harbor (environments, verifier, orchestration) + ACP (multi-turn agent communication).

```
benchflow SDK.run()
  → Harbor DockerEnvironment.start()
  → Install ACP agent in container (npm)
  → ContainerProcess: docker compose exec -i (live stdio pipe)
  → ACP: initialize → session/new(cwd=/app) → session/prompt (loop)
  → Agent runs tools on container filesystem
  → Harbor Verifier: tests/test.sh → reward.txt
  → DockerEnvironment.stop()
```

## Key files

- `src/benchflow/sdk.py` — SDK.run() orchestrates everything
- `src/benchflow/container.py` — live stdio pipe to Docker container process
- `src/benchflow/acp/client.py` — ACP JSON-RPC client
- `src/benchflow/acp/container_transport.py` — ACP transport over container pipe
- `src/benchflow/acp/session.py` — tracks tool calls, messages, thoughts
- `src/benchflow/viewer.py` — HTML trajectory viewer for humans
- `src/benchflow/trajectories/` — proxy, otel, atif, claude_code converter
- `src/benchflow/cli/main.py` — benchflow run, benchflow view

## Supported agents

Agents in `AGENT_INSTALLERS` (sdk.py) are auto-installed in containers:
- `claude-agent-acp` — Claude Code via ACP (primary)
- `pi-acp` — Pi agent
- `openclaw` — OpenClaw
- `codex-acp` — OpenAI Codex (needs OPENAI_API_KEY)
- `gemini` — Gemini CLI (needs Google key)

## How to run

```bash
export $(cat .env | xargs)  # ANTHROPIC_API_KEY

# SDK
uv run python -c "
import asyncio
from benchflow.sdk import SDK
result = asyncio.run(SDK().run('.ref/harbor/examples/tasks/hello-world',
    agent='claude-agent-acp',
    model='claude-haiku-4-5-20251001',
    agent_env={'ANTHROPIC_API_KEY': '...'}))
print(result)
"

# CLI
benchflow run -t .ref/harbor/examples/tasks/hello-world -a claude-agent-acp
```

## Testing

```bash
uv run pytest tests/     # unit tests (no Docker needed)
```

Real e2e tests require Docker + ANTHROPIC_API_KEY.

## Harbor dependency

benchflow imports Harbor as a library: `harbor @ git+https://github.com/benchflow-ai/harbor.git`

Harbor provides: DockerEnvironment, Task, TaskConfig, TrialPaths, Verifier, and all models.
benchflow re-exports everything: `from benchflow import Job, Trial, TaskConfig` works.

## Output structure

```
jobs/{job_name}/{trial_name}/
├── result.json           # rewards, agent info, timing
├── prompts.json          # prompts sent to agent
├── trajectory/
│   └── acp_trajectory.jsonl  # ACP session/update events
├── agent/                # Harbor agent logs
├── verifier/
│   ├── reward.txt        # 0 or 1
│   └── ctrf.json         # pytest results
└── artifacts/
```
