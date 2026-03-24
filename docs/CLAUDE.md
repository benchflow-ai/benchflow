# benchflow

Multi-turn agent benchmarking framework. Superset of Harbor.

## Architecture

benchflow = Harbor (environments, verifier, orchestration) + ACP (multi-turn agent communication).

```
benchflow SDK.run()
  → Harbor Environment.start() (Docker or Daytona)
  → Install ACP agent in sandbox (npm)
  → LiveProcess: Docker exec or Daytona SSH (live stdio pipe)
  → ACP: initialize → session/new(cwd) → session/set_model → session/prompt (loop)
  → Agent runs tools on sandbox filesystem
  → Harbor Verifier: tests/test.sh → reward.txt
  → Environment.stop()
```

## Key files

- `src/benchflow/sdk.py` — SDK.run() orchestrates everything
- `src/benchflow/job.py` — Job orchestration with concurrency, retries, resume
- `src/benchflow/metrics.py` — collect_metrics() for aggregating results
- `src/benchflow/process.py` — LiveProcess abstraction (DockerProcess, DaytonaProcess)
- `src/benchflow/agents/registry.py` — agent configs (install, launch, env requirements)
- `src/benchflow/acp/client.py` — ACP JSON-RPC client
- `src/benchflow/acp/container_transport.py` — ACP transport over container pipe
- `src/benchflow/acp/session.py` — tracks tool calls, messages, thoughts
- `src/benchflow/viewer.py` — HTML trajectory viewer
- `src/benchflow/cli/main.py` — benchflow run, benchflow view

## Supported agents

Agents registered in `src/benchflow/agents/registry.py`:
- `claude-agent-acp` — Claude Code via ACP (primary, tested)
- `pi-acp` — Pi coding agent via ACP (tested, needs `@mariozechner/pi-coding-agent`)
- `openclaw` — OpenClaw (incompatible — needs gateway session lifecycle)
- `codex-acp` — OpenAI Codex (needs OPENAI_API_KEY, untested)
- `gemini` — Gemini CLI (needs GOOGLE_API_KEY, untested)

## How to run

```bash
source .env  # ANTHROPIC_API_KEY, DAYTONA_API_KEY

# SDK — API keys auto-inherited from environment
python -c "
import asyncio
from benchflow import SDK
result = asyncio.run(SDK().run(
    '.ref/terminal-bench-2/log-summary-date-ranges',
    agent='claude-agent-acp',
    model='claude-haiku-4-5-20251001',
    environment='daytona',
))
print(result.rewards)
"

# Job — run multiple tasks with concurrency
python -c "
import asyncio
from benchflow import Job, JobConfig
result = asyncio.run(Job(
    tasks_dir='.ref/terminal-bench-2',
    jobs_dir='jobs/tb2-run',
    config=JobConfig(agent='claude-agent-acp', model='claude-haiku-4-5-20251001',
                     environment='daytona', concurrency=64),
).run())
print(f'{result.passed}/{result.total}')
"

# CLI
benchflow run -t .ref/terminal-bench-2/log-summary-date-ranges -a claude-agent-acp
benchflow view jobs/tb2-run/<trial-name>/
```

## Testing

Use Haiku 4.5 (`claude-haiku-4-5-20251001`) for all testing/dogfood runs.

```bash
pytest tests/     # unit tests (no Docker needed)
```

Real e2e tests require Daytona or Docker + ANTHROPIC_API_KEY.
Dogfood script: `docs/dogfood/DOGFOOD.md`

## Harbor dependency

benchflow imports Harbor as a library: `harbor @ git+https://github.com/benchflow-ai/harbor.git`

Harbor provides: DockerEnvironment, DaytonaEnvironment, Task, TaskConfig, TrialPaths, Verifier, and all models.
benchflow re-exports everything: `from benchflow import Job, JobConfig, SDK, collect_metrics` works.

## Output structure

```
jobs/{job_name}/{trial_name}/
├── result.json           # rewards, agent info, timing
├── prompts.json          # prompts sent to agent
├── trajectory/
│   └── acp_trajectory.jsonl  # ACP tool calls + agent thoughts
├── verifier/
│   ├── reward.txt        # float reward
│   ├── test-stdout.txt   # verifier output
│   └── ctrf.json         # pytest results
└── summary.json          # job-level aggregates (at job_name level)
```

## Key docs

- `PLAN.md` — project roadmap and status
- `docs/parity/RESULTS.md` — benchmark results vs official numbers
- `docs/GAP_ANALYSIS.md` — feature gaps and testing findings
- `docs/dogfood/DOGFOOD.md` — end-to-end test prompt
