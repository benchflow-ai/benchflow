# Integration Tests

On-demand end-to-end tests that run 9 SkillsBench tasks across all 8 registered agents on Daytona. Not part of CI — invoke manually to validate the full pipeline.

## Prerequisites

| Variable | Required for |
|---|---|
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | gemini, pi-acp, openclaw, opencode, openhands |
| `DAYTONA_API_KEY` | all agents (sandbox backend) |
| `CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_API_KEY` | claude-agent-acp |
| `OPENAI_API_KEY` | codex-acp |

## Quick Start

```bash
# All 8 agents in parallel (each runs 9 tasks concurrently on Daytona)
export GEMINI_API_KEY=... DAYTONA_API_KEY=... CLAUDE_CODE_OAUTH_TOKEN=... OPENAI_API_KEY=...
tests/integration/run.sh

# Specific agents only
tests/integration/run.sh gemini pi-acp claude-agent-acp

# Review results from a previous run (no API calls)
tests/integration/run.sh --check-only
```

## What It Does

1. **Resolves tasks** — downloads the full SkillsBench task set, then creates a symlinked subset of 9 selected tasks.
2. **Launches agents in parallel** — each agent is started as a background process running `bench eval create` with concurrency=30.
3. **Waits and reports** — as each agent finishes, prints its score line. After all complete, runs `check_results.py` to validate output schema and print the results table.

## Architecture

```
tests/integration/
├── run.sh              # Shell driver — parallel agent launch + wait
├── check_results.py    # Result validator — schema checks + score table
└── configs/            # Per-agent YAML configs (reference; run.sh uses CLI args)
    ├── claude-agent-acp.yaml
    ├── codex-acp.yaml
    ├── gemini.yaml
    └── ...
```

Output lands in `jobs/integration/<agent>/`:
```
jobs/integration/
├── gemini/
│   ├── 2026-05-15__16-43-54/    # run directory
│   │   ├── jax-computing-basics__abc123/
│   │   │   ├── result.json
│   │   │   ├── trajectory/acp_trajectory.jsonl
│   │   │   └── ...
│   │   └── ...
│   └── summary.json
├── claude-agent-acp/
│   └── ...
└── .logs/                        # per-agent stdout/stderr logs
    ├── gemini.log
    └── ...
```

## Selected Tasks

The 9 tasks (3 low / 3 medium / 3 high complexity):

| Task | Complexity |
|---|---|
| jax-computing-basics | Low |
| python-scala-translation | Low |
| jpg-ocr-stat | Low |
| grid-dispatch-operator | Medium |
| threejs-to-obj | Medium |
| data-to-d3 | Medium |
| lake-warming-attribution | High |
| weighted-gdp-calc | High |
| shock-analysis-supply | High |

## Agents

All 8 registered agents run by default:

| Agent | Default Model | Notes |
|---|---|---|
| claude-agent-acp | claude-haiku-4-5-20251001 | Needs `CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_API_KEY` |
| codex-acp | gpt-5.4-nano | Needs `OPENAI_API_KEY` |
| pi-acp | gemini-3.1-flash-lite-preview | |
| openclaw | gemini-3.1-flash-lite-preview | |
| gemini | gemini-3.1-flash-lite-preview | |
| opencode | gemini-3.1-flash-lite-preview | |
| harvey-lab-harness | gemini-3.1-flash-lite-preview | |
| openhands | gemini-3.1-flash-lite-preview | |

Agents missing credentials are automatically skipped.

## Standalone YAML Configs

The `configs/` directory has per-agent YAML files usable with `bench eval create -f`:

```bash
uv run bench eval create -f tests/integration/configs/gemini.yaml
```

These are reference configs — `run.sh` uses CLI arguments directly for more control over task filtering.

## Result Validation

`check_results.py` checks:
- Every `result.json` has required fields (`task_name`, `agent`, `model`, `rewards`, `timing`)
- No infrastructure errors (sandbox failures vs. task failures)
- `summary.json` exists with required keys (`total`, `passed`, `failed`, `score`)

```bash
# Run validator standalone
uv run python tests/integration/check_results.py jobs/integration gemini pi-acp
```

## Cost

Approximate cost per full run: ~$0.36 (9 tasks × 8 agents using flash-lite/haiku/nano models).
