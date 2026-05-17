# Getting started
A 5-minute path from install to first eval.

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- Docker for local sandboxes, `DAYTONA_API_KEY` for Daytona cloud runs, or Modal auth for Modal-backed runs
- An API key or subscription/OAuth auth for at least one agent (see below)

## Install

```bash
uv tool install benchflow
```

This gives you the `benchflow` (alias `bench`) CLI plus the Python SDK. To install for editable development:

```bash
git clone https://github.com/benchflow-ai/benchflow
cd benchflow
uv sync --extra dev --locked
```

## Auth: OAuth, long-lived token, or API key

You don't need an API key if you're a Claude / Codex / Gemini subscriber. Three options, pick one per agent:

### Option 1 — Subscription OAuth from host CLI login

If you've logged into the agent's CLI on your host (`claude login`, `codex --login`, `gemini` interactive flow), benchflow picks up the credential file and copies it into the sandbox. No API key billing.

| Agent | How to log in on the host | What benchflow detects | Replaces env var |
|-------|---------------------------|------------------------|------------------|
| `claude-agent-acp` | `claude login` (Claude Code CLI) | `~/.claude/.credentials.json` | `ANTHROPIC_API_KEY` |
| `codex-acp` | `codex --login` (Codex CLI) | `~/.codex/auth.json` | `OPENAI_API_KEY` |
| `gemini` | `gemini` (interactive login) | `~/.gemini/oauth_creds.json` | `GEMINI_API_KEY` |

When benchflow finds the detect file, you'll see:

```
Using host subscription auth (no ANTHROPIC_API_KEY set)
```

### Option 2 — Long-lived OAuth token (CI / headless)

For CI pipelines, scripts, or anywhere the host can't run an interactive browser login, generate a 1-year OAuth token with `claude setup-token` and export it:

```bash
claude setup-token            # walks you through browser auth, prints a token
export CLAUDE_CODE_OAUTH_TOKEN=<paste-token>
```

benchflow auto-inherits `CLAUDE_CODE_OAUTH_TOKEN` from your shell into the sandbox; the Claude CLI inside reads it directly. Same auth precedence as plain `claude` ([Anthropic docs](https://code.claude.com/docs/en/authentication#authentication-precedence)): API keys override OAuth tokens, so unset `ANTHROPIC_API_KEY` if you want the token to win.

`claude setup-token` only authenticates Claude. Codex and Gemini do not have an equivalent today — use Option 1 (host login) or Option 3 (API key).

### Option 3 — API key

Set the API-key env var directly. Works with every agent:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
export LLM_API_KEY=...           # OpenHands / LiteLLM-compatible providers
```

benchflow auto-inherits well-known API key env vars from your shell into the sandbox.

### Precedence

If multiple credentials are set, benchflow / the agent CLI uses (high to low): cloud provider creds → `ANTHROPIC_AUTH_TOKEN` → `ANTHROPIC_API_KEY` → `apiKeyHelper` → `CLAUDE_CODE_OAUTH_TOKEN` → host subscription OAuth. To force a lower-priority option, unset the higher one in your shell before running.

## Run your first eval

```bash
# Single task from a remote repo
GEMINI_API_KEY=... bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks/edit-pdf \
  --agent gemini \
  --model gemini-3.1-pro-preview \
  --sandbox docker

# Single task from local path
GEMINI_API_KEY=... bench eval create \
  --tasks-dir tasks/edit-pdf \
  --agent gemini \
  --model gemini-3.1-pro-preview \
  --sandbox daytona \
  --skills-dir tasks/edit-pdf/environment/skills \
  --agent-env BENCHFLOW_SKILL_NUDGE=name

# A whole batch from YAML config
bench eval create --config benchmarks/skillsbench-claude-glm51.yaml

# Batch from remote repo with concurrency
GEMINI_API_KEY=... bench eval create \
    --source-repo benchflow-ai/skillsbench --source-path tasks \
    --agent gemini --model gemini-3.1-pro-preview --sandbox daytona --concurrency 32

# List the registered agents
bench agent list
```

`bench eval create` is the primary command for running evaluations — it works for
single tasks, batch runs, and remote repos. Use `--source-repo <org/repo>
--source-path <subpath>` to fetch from a remote repo, `--tasks-dir <dir>` for a
local directory, or `--config <config.yaml>` for a YAML config. Results land under
`evaluations/<eval-name>/<rollout-name>/` — `result.json` for the verifier output,
`trajectory/acp_trajectory.jsonl` for the full agent trace.

When you mount skills, use `BENCHFLOW_SKILL_NUDGE=name` as the default docs
option. It tells the agent which skills are available and where to read them.
For more context in the prompt, use `description` or `full`; omit the env var
to keep BenchFlow's runtime default off.

## Run from Python

The CLI is a thin shim over the Python API. For programmatic use:

```python
import benchflow as bf
from benchflow import RolloutConfig, Scene
from benchflow.task_download import resolve_source

config = RolloutConfig(
    task_path=resolve_source("benchflow-ai/skillsbench", path="tasks/edit-pdf"),
    scenes=[Scene.single(agent="gemini", model="gemini-3.1-pro-preview")],
    environment="docker",
)
result = await bf.run(config)
print(result.rewards)         # {'reward': 1.0}
print(result.n_tool_calls)
```

`Rollout` is decomposable — invoke each lifecycle phase individually for custom flows. See [Concepts: rollout lifecycle](./concepts.md#rollout-lifecycle).

## What to read next

| If you want to… | Read |
|------------------|------|
| Understand the model — Rollout, Scene, Role, Verifier | [Concepts](./concepts.md) |
| Author a task | [Task authoring](./task-authoring.md) |
| Run multi-agent patterns (coder/reviewer, simulated user, BYOS) | [Use cases](./use-cases.md) |
| Run multi-round single-agent (progressive disclosure) | [Progressive disclosure](./progressive-disclosure.md) |
| Evaluate skills, not tasks | [Skill eval](./skill-eval.md) |
| Understand the security model | [Sandbox hardening](./sandbox-hardening.md) |
| CLI flags + commands | [CLI reference](./reference/cli.md) |
| Python API surface | [Python API reference](./reference/python-api.md) |
