# Getting started

A 5-minute path from install to first eval.

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- Docker (for local sandboxes) and/or `DAYTONA_API_KEY` (for cloud sandboxes)
- An API key or subscription/OAuth auth for at least one agent (see below)

## Install

```bash
uv tool install benchflow
```

This gives you the `benchflow` (alias `bench`) CLI plus the Python SDK. To install for editable development:

```bash
git clone https://github.com/benchflow-ai/benchflow
cd benchflow
uv venv -p 3.12 .venv && uv pip install -e ".[dev]"
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
# Single task with Gemini
GEMINI_API_KEY=... bench eval create -t .ref/terminal-bench-2/regex-log -a gemini \
    -m gemini-3.1-pro-preview -e docker

# A whole batch with concurrency
GEMINI_API_KEY=... bench eval create -t .ref/terminal-bench-2 -a gemini \
    -m gemini-3.1-pro-preview -e daytona -c 32

# List the registered agents
bench agent list
```

`bench eval create -t <task>` runs once on a single task or, if the path contains multiple `task.toml`-bearing subdirectories, batches them. Results land under `jobs/<job-name>/<trial-name>/` — `result.json` for the verifier output, `trajectory/acp_trajectory.jsonl` for the full agent trace.

## Run from Python

The CLI is a thin shim over the Python API. For programmatic use:

```python
import benchflow as bf
from benchflow.trial import TrialConfig, Scene
from pathlib import Path

config = TrialConfig(
    task_path=Path(".ref/terminal-bench-2/regex-log"),
    scenes=[Scene.single(agent="gemini", model="gemini-3.1-pro-preview")],
    environment="docker",
)
result = await bf.run(config)
print(result.rewards)         # {'reward': 1.0}
print(result.n_tool_calls)
```

`Trial` is decomposable — invoke each lifecycle phase individually for custom flows. See [Concepts: trial lifecycle](./concepts.md#trial-lifecycle).

## What to read next

| If you want to… | Read |
|------------------|------|
| Understand the model — Trial, Scene, Role, Verifier | [`concepts.md`](./concepts.md) |
| Author a task | [`task-authoring.md`](./task-authoring.md) |
| Run multi-agent patterns (coder/reviewer, simulated user, BYOS) | [`use-cases.md`](./use-cases.md) |
| Run multi-round single-agent (progressive disclosure) | [`progressive-disclosure.md`](./progressive-disclosure.md) |
| Evaluate skills, not tasks | [`skill-eval.md`](./skill-eval.md) |
| Understand the security model | [`sandbox-hardening.md`](./sandbox-hardening.md) |
| CLI flags + commands | [`reference/cli.md`](./reference/cli.md) |
| Python API surface | [`reference/python-api.md`](./reference/python-api.md) |
