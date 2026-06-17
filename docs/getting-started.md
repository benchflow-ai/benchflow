# Getting started
A 5-minute path from install to first eval.

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- Docker for local sandboxes; install `benchflow[sandbox-daytona]` + set `DAYTONA_API_KEY` for Daytona cloud runs, or install `benchflow[sandbox-modal]` for Modal-backed runs
- An API key or subscription/OAuth auth for at least one agent (see below)

## Install

`0.6.2` is the latest stable release on PyPI. Install (or upgrade) with uv or pip:

```bash
uv tool install benchflow                  # add --upgrade to refresh
pip install --upgrade benchflow            # pip equivalent
```

If `uv` reports `Executables already exist: bench, benchflow`, rerun with
`--force` to replace older non-`uv` entrypoints. Confirm with `bench --version`;
the stable line should report `0.6.2`. See [Release channels](./release.md) for
the full command matrix.

For optional sandbox integrations, include the extra in the tool install:

```bash
uv tool install 'benchflow[sandbox-daytona]'
uv tool install 'benchflow[sandbox-modal]'
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

`claude setup-token` only authenticates Claude. Codex can also use a provided subscription access token, such as `CODEX_ACCESS_TOKEN` from a host/orchestrator integration; benchflow passes it through to Codex without copying `~/.codex/auth.json`. Gemini does not have an equivalent today — use Option 1 (host login) or Option 3 (API key).

### Option 3 — API key

Set the API-key env var directly. Works with every agent:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export CODEX_API_KEY=sk-...       # Codex alias for OPENAI_API_KEY
export GEMINI_API_KEY=...
export LLM_API_KEY=...           # OpenHands / LiteLLM-compatible providers
export AZURE_API_KEY=...
export AZURE_API_ENDPOINT=https://<resource>.openai.azure.com/
```

benchflow auto-inherits well-known API key env vars from your shell into the sandbox.
Provider-prefixed models can use credentials that differ from the agent's
native default auth. For Azure Foundry, use models such as
`azure-foundry-openai/gpt-5.5` or
`azure-foundry-anthropic/claude-opus-4-5`; benchflow derives the Azure resource
from `AZURE_API_ENDPOINT` and routes the selected agent through a generated
LiteLLM gateway config.

Several providers with user-supplied endpoints — `deepseek`, `glm`, `kimi`,
`minimax`, `hunyuan`, and others — follow the `<PROVIDER>_API_KEY` +
`<PROVIDER>_BASE_URL` convention; providers with fixed endpoints (such as
`zai` or `openai`) need only the API key. For example, `deepseek/<model>`
reads:

```bash
export DEEPSEEK_API_KEY=...
export DEEPSEEK_BASE_URL=https://api.deepseek.com
```

If the base URL is missing, the run fails with
`Provider 'deepseek' for model 'deepseek/<model>' requires DEEPSEEK_BASE_URL to build the provider base URL.`

These variables must be **exported** to reach the benchflow runtime — a plain
shell assignment or a `source .env` without `export` stays local to your shell
and never reaches the `bench` process. The portable pattern for a `.env` file:

```bash
set -a; source .env; set +a
bench eval run ...
```

(benchflow also picks up well-known credential keys from a `.env` file in the
current directory; exporting works from any directory.)

### Precedence

If multiple credentials are set, benchflow / the agent CLI uses provider-specific
credentials selected by the model prefix first, then the agent's native auth
precedence. For Claude, native auth is (high to low): cloud provider creds →
`ANTHROPIC_AUTH_TOKEN` → `ANTHROPIC_API_KEY` → `apiKeyHelper` →
`CLAUDE_CODE_OAUTH_TOKEN` → host subscription OAuth. To force a lower-priority
option, unset the higher one in your shell before running.

## Run your first eval

```bash
# Single task from a local directory
GEMINI_API_KEY=... bench eval run \
  --tasks-dir tasks/edit-pdf \
  --agent gemini \
  --model gemini-3.1-pro-preview \
  --sandbox docker

# Single task with mounted skills
GEMINI_API_KEY=... bench eval run \
  --tasks-dir tasks/edit-pdf \
  --agent gemini \
  --model gemini-3.1-pro-preview \
  --sandbox daytona \
  --skill-mode with-skill \
  --skills-dir tasks/edit-pdf/environment/skills \
  --agent-env BENCHFLOW_SKILL_NUDGE=name

# A whole batch from YAML config
bench eval run --config benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml

# Batch over a local tasks directory with concurrency
GEMINI_API_KEY=... bench eval run \
    --tasks-dir tasks \
    --agent gemini --model gemini-3.1-pro-preview --sandbox daytona --concurrency 32

# List the registered agents
bench agent list
```

`bench eval run` is the primary command for running evaluations — it works for
single tasks, batch runs, and remote repos. Use `--tasks-dir <dir>` for a local
directory or `--config <config.yaml>` for a YAML config.

You can also fetch tasks straight from a remote repo with
`--source-repo <org/repo> --source-path <subpath>`, but note that this clones
the full repository (`git clone --depth 1` into `.cache/datasets/<org>/<repo>/`
under the enclosing git repo root, or the current directory when you run
outside one) — large for big task repos. To download only the task you need,
use a sparse checkout and point `--tasks-dir` at it:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/benchflow-ai/skillsbench
cd skillsbench && git sparse-checkout set tasks/edit-pdf
bench eval run --tasks-dir tasks/edit-pdf --agent gemini --model gemini-3.1-pro-preview
```

When you mount skills, use `BENCHFLOW_SKILL_NUDGE=name` as the default docs
option. See [Architecture: skill loading](./architecture.md#skill-loading) for
how mounted skills reach the agent and how `name`, `description`, and `full`
differ.

### Where results land

Each run writes under `--jobs-dir` (default `jobs/`):

```
<jobs-dir>/
  summary.json                      # copy of the latest job summary (overwritten by the next run)
  <YYYY-MM-DD__HH-MM-SS>/           # job directory, named by start time
    summary.json                    # job-level aggregate
    <task>__<hash8>/                # one rollout: task name + 8-char id
      result.json                   # rollout summary: rewards, errors, token usage/cost
      rewards.jsonl                 # reward record for this rollout
      timing.json                   # per-phase timing breakdown
      prompts.json                  # prompts sent to the agent
      trajectory/
        acp_trajectory.jsonl        # full agent trace (ACP events)
        llm_trajectory.jsonl        # raw provider requests/responses (when the usage-tracking proxy captured exchanges)
      trainer/
        verifiers.jsonl             # trainer-ready scored trajectory (Verifiers/ORS record)
        atif.json                   # ATIF trajectory record (omitted if the trajectory is empty)
        adp.jsonl                   # ADP trajectory record
      verifier/
        ctrf.json                   # CTRF test report (when test.sh emits one)
        reward.txt                  # raw verifier reward (0.0-1.0)
        test-stdout.txt             # verifier stdout
```

### Reading results

Exit code 0 means the pipeline completed — it is not a pass/fail signal. A
rollout whose reward is below the pass threshold still exits 0 and prints
`[FAIL]` with `Score: 0/1`: `Score` is pass-threshold aggregation (a task
counts as passed only at reward 1.0), while `reward` — in `result.json` and
`verifier/reward.txt` — is the raw verifier value. Config errors (unknown
agents, missing credentials) exit 1, and so do runs with agent or verifier
errors. CLI usage errors (bad flags) exit 2.

The Docker sandbox needs the Docker daemon running. There is no up-front
check — if the daemon is down the run fails partway through rather than at
startup, so start Docker before `bench eval run --sandbox docker`.

## Run from Python

The CLI is a thin shim over the Python API. For programmatic use:

```python
import benchflow as bf
from benchflow import RolloutConfig, Scene
from benchflow._utils.benchmark_repos import resolve_source

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
| Understand how BenchFlow runs *any* benchmark (the three-layer model) | [Run any benchmark](./running-any-benchmark.md) |
| Understand the model — Rollout, Scene, Role, Verifier | [Concepts](./concepts.md) |
| Author a task | [Task authoring](./task-authoring.md) |
| Run multi-agent patterns (coder/reviewer, simulated user, BYOS) | [Use cases](./use-cases.md) |
| Run multi-round single-agent (progressive disclosure) | [Progressive disclosure](./progressive-disclosure.md) |
| Evaluate skills, not tasks | [Skill eval](./skill-eval.md) |
| Understand the security model | [Sandbox hardening](./sandbox-hardening.md) |
| CLI flags + commands | [CLI reference](./reference/cli.md) |
| Python API surface | [Python API reference](./reference/python-api.md) |
