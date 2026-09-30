# Getting started

From install to your first evaluation, its results, and a task of your own. The first evaluation needs no model and no API key. The whole path takes about 15 minutes, most of it building sandbox images the first time.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/). The install command's `--python 3.12` lets uv provision a compatible Python.
- Docker, running. On Linux, also install the buildx plugin (`docker-buildx-plugin` in Docker's apt and dnf repositories): without it Compose uses the legacy builder, which cannot build Dockerfiles that use BuildKit features. Apple Container is also built in on supported Apple Silicon Macs, and Daytona, Modal and AgentCore run sandboxes in the cloud (install the matching extra below).
- git 2.26 or newer, so that `--source-repo` downloads only the task folder you name. Older git downloads the whole repository.
- For agent runs, a subscription login or an API key for at least one agent (see [Auth](#auth-oauth-long-lived-token-or-api-key)). The `oracle` and `nop` agents need neither.

## Install

Install or upgrade to the latest stable release from PyPI with `uv`:

```bash
uv tool install --python 3.12 --upgrade benchflow
```

Confirm with `bench --version`. If `uv` reports `Executables already exist: bench, benchflow`, rerun with `uv tool install --python 3.12 --upgrade --force benchflow` to replace older non-`uv` entrypoints. See [Release channels](./release.md) for the full command matrix.

BenchFlow's CLI package requires Python 3.12 or newer. If `uv` is allowed to reuse Python 3.10 or 3.11, it may resolve an old `benchflow` package that does not provide the `bench` / `benchflow` executables and fail with `No executables are provided by package benchflow`. Keeping `--python 3.12` in the install command avoids that resolver fallback.

For optional sandbox integrations, include the extra in the tool install:

```bash
uv tool install --python 3.12 --upgrade 'benchflow[sandbox-daytona]'
uv tool install --python 3.12 --upgrade 'benchflow[sandbox-modal]'
uv tool install --python 3.12 --upgrade 'benchflow[sandbox-agentcore]'
```

This gives you the isolated `benchflow` (alias `bench`) CLI. To import the Python SDK from your own program, add `benchflow` to that program's environment (`uv add benchflow` or `pip install benchflow`; see [Run from Python](#run-from-python)). To install for editable development:

```bash
git clone https://github.com/benchflow-ai/benchflow
cd benchflow
uv sync --extra dev --locked
```

## Run everything: doctor, smoke, eval

Four commands take a fresh machine to a real evaluation. Run them in order and fix whatever one reports before moving to the next:

```bash
bench doctor        # is this machine ready? one PASS/WARN/FAIL line per check, each with a fix
bench eval run --source-repo benchflow-ai/skillsbench --source-path tasks/citation-check \
  --agent oracle --sandbox docker --jobs-dir jobs/oracle
bench eval smoke    # after you log in: the bundled hello-world task, once per agent you are logged in to
bench eval run --source-repo benchflow-ai/skillsbench --source-path tasks/citation-check \
  --agent claude --model claude-haiku-4-5-20251001 --sandbox docker --jobs-dir jobs/claude
```

`bench doctor` checks Python and uv, the Docker daemon and its buildx plugin (plus the Colima profile's state and memory when Docker runs on Colima), the Daytona SDK and key when you use Daytona, which credential each agent would use and when it expires, the agent versions the sandbox installs, and HTTPS access to the agent install and model endpoints. Credentials are shown by name, source and expiry, never by value. It exits 1 if anything required fails; until you log in to an agent that includes the `agent credentials` check, which you can ignore for oracle runs. Use `--sandbox daytona` to check the Daytona path, `--json` for scripts and `--offline` to skip the network probes.

If doctor reports no working agent credential, pick an option from [Auth](#auth-oauth-long-lived-token-or-api-key) below and run it again. A common case on macOS is a WARN for an expired `~/.claude/.credentials.json`: the Claude CLI keeps its live login in the Keychain, so export a `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token` instead.

The first `bench eval run` uses the `oracle` agent, which runs the task's own reference solution: it checks the sandbox, the task and its verifier without a model, and ends with `Score: 1/1`. [Run your first eval](#run-your-first-eval) explains what it does.

`bench eval smoke` reruns the checks, then runs the bundled hello-world task once per credentialed agent, one at a time on Docker (`--sandbox daytona` for Daytona), and prints agent, reward, time and trajectory path. It exits 1 unless every run scores 1.0. Choose agents and models with `--agent claude=claude-haiku-4-5-20251001 --agent codex`; a named agent runs even if doctor only warned about its credential. Logs, results and `smoke-summary.json` land under `jobs/smoke/<timestamp>/`. It runs model agents only; for the oracle, use `bench eval run --agent oracle` as above.

Once the smoke passes, every run below works the same way; see [CLI reference](./reference/cli.md#bench-doctor) for all checks and flags.

## Auth: OAuth, long-lived token, or API key

You don't need an API key if you're a Claude or ChatGPT subscriber. Three options, pick one per agent:

### Option 1 — Subscription OAuth from host CLI login

If you've logged into the agent's CLI on your host (`claude auth login` or `codex login`), benchflow picks up the credential file and copies it into the sandbox. No API key billing.

| Agent | How to log in on the host | What benchflow detects | Replaces env var |
|-------|---------------------------|------------------------|------------------|
| `claude-agent-acp` | `claude auth login` (Claude Code CLI) | `~/.claude/.credentials.json` | `ANTHROPIC_API_KEY` |
| `codex-acp` | `codex login` (Codex CLI) | `~/.codex/auth.json` | `OPENAI_API_KEY` |
| `gemini` | — (BenchFlow runs Gemini through its LiteLLM proxy, which needs an API key; a `~/.gemini/oauth_creds.json` login is not used, so export `GEMINI_API_KEY`) | — | `GEMINI_API_KEY` |
| `antigravity` | — (Google sign-in lives in the OS keyring and cannot be copied into a sandbox; use `GEMINI_API_KEY`) | — | `GEMINI_API_KEY` |

When benchflow finds the detect file, you'll see:

```
Using host subscription auth (no ANTHROPIC_API_KEY set)
```

On macOS, `claude auth login` keeps the live login in the Keychain and does not refresh `~/.claude/.credentials.json`, the file BenchFlow reads; use Option 2 there.

### Option 2 — Long-lived OAuth token (CI / headless)

For macOS, CI pipelines, scripts, or anywhere the host can't run an interactive browser login, generate a 1-year OAuth token with `claude setup-token` and export it:

```bash
claude setup-token            # walks you through browser auth, prints a token
export CLAUDE_CODE_OAUTH_TOKEN='<paste-token>'
```

benchflow auto-inherits `CLAUDE_CODE_OAUTH_TOKEN` from your shell into the sandbox; the Claude CLI inside reads it directly. Same auth precedence as plain `claude` ([Anthropic docs](https://code.claude.com/docs/en/authentication#authentication-precedence)): API keys override OAuth tokens, so unset `ANTHROPIC_API_KEY` if you want the token to win.

`claude setup-token` only authenticates Claude. For Codex on a ChatGPT login in CI, put the contents of a logged-in `~/.codex/auth.json` into `CODEX_AUTH_JSON` (a CI secret); benchflow hands it to Codex in the sandbox and `bench doctor` reports the login and plan it found. Codex can also use a provided subscription access token, such as `CODEX_ACCESS_TOKEN` from a host/orchestrator integration; benchflow passes it through to Codex without copying `~/.codex/auth.json`. Gemini does not have an equivalent today — use Option 3 (API key).

A CI job then needs only environment variables:

```bash
export CLAUDE_CODE_OAUTH_TOKEN="$CLAUDE_TOKEN_SECRET"      # claude-agent-acp
export CODEX_AUTH_JSON="$(cat codex-auth.json)"           # codex-acp, ChatGPT login
export DAYTONA_API_KEY="$DAYTONA_SECRET"                  # --sandbox daytona
bench eval run --tasks-dir tasks --agent claude-agent-acp --model claude-haiku-4-5-20251001 \
  --sandbox daytona --fresh --fail-under 0.8 --summary-out summary.json
```

`--fresh` starts a new job instead of resuming the last one in the jobs dir, `--fail-under`/`--fail-on` turn results into a failing exit code, and `--summary-out` writes a `benchflow.run-summary` JSON document; see [Exit codes](./reference/cli.md#exit-codes). For timestamped CI logs set `BENCHFLOW_LOG_FORMAT=time` (and `BENCHFLOW_LOG_LEVEL=DEBUG` for more detail).

### Option 3 — API key

Set the API-key env var directly. Works with every agent:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export CODEX_API_KEY=sk-...       # Codex alias for OPENAI_API_KEY
export GEMINI_API_KEY=...       # Gemini CLI and Antigravity CLI (agy)
export LLM_API_KEY=...           # OpenHands / LiteLLM-compatible providers
export AZURE_API_KEY=...
export AZURE_API_ENDPOINT='https://<resource>.openai.azure.com/'
```

benchflow auto-inherits well-known API key env vars from your shell into the sandbox. Provider-prefixed models can use credentials that differ from the agent's native default auth. For Azure Foundry, use models such as `azure-foundry-openai/gpt-5.5` or `azure-foundry-anthropic/claude-opus-4-5`; benchflow derives the Azure resource from `AZURE_API_ENDPOINT` and routes the selected agent through a generated LiteLLM gateway config.

Several providers with user-supplied endpoints — `glm`, `kimi`, `minimax`, `hunyuan`, and others — follow the `<PROVIDER>_API_KEY` + `<PROVIDER>_BASE_URL` convention; providers with fixed or default endpoints (such as `deepseek`, `zai`, or `openai`) need only the API key. Override the DeepSeek endpoint only when necessary:

```bash
export DEEPSEEK_API_KEY=...
export DEEPSEEK_BASE_URL=https://api.deepseek.com  # optional default override
```

These variables must be **exported** to reach the benchflow runtime — a plain shell assignment or a `source .env` without `export` stays local to your shell and never reaches the `bench` process. The portable pattern for a `.env` file:

```bash
set -a; source .env; set +a
bench eval run ...
```

(benchflow also picks up well-known credential keys from a `.env` file in the current directory; exporting works from any directory.)

### Precedence

If multiple credentials are set, benchflow / the agent CLI uses provider-specific credentials selected by the model prefix first, then the agent's native auth precedence. For Claude, native auth is (high to low): cloud provider creds → `ANTHROPIC_AUTH_TOKEN` → `ANTHROPIC_API_KEY` → `apiKeyHelper` → `CLAUDE_CODE_OAUTH_TOKEN` → host subscription OAuth. To force a lower-priority option, unset the higher one in your shell before running.

## Run your first eval

No key needed: the `oracle` agent runs the task's reference solution (`oracle/solve.sh`, or `solution/solve.sh` in older tasks) instead of a model.

```bash
bench eval run --source-repo benchflow-ai/skillsbench --source-path tasks/citation-check \
  --agent oracle --sandbox docker --jobs-dir jobs/oracle
```

It fetches the task, builds its Docker image (a couple of minutes the first time; later runs reuse the cached layers), runs the solution, runs the task's verifier, and prints the score:

```text
✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
Artifacts: jobs/oracle/2026-09-30__06-50-12
Summary:   jobs/oracle/2026-09-30__06-50-12/summary.json
View:      bench eval view jobs/oracle/2026-09-30__06-50-12
```

Then the same task with a model. With a Claude subscription token exported (see [Auth](#auth-oauth-long-lived-token-or-api-key)):

```bash
bench eval run --source-repo benchflow-ai/skillsbench --source-path tasks/citation-check \
  --agent claude --model claude-haiku-4-5-20251001 --sandbox docker --jobs-dir jobs/claude
```

`--agent claude` is short for `claude-agent-acp`, Claude Code over ACP; `bench agent list` shows every registered agent. The Claude Code that BenchFlow installs in the sandbox (`claude-agent-acp` 0.81.2 with Claude Code 2.1.280) accepts `claude-haiku-4-5-20251001` (or `claude-haiku-4-5`), `claude-sonnet-5` and `claude-opus-5-5`, and the aliases `haiku`, `sonnet` and `opus`. It refuses older ids such as `claude-sonnet-4-6`.

`--source-repo` and `--source-path` name a folder in a GitHub repository. BenchFlow downloads only that folder (a sparse clone; the citation-check task is a few megabytes) into `.cache/datasets/<org>/<repo>/`, under the enclosing git repository's root or the current directory outside one, and reuses it on later runs; another path from the same repository joins the same cache. `--source-ref` picks a branch, tag or commit. A folder that holds no BenchFlow task (a foreign benchmark that BenchFlow converts) gets the whole repository, as does a source with no path. To run several tasks, name the tasks folder, which downloads every task in it, and pick tasks with `--include` (or leave it out to run them all):

```bash
bench eval run --source-repo benchflow-ai/skillsbench --source-path tasks \
  --include citation-check --include 3d-scan-calc \
  --agent oracle --sandbox docker --concurrency 2 --jobs-dir jobs/two-tasks
```

Swap `--agent oracle` for an agent and a model to run them with a model.

`--tasks-dir <dir>` runs tasks from a local folder (one task, or a folder of tasks), and `--config <config.yaml>` runs a YAML run config. For task-local skills, add `--skill-mode with-skill`; see [Architecture: skill loading](./architecture.md#skill-loading) for how mounted skills reach the agent.

Each `--jobs-dir` holds the jobs of one experiment (the default is `jobs/`). Running the same command again resumes the latest job there: finished tasks are kept and not rerun, and the run says so. Add `--fresh` to start a new job, or use another `--jobs-dir`.

### Where results land

Each run writes under `--jobs-dir` (default `jobs/`):

```
<jobs-dir>/
  summary.json                      # copy of the latest job summary (overwritten by the next run)
  <YYYY-MM-DD__HH-MM-SS>/           # job directory, named by start time
    summary.json                    # job-level aggregate (pass counts plus mean_reward — mean over scored rollouts)
    evaluation.json                 # the job's configuration, used to resume it (agent_env key names only)
    results.jsonl                   # one row per rollout (also verifiers.jsonl, adp.jsonl)
    <task>__<hash8>/                # one rollout: task name + 8-char id
      result.json                   # rollout summary: rewards, errors, token usage/cost
      config.json                   # the rollout's resolved configuration (secret values left out)
      results.jsonl                 # Verifiers/Prime-RL shaped rollout row
      rewards.jsonl                 # reward record for this rollout
      timing.json                   # per-phase timing breakdown
      prompts.json                  # prompts sent to the agent
      agent/                        # the agent's own logs (install output, adapter log)
      trajectory/
        acp_trajectory.jsonl        # full agent trace (ACP events)
        llm_trajectory.jsonl        # raw provider requests/responses (when the usage-tracking proxy captured exchanges; not for subscription logins)
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

```bash
bench eval list jobs/            # one row per experiment folder, with its score
bench eval metrics jobs/claude   # passed, failed, errored, score, solve rate, pass@k, tool calls, duration
bench eval view jobs/claude      # every run in jobs/claude, in the browser
```

`bench eval metrics jobs/` counts every run under `jobs/` together, whatever agent ran it; point it at one experiment's folder to read that experiment.

To read a run the way a reviewer would, open it in the browser: `bench eval view` serves a local page (at the printed `http://localhost:8888` URL; `--port` picks another) with the full trajectory, the verifier output and whether the run completed and was scored. Point it at one job, a folder of jobs such as `jobs/`, or a single rollout. `bench eval run` prints this command at the end of each run. On a remote machine, forward the port first, for example `ssh -L 8888:localhost:8888 <host>`, then open the URL on your own machine.

Exit code 0 means the pipeline completed — it is not a pass/fail signal. A rollout whose reward is below the pass threshold still exits 0 and prints `[FAIL]` with `Score: 0/1`: `Score` is pass-threshold aggregation (a task counts as passed only at reward 1.0), while `reward` — in `result.json` and `verifier/reward.txt` — is the raw verifier value. Config errors (unknown agents, missing credentials) exit 1, and so do runs with agent or verifier errors. CLI usage errors (bad flags) exit 2.

The Docker sandbox needs the Docker daemon running. For `--sandbox docker`, `bench eval run` runs `bench doctor`'s Docker check before it creates a job and stops with the same fix line when the CLI or the daemon is missing; for `--sandbox daytona` it checks `DAYTONA_API_KEY` against the Daytona API the same way. Set `BENCHFLOW_SKIP_PREFLIGHT=1` to skip these checks.

To analyse finished runs in Python or a notebook (pass rates per agent and model, comparing two runs, exporting to pandas), see [Analysing runs](./analysing-runs.md).

To re-score a finished run after fixing a verifier, without running the agent again, run it with `--freeze-workspace` and see [Regrade stored runs](./regrade.md). To score a task in a verifier sandbox that shares nothing with the agent's, see [Separate verifier sandboxes](./separate-verifier.md).

## Write your own task

A task is a folder: `task.md` (the prompt, with YAML settings on top), `environment/Dockerfile` (the sandbox), `verifier/test.sh` (writes a reward from 0.0 to 1.0) and, for the oracle, `oracle/solve.sh`. Scaffold one, fill in its placeholders, check it, and run the oracle, then the empty `nop` agent, which should score 0:

```bash
bench tasks init my-task          # writes tasks/my-task/ with [REPLACE: ...] placeholders
bench tasks check tasks/my-task   # lists every placeholder still to fill
bench eval run --tasks-dir tasks/my-task --agent oracle --sandbox docker --jobs-dir jobs/my-task-oracle
bench eval run --tasks-dir tasks/my-task --agent nop --sandbox docker --jobs-dir jobs/my-task-nop
```

[Task authoring](./task-authoring.md) walks through the files with a complete example.

## Run from Python

The CLI is a thin shim over the Python API. Install `benchflow` into your program's environment (`uv add benchflow` or `pip install benchflow`), then:

```python
import benchflow as bf

task = bf.resolve_source("benchflow-ai/skillsbench", path="tasks/citation-check")
result = bf.run_sync(bf.RolloutConfig(task_path=task, agent="oracle", environment="docker"))
print(result.reward, result.passed)   # 1.0 True
print(result.rollout_dir)             # result.json, trajectory/, verifier/
```

`bf.run_sync` blocks until the rollout ends, also inside a running event loop such as Jupyter's; in async code use `result = await bf.arun(config)`. For a model, set `agent="claude-agent-acp", model="claude-haiku-4-5-20251001"`. Runnable scripts for single runs, batches and environment manifests are in [`docs/examples/python-sdk/`](./examples/python-sdk/).

`Rollout` is decomposable — invoke each lifecycle phase individually for custom flows. See [Concepts: rollout lifecycle](./concepts.md#rollout-lifecycle).

## What to read next

| If you want to… | Read |
|------------------|------|
| Upgrade from 0.7 | [What's new in 0.8](./whats-new-0.8.md) |
| Understand how BenchFlow runs *any* benchmark (the three-layer model) | [Run any benchmark](./running-any-benchmark.md) |
| Understand the model — Rollout, Scene, Role, Verifier | [Concepts](./concepts.md) |
| Author a task | [Task authoring](./task-authoring.md) |
| Run multi-agent patterns (coder/reviewer, simulated user, BYOS) | [Use cases](./use-cases.md) |
| Run multi-round single-agent (progressive disclosure) | [Progressive disclosure](./progressive-disclosure.md) |
| Evaluate skills, not tasks | [Skill eval](./skill-eval.md) |
| Understand the security model | [Sandbox hardening](./sandbox-hardening.md) |
| CLI flags + commands | [CLI reference](./reference/cli.md) |
| Python API surface | [Python API reference](./reference/python-api.md) |
