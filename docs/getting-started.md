# Getting started

Run one scored BenchFlow evaluation on your own machine. This path uses Docker
and an existing Codex, Claude, or Gemini login; it does not require Daytona or
another cloud sandbox account.

Want the shortest copy-paste path? [Start a real local task in five
minutes](./start-in-5-minutes.md). This page explains each part of that run and
the available alternatives.

## What you need

- [`uv`](https://docs.astral.sh/uv/) — the install command below can provision
  the required Python 3.12 runtime for you
- Docker Desktop or another working Docker daemon
- One supported agent login or API key

Start Docker, then confirm it is reachable:

```bash
docker info >/dev/null
```

If this command fails, start Docker before continuing. BenchFlow builds a task
image and runs the agent inside it; the agent itself does not run directly on
your host.

## 1. Install BenchFlow

```bash
uv tool install --python 3.12 --upgrade benchflow
bench --version
```

`benchflow` and `bench` are aliases for the same CLI. If `uv` reports
`Executables already exist: bench, benchflow`, repeat the install with
`--force` to replace an older entrypoint.

Working from a source checkout instead? Use the repository environment:

```bash
git clone https://github.com/benchflow-ai/benchflow
cd benchflow
uv sync --extra dev --locked
uv run bench --version
```

## 2. Sign in to one agent

An existing subscription login is enough; an API key is not required for the
Codex or Claude examples.

```bash
# ChatGPT subscription through Codex CLI
codex login

# Or Claude subscription through Claude Code
claude auth login

# Or Gemini CLI's interactive login
gemini
```

BenchFlow detects the saved host credential and makes it available inside the
sandbox. If you prefer API keys, CI tokens, or a provider-hosted model, see
[Authentication](./authentication.md).

## 3. Run one local evaluation

This example downloads one public SkillsBench task, runs Codex inside a local
Docker sandbox, executes the task's verifier, and saves the full trajectory:

```bash
bench eval run \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks/3d-scan-calc \
  --agent codex \
  --model gpt-5.5 \
  --sandbox docker \
  --concurrency 1
```

Use one of these agent/model pairs if you signed in somewhere else:

| Host login | Replace the two flags with |
|---|---|
| Claude Code | `--agent claude --model claude-sonnet-4-6` |
| Gemini CLI | `--agent gemini --model gemini-3.1-pro-preview` |

The first run may take several minutes while Docker downloads and builds the
task image. The benchmarked agent may pass or fail the task; either outcome is
a valid completed evaluation.

## 4. Read the result

The console ends with `[PASS]`, `[FAIL]`, or an execution error. `[FAIL]` means
the agent completed but did not reach the verifier's pass threshold; it does
not mean BenchFlow itself failed.

Results are written under `jobs/`:

```text
jobs/
  <timestamp>/
    summary.json
    <task>__<hash>/
      result.json
      timing.json
      prompts.json
      trajectory/
        acp_trajectory.jsonl
        llm_trajectory.jsonl  # optional provider-level trace
      verifier/
        reward.txt
        test-stdout.txt
```

Use the CLI to summarize them:

```bash
bench eval list jobs/
bench eval metrics jobs/
```

`result.json` is the quickest place to check the raw reward, error status,
tool-call count, and token usage. `trajectory/acp_trajectory.jsonl` contains
the agent/tool interaction trace.

## Run your own local task

Point `--tasks-dir` at either one task package or a directory of task packages:

```bash
bench eval run \
  --tasks-dir tasks/my-task \
  --agent codex \
  --model gpt-5.5 \
  --sandbox docker
```

Docker is BenchFlow's default, so `--sandbox docker` may be omitted. Keeping it
in your first commands makes the execution location explicit.

## Where to go next

| Goal | Read |
|---|---|
| Use a different login, API key, or provider | [Authentication](./authentication.md) |
| Run local batches, YAML configs, or skill comparisons | [Running evaluations](./running-evaluations.md) |
| Decide between Docker, Apple Container, Daytona, Modal, and AgentCore | [Sandboxes](./sandboxes.md) |
| Understand task, agent, rollout, and verifier terminology | [Concepts](./concepts.md) |
| Create a task | [Task authoring](./task-authoring.md) |
| Look up every flag | [CLI reference](./reference/cli.md) |

For local development and small evaluation sets, you can stop here: no
Daytona setup is necessary.
