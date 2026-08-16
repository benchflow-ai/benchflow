# Start in 5 minutes

Launch a real, scored BenchFlow evaluation on your own machine in about five
minutes. This path uses a public SkillsBench task, your existing ChatGPT
subscription through Codex, and local Docker. You do not need Daytona or a
model API key.

The task is `3d-scan-calc`, not a toy prompt. The agent must parse a binary STL,
remove disconnected scan debris, recover a material ID stored in each triangle,
look up its density, calculate the largest component's mass, and write a JSON
report that an independent verifier checks to 0.1% accuracy.

> Five minutes is a quickstart target, not a timeout or performance guarantee.
> A first run may take longer while Docker pulls images or BenchFlow downloads
> the task and agent. A cold local smoke test for this guide completed in 278.7
> seconds with a 1/1 score; machine and network speeds vary.

## 0:00 — Check the two prerequisites

Start Docker and confirm that the daemon is reachable:

```bash
docker info >/dev/null
```

Install the [Codex CLI](https://github.com/openai/codex), sign in with your
ChatGPT subscription, and confirm the saved login:

```bash
codex login
codex login status
```

If you already use Codex, the login command normally opens no new setup work.

## 1:00 — Install BenchFlow

[`uv`](https://docs.astral.sh/uv/) can install BenchFlow and provision the
required Python 3.12 runtime in one command:

```bash
uv tool install --python 3.12 --upgrade benchflow
bench --version
```

If `uv` reports that the `bench` or `benchflow` executable already exists,
repeat the install with `--force` to replace the stale entrypoint.

## 2:00 — Run the real task locally

Copy this command as-is:

```bash
bench eval run \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks/3d-scan-calc \
  --agent codex \
  --model gpt-5.5 \
  --sandbox docker \
  --concurrency 1 \
  --jobs-dir jobs/first-local-run
```

BenchFlow now performs the entire evaluation lifecycle:

1. fetches the real task package and builds its local Docker image;
2. makes your saved Codex login available to the agent in the sandbox;
3. lets the agent inspect the STL and create `mass_report.json`;
4. runs the task's verifier outside the agent's control; and
5. saves the score, timings, token usage, and full ACP trajectory.

The default skill mode is `no-skill`, so this first run measures the base agent.
You can compare the task's bundled mesh-analysis skill later with
`--skill-mode with-skill`.

## Read the result

A completed run ends with a summary like this:

```text
✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
Artifacts: jobs/first-local-run/<timestamp>
```

The benchmarked agent is not guaranteed to pass. A 0/1 score with no execution
error still means BenchFlow ran the agent and verifier successfully; it means
the agent's answer did not satisfy the task.

Summarize the saved run from the CLI:

```bash
bench eval list jobs/first-local-run
bench eval metrics jobs/first-local-run
```

The important files are:

```text
jobs/first-local-run/<timestamp>/
  summary.json
  3d-scan-calc__<hash>/
    result.json
    timing.json
    prompts.json
    trajectory/acp_trajectory.jsonl
    verifier/reward.txt
    verifier/test-stdout.txt
```

- `summary.json` gives the job-level pass rate, elapsed time, token totals, and
  telemetry coverage.
- `result.json` is the quickest per-task record of reward, errors, tool calls,
  and token usage.
- `trajectory/acp_trajectory.jsonl` records the agent and tool interaction.
- `verifier/reward.txt` and `test-stdout.txt` explain the score.

Some agent/provider combinations also write
`trajectory/llm_trajectory.jsonl`; subscription-backed ACP agents can report
usage without producing that optional provider-level trace.

Use a different `--jobs-dir` for an independent second trial. Reusing
`jobs/first-local-run` intentionally resumes the existing run and skips work
that is already complete.

## Use Claude or Gemini instead

Keep the task, Docker, and output flags unchanged, and replace the agent/model
pair after signing in:

| Login | Flags |
|---|---|
| Claude Code | `--agent claude --model claude-sonnet-4-6` |
| Gemini CLI | `--agent gemini --model gemini-3.1-pro-preview` |

For API keys, CI credentials, and provider-hosted models, see
[Authentication](./authentication.md). For batches and pinned dataset runs,
continue to [Running evaluations](./running-evaluations.md).

## Reproduce the documentation smoke test from source

The repository keeps a versioned copy of this real task so maintainers can test
the guide without depending on a fresh remote clone:

```bash
uv sync --extra dev --locked
uv run bench eval run \
  --tasks-dir docs/examples/task-md/real-skillsbench/3d-scan-calc \
  --agent codex \
  --model gpt-5.5 \
  --sandbox docker \
  --concurrency 1 \
  --jobs-dir jobs/docs-start-in-5-minutes
```

The documented smoke run used local Docker, made nine tool calls, received a
1.0 verifier reward, and recorded complete timing and token metadata.
