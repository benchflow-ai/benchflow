# Export training data

`bench train convert` turns finished trials into JSONL rows a trainer reads: the conversation with its tool calls, the reward, and the token use. Use it to build fine-tuning data from passing trials, or reward-labelled data for RL, from runs you already have.

## Make a job to export

Any finished job works. To make one, run an agent with `--jobs-dir`, so you know where the job lands. This is the quick start's task with OpenCode on DeepSeek, with `DEEPSEEK_API_KEY` set ([Agents and models](../evaluate/agents.md#log-in-or-set-a-key)):

```bash
bench run benchflow-ai/skillsbench/tasks/econ-detrending-correlation \
  -a opencode -m deepseek/deepseek-v4-flash --jobs-dir jobs/opencode
```

## Export a job

Convert the job folder your run printed on its `Artifacts:` line (ours was `jobs/opencode/2026-11-04__17-49-57`), then check the file:

```bash
bench train convert jobs/opencode/2026-11-04__17-49-57 --out train/opencode.jsonl
bench train validate train/opencode.jsonl
```

```text
Converted 1 row(s) from 1 rollout(s) -> train/opencode.jsonl
{"ok": true, "rows": 1, "rows_with_tool_calls": 1}
```

`--format` picks the layout: `prime-sft` (the default), `trl-sft`, or `branch-tree` for runs branched with [`bench branch`](../reference/commands.md#bench-branch). `--min-reward 1` keeps only trials whose reward is 1, which leaves out a rubric-reviewed trial that passed with a lower score. `--row-mode exchange` writes one row per model call instead of one per trial; for TRL that matters, since a row per trial puts the tool calls in the prompt and trains only the final reply ([Connect a trainer](./trainer.md#train-one-step-with-trl)). Secrets are redacted on the way out unless you pass `--no-redact`.

## What a row holds

One row from an OpenCode trial on DeepSeek, trimmed:

```json
{
  "task_name": "econ-detrending-correlation",
  "agent": "opencode",
  "model": "deepseek-v4-flash",
  "reward": 1.0,
  "score": 1.0,
  "source": "benchflow-llm-trajectory",
  "messages": [
    {"role": "user", "content": "In macroeconomics, understanding business cycle dynamics requires separating the trend com..."},
    {"role": "assistant", "content": "", "tool_calls": "[...]"},
    {"role": "tool", "content": "total 160\ndrwx------ 1 agent agent  4096 Nov  4 17:47 .\n..."},
    "..."
  ],
  "tool_defs": "[...]",
  "token_usage": {"n_tool_calls": 14, "n_input_tokens": 241358, "n_output_tokens": 6482, "total_tokens": 247840, "cost_usd": 0.00857596, "usage_source": "provider_response", "price_source": "litellm"}
}
```

The messages come from `trajectory/llm_trajectory.jsonl`, the model calls BenchFlow's proxy recorded, so they are what the model saw and returned. This row has 30 messages, and its `cost_usd` uses 0.9.1's placeholder price for DeepSeek ([Usage and telemetry](../scale/telemetry.md#cost-and-models-without-a-price)). A `trl-sft` row is laid out for TRL instead: `prompt`, `completion` and `tools` in place of `messages` and `tool_defs`, and the task under `task_id` rather than `task_name`. `--reward-vector` adds each rubric criterion or verifier metric to the row ([Commands](../reference/commands.md#bench-train-convert)), and `--group-advantage grpo` adds group-normalized advantages, which [BenchFlow as an RL environment](./environment.md#rewards-and-advantages) shows.

## Check rows before training

`bench train validate` reads the file back and counts the rows that hold tool calls. Run it every time: a row with no tool calls holds only a prompt and a final answer, which teaches a model nothing about doing the task.

> **Known issue in 0.9.1 and 0.10.0rc1:** Claude Code trials don't convert fully. The row keeps a token-count notice and the final answer, and loses every tool call.

The count catches it. Plain `validate` still says `"ok": true`; with `--require-tool-calls` it exits 1, so a script stops there:

```bash
bench train convert jobs/2026-11-04__16-43-38 --out train/rows.jsonl
bench train validate train/rows.jsonl
bench train validate train/rows.jsonl --require-tool-calls
```

```text
Converted 1 row(s) from 1 rollout(s) -> train/rows.jsonl
{"ok": true, "rows": 1, "rows_with_tool_calls": 0}
not all trainer rows contain tool calls: 0 / 1
  next: `bench train validate --help` describes the files it reads
```

Until that's fixed, build training rows from other agents' trials, such as OpenCode's or Pi's, or read Claude Code's full trajectory from `trajectory/acp_trajectory.jsonl`.

`bench train token-coverage <job>` reports whether each model call carries token ids and logprobs, which on-policy RL needs; see [Token ids and logprobs](./trainer.md#token-ids-and-logprobs).

## Trajectory and rollout formats

A trial (a rollout, in training terms) is written in several files, each for a different reader: one for debugging, one for exact model calls, and three for trainers and other tools.

### The trajectory files

From an OpenCode trial on DeepSeek, whose trajectory holds 1 user message, 14 thoughts, 14 tool calls and 1 final message:

| File | Holds | Read it for |
|---|---|---|
| `trajectory/acp_trajectory.jsonl` | what the agent reported, one event per line: `user_message`, `agent_thought`, `tool_call`, `agent_message` | debugging, and auditing what the agent did |
| `trajectory/llm_trajectory.jsonl` | each model request and response as BenchFlow's proxy saw them, with timing and token usage | exactly what the model saw and returned; training data |
| `trainer/atif.json` | the trial in ATIF, the Agent Trajectory Interchange Format, version `ATIF-v1.7` | other tools that read ATIF |
| `trainer/verifiers.jsonl` | one row: `prompt`, `completion`, `reward`, `metrics`, `is_truncated` | trainers that read verifiers-style rows |
| `trainer/adp.jsonl` | the same trial as ADP (Agent Data Protocol) records | tools that read ADP |

The job folder adds `results.jsonl`, one line per trial with its reward, timing and token usage. When an agent's model calls don't go through the proxy, as with a subscription login, `llm_trajectory.jsonl` can't record them, and `acp_trajectory.jsonl` is the full record.

### ATIF

`trainer/atif.json` has a header and a list of steps:

```text
atif keys: ['agent', 'final_metrics', 'schema_version', 'session_id', 'steps']
schema_version: ATIF-v1.7
steps: 17, each with ['message', 'source', 'step_id']
```

BenchFlow writes version 1.7 of the format.

### The rollout row

`bench train convert` turns trials into rows for a trainer, in `prime-sft` (the default) or `trl-sft` layout: the conversation from `llm_trajectory.jsonl` with its tool calls, the reward, and the token usage. [Export training data](./export.md) shows a row and the options. While a job runs, `bench train stream` emits one line per finished trial, in the rollout-stream format ([Connect a trainer](./trainer.md)).

### Schema versions

Every JSON document BenchFlow writes for other programs carries a versioned schema, published in [`docs/reference/schemas/`](../reference/schemas/):

| Schema file | Describes |
|---|---|
| `benchflow-trial.v1.schema.json` | a trial's exported record |
| `benchflow-job.v2.schema.json` | a job's exported summary |
| `benchflow-run-summary.v2.schema.json` | the file `--summary-out` writes |
| `benchflow-comparison.v2.schema.json` | a comparison of two jobs |
| `benchflow-training-signal.v1.schema.json` | the reward and advantage fields of a training row |
| `benchflow-rollout-stream.v1.schema.json` | each line of `bench train stream` |

A new version gets a new file; old ones stay, so a reader built for one version keeps working. [Files BenchFlow reads and writes](../reference/files.md) lists every field.
