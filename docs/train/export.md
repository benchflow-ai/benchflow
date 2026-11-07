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

`--format` picks the layout: `prime-sft` (the default), `trl-sft`, or `branch-tree` for runs branched with [`bench branch`](../reference/commands.md#bench-branch). `--min-reward 1` keeps only passing trials. `--row-mode exchange` writes one row per model call instead of one per trial; for TRL that matters, since a row per trial puts the tool calls in the prompt and trains only the final reply ([Connect a trainer](./trainer.md#train-one-step-with-trl)). Secrets are redacted on the way out unless you pass `--no-redact`.

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

The messages come from `trajectory/llm_trajectory.jsonl`, the model calls BenchFlow's proxy recorded, so they are what the model saw and returned. This row has 30 messages. A `trl-sft` row is laid out for TRL instead: `prompt`, `completion` and `tools` in place of `messages` and `tool_defs`, and the task under `task_id` rather than `task_name`. `--reward-vector` adds each rubric criterion or verifier metric to the row ([Commands](../reference/commands.md#bench-train-convert)), and `--group-advantage grpo` adds group-normalized advantages, which [BenchFlow as an RL environment](./environment.md#rewards-and-advantages) shows.

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
