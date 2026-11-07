# Trajectory and rollout formats

A trial (a rollout, in training terms) is written in several files, each for a different reader: one for debugging, one for exact model calls, and three for trainers and other tools.

## The trajectory files

From an OpenCode trial on DeepSeek, whose trajectory holds 1 user message, 14 thoughts, 14 tool calls and 1 final message:

| File | Holds | Read it for |
|---|---|---|
| `trajectory/acp_trajectory.jsonl` | what the agent reported, one event per line: `user_message`, `agent_thought`, `tool_call`, `agent_message` | debugging, and auditing what the agent did |
| `trajectory/llm_trajectory.jsonl` | each model request and response as BenchFlow's proxy saw them, with timing and token usage | exactly what the model saw and returned; training data |
| `trainer/atif.json` | the trial in ATIF, the Agent Trajectory Interchange Format, version `ATIF-v1.7` | other tools that read ATIF |
| `trainer/verifiers.jsonl` | one row: `prompt`, `completion`, `reward`, `metrics`, `is_truncated` | trainers that read verifiers-style rows |
| `trainer/adp.jsonl` | the same trial as ADP (Agent Data Protocol) records | tools that read ADP |

The job folder adds `results.jsonl`, one line per trial with its reward, timing and token usage. When an agent's model calls don't go through the proxy, as with a subscription login, `llm_trajectory.jsonl` can't record them, and `acp_trajectory.jsonl` is the full record.

## ATIF

`trainer/atif.json` has a header and a list of steps:

```text
atif keys: ['agent', 'final_metrics', 'schema_version', 'session_id', 'steps']
schema_version: ATIF-v1.7
steps: 17, each with ['message', 'source', 'step_id']
```

BenchFlow writes version 1.7 of the format.

## The rollout row

`bench train convert` turns trials into rows for a trainer, in `prime-sft` (the default) or `trl-sft` layout: the conversation from `llm_trajectory.jsonl` with its tool calls, the reward, and the token usage. [Export training data](./export.md) shows a row and the options. While a job runs, `bench train stream` emits one line per finished trial, in the rollout-stream format ([Connect a trainer](./trainer.md)).

## Schema versions

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
