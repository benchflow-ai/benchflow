# Streaming rollouts to a trainer

`bench train stream <job>` and `bf.stream_rollouts(job_dir)` pass each finished rollout (one trial attempt) to your trainer while the job is still running. A record holds the reward and the group. When BenchFlow's gateway captured them, it also holds the token ids and logprobs of every model call.

BenchFlow does not train, sync weights or host the policy. You run your own OpenAI-compatible server (vLLM or SGLang), run tasks against it, and read the stream.

To look at a job's trajectories in a browser instead, use `bench view <job folder>`.

## Quick start

Terminal 1, the job (the policy server is yours; see [Self-hosted policy servers](#self-hosted-policy-servers)):

```bash
bench run tasks/ -a claude-agent-acp -m vllm/my-policy --sandbox docker --jobs-dir jobs/ \
  --env BENCHFLOW_PROVIDER_BASE_URL=http://my-gpu-host:8000/v1 \
  --env BENCHFLOW_PROVIDER_API_KEY=... \
  --env BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1
```

Terminal 2, the trainer's feed. The job folder may not exist yet; the stream waits for it.

```bash
bench train stream jobs/ > rollouts.jsonl
```

Python, in another process:

```python
import benchflow as bf

for rollout in bf.stream_rollouts("jobs/", group_size=8):
    for seq in rollout.sequences:  # empty unless training-grade
        print(rollout.task, rollout.advantage, len(seq["completion_ids"]))
```

Replace the `print` with your trainer's call. Each sequence carries `prompt_ids`, `completion_ids`, `completion_mask` and `completion_logprobs` ([fields](#sequences)).

Or run the job and read it in one process:

```python
import asyncio

import benchflow as bf


async def main() -> None:
    config = bf.EvaluationConfig(
        agent="claude-agent-acp",
        model="vllm/my-policy",
        agent_env={
            "BENCHFLOW_PROVIDER_BASE_URL": "http://my-gpu-host:8000/v1",
            "BENCHFLOW_PROVIDER_API_KEY": "...",
            "BENCHFLOW_CAPTURE_TOKEN_LOGPROBS": "1",
        },
    )
    evaluation = bf.Evaluation("tasks/", "jobs/", config=config)
    job = asyncio.create_task(evaluation.run())
    async for rollout in bf.astream_rollouts(evaluation.job_dir, group_size=8):
        print(rollout.task, rollout.reward, rollout.advantage)
    await job


asyncio.run(main())
```

`Evaluation.stream()` yields `(task name, RolloutResult)` pairs in the same process. Use the rollout stream when you need token data, groups, or a separate process.

## Behaviour

- **Order.** A rollout is emitted once, when its `result.json` exists. BenchFlow writes `result.json` atomically and last, after the trajectory and the gateway log, so a half-written rollout is never read. Rollouts already finished come first, then new ones in the order they finish.
- **End.** The stream ends when the job finishes: `.evaluation.lock` is gone and `summary.json` exists. A job stopped by Ctrl-C or SIGTERM also writes `summary.json`, a partial one with `"stopped": true`; the stream reads that as stopped, not finished (exit 1). Pass the job folder, or a `--jobs-dir` that holds one job. `bench run` and `bf.Evaluation` write both. `bf.run` and `bf.run_batch` write neither, so for their folders use `--no-follow` or `--timeout`.
- **One scan.** `--no-follow` (`follow=False`) scans once and stops.
- **Unscored.** A rollout with a verifier error or a crash has `reward: null`, never 0. It never enters a baseline.

### Groups

`--group-size N` (`group_size=N`) holds rollouts until N of a group have finished. Then it emits them together, each with a GRPO advantage.

| Situation | `advantage` | `group_complete` |
| --- | --- | --- |
| At least 2 scored rollouts in the group | `(reward − mean) / (std + 1e-4)` over the scored rollouts (sample std) | `true` |
| Every scored reward is equal | `0.0` | `true` |
| Fewer than 2 scored rollouts | `null` | `true` |
| The rollout itself is unscored | `null` | `true` |
| Group still short when the stream ends | `null` | `false` |
| No `--group-size` | `null` | `null` |

`--group-by` sets the group key. The default is `task,agent,model,job`, the same as `bench train convert`. A group never spans jobs unless you drop `job`. The keys are explained in [Training signal](./training-signal.md#group-advantages).

### Retries

If the job will retry a rollout, the stream waits for the outcome:

- When a retry replaces it, the old rollout comes out with `retried: true`, no advantage and no group.
- If the job ends first, the rollout was the trial's last attempt and counts as final.

A rollout that replaces an earlier attempt names it in `replaces` (its `rollout_path`), with or without `--group-size`. A trainer that computes its own baseline can drop the replaced one.

### Exit codes

| Situation | CLI | Python |
| --- | --- | --- |
| Job finished | exit 0 after the last rollout | iterator ends |
| Job's process died first (lock names a dead process on this host) | every finished rollout, then exit 1 (run `bench eval resume <job>`) | `JobProcessGone` after the last rollout |
| Job was stopped (Ctrl-C or SIGTERM: no lock, and `summary.json` has `"stopped": true`) | every finished rollout, then exit 1 (run `bench eval resume <job>`) | `JobProcessGone` after the last rollout |
| `--timeout S` reached | every finished rollout, then exit 3 | `StreamTimeout` |
| Folder missing with `--no-follow`; bad `--format`, `--group-size` or `--group-by` | exit 2 | `JobNotFound` / `ValueError` |
| Unreadable `result.json` | skipped, one warning on stderr | skipped, `on_warning` callback |

Records go to stdout, one JSON object per line. Status and warnings go to stderr.

## The record: `benchflow.rollout-stream.v1`

Schema: [`benchflow-rollout-stream.v1.schema.json`](./schemas/benchflow-rollout-stream.v1.schema.json).

Here `schema_version` is the string `benchflow.rollout-stream.v1`. Unlike the [JSON export](./json-export.md) documents, there is no `kind` or `schema_minor`. New optional fields keep v1, and the schema allows extra fields. Removing or changing a field makes v2.

An example record, from `claude-agent-acp` on a vLLM-shaped server (ids synthetic, token lists shortened):

```json
{
  "schema_version": "benchflow.rollout-stream.v1",
  "job": "2026-01-01__12-00-00",
  "rollout": "hello-pass__00000001",
  "rollout_path": "hello-pass__00000001",
  "task": "hello-pass",
  "agent": "claude-agent-acp",
  "model": "vllm/fake-policy",
  "group_id": "task=hello-pass|agent=claude-agent-acp|model=vllm/fake-policy|job=2026-01-01__12-00-00",
  "group": {"task": "hello-pass", "agent": "claude-agent-acp", "model": "vllm/fake-policy", "job": "2026-01-01__12-00-00"},
  "reward": 1.0,
  "scored": true,
  "outcome": "passed",
  "error": null,
  "finished_at": "2026-01-01 12:02:00.000000",
  "token_capture": {
    "status": "captured", "training_grade": true, "path": "vllm",
    "calls": 3, "captured_calls": 3, "complete_calls": 3, "unavailable": {},
    "prefix": {"pairs": 1, "extends_previous_call": 1, "breaks": []},
    "threads": [
      {"thread": 0, "kind": "agent", "calls": [0, 1]},
      {"thread": 1, "kind": "helper", "calls": [2]}
    ]
  },
  "calls": [
    {"index": 0, "provider": "vllm", "prompt_token_ids": [60, 124, 116, "..."],
     "completions": [{"index": 0, "token_ids": [87, 114, 105, "..."], "logprobs": [-0.125, -0.25, -0.375, "..."]}],
     "unavailable": {}}
  ],
  "sequences": [
    {"thread": 0, "kind": "agent", "calls": [0, 1],
     "prompt_ids": ["<prompt ids>"], "completion_ids": ["<completion ids>"],
     "completion_mask": ["1 per sampled token, 0 per tool-result and next-turn token"],
     "completion_logprobs": ["one float per completion id, 0.0 where the mask is 0"]},
    {"thread": 1, "kind": "helper", "calls": [2], "...": "..."}
  ],
  "advantage": null,
  "group_complete": null,
  "replaces": null,
  "retried": false
}
```

- `calls`: every captured model call, in the order of `trajectory/llm_trajectory.jsonl` (`index` is the line). Ids and logprobs are the server's own ([token capture](./token-capture.md)). Fields it could not return are listed in `unavailable` with a reason.
- `token_capture`: the `bench train token-coverage` summary of the rollout, plus `path` (the route of the captured calls: `vllm`, `sglang`, …). `status` is `captured`, `capture_off` (gateway used, capture not on) or `no_gateway_capture` (subscription login, oracle, or an agent the gateway can't route).
- `token_capture.threads`: the conversations in the rollout. When any call offers tools, calls with the same tool set form one `agent` thread. Each tool-less call (such as Claude Code's session-title prompt) is its own `helper` thread. A rollout with no tools at all is one `chat` thread. Token-in/token-out is checked per thread.

### Sequences

`sequences` holds one entry per thread. It is present only when the rollout is training-grade: every call complete, one logprob per sampled token, and no prefix break. Otherwise `token_capture.reason` says why; nothing is dropped silently.

| Field | Contents |
| --- | --- |
| `thread`, `kind`, `calls` | The thread this sequence merges, and the calls in it. |
| `prompt_ids` | The thread's first prompt. |
| `completion_ids` | Everything after it: each call's sampled tokens, plus the tokens the next prompt adds, such as tool results. |
| `completion_mask` | 1 for sampled tokens, 0 for added ones. |
| `completion_logprobs` | The logprob of each sampled token; 0.0 where the mask is 0. |

Only the first choice of each call is used. This is the shape multi-turn RL trainers take.

## Self-hosted policy servers

Use a `vllm/<model>` or `sglang/<model>` model, where `<model>` is the name the server serves. Pass three variables with `--env`:

| Variable | Value |
| --- | --- |
| `BENCHFLOW_PROVIDER_BASE_URL` | The server's `/v1` URL. |
| `BENCHFLOW_PROVIDER_API_KEY` | The server's `--api-key`, any string if it has none. |
| `BENCHFLOW_CAPTURE_TOKEN_LOGPROBS` | `1` |

The gateway asks vLLM for ids with `return_token_ids` and SGLang with its `sglext` flags, and every call asks for logprobs. `bench train token-coverage <job>` then says, per rollout, whether the run is training-grade and on which route.

What a run on a real policy server needs. This is not validated on a GPU; the contract is tested end to end against a local server that returns the same fields.

- **vLLM.** A release that has `return_token_ids` on the chat endpoint (vllm-project/vllm#22587) and the fix for missing ids on streamed tool calls (vllm-project/vllm#29074); agents stream and call tools. Serve with tool calling on, for example `vllm serve Qwen/Qwen3-8B --served-model-name my-policy --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 65536 --api-key $KEY`. Logprobs of sampled tokens need no flag. `BENCHFLOW_CAPTURE_TOP_LOGPROBS=k` needs `k <= --max-logprobs` (default 20).
- **SGLang.** A release with the response-level `sglext` ids (sgl-project/sglang#34488); for example `python -m sglang.launch_server --model-path Qwen/Qwen3-8B --served-model-name my-policy --tool-call-parser qwen25 --api-key $KEY`. BenchFlow sends the per-request flags; `--return-input-ids --return-output-ids` force them server-wide instead.
- **GPU.** Size it for the model and the agent's context, not for BenchFlow. Coding agents send long prompts, so plan for 32k-64k tokens per sequence. An 8B bf16 model fits one 40-80 GB GPU at that length; larger models need tensor parallelism.
- **Reachability.** On `--sandbox docker` the gateway runs on your machine, so `http://localhost:8000/v1` works. On Daytona and other sandbox-proxy providers the gateway runs inside the sandbox. There, the server must be reachable from the sandbox: a public or tunnelled URL, protected by `--api-key`.
- **Token-in/token-out on a real template.** A run is training-grade only if each prompt re-renders the history exactly as the model sampled it. Chat templates that rewrite past turns (for example by dropping earlier reasoning) break this, and so does an agent that compacts its context. `bench train token-coverage` lists the breaking calls. Such rollouts still stream with `calls` but no `sequences`.

## Next steps

- [Token ids and logprobs](./token-capture.md): what the gateway captures and how to check it.
- [Training signal](./training-signal.md): the same groups and advantages for `bench train convert`.
- [Train with TRL GRPO](../training/grpo-recipe.md): an end-to-end recipe.
- [When a run fails](../jobs/when-a-run-fails.md): why a rollout is unscored.
