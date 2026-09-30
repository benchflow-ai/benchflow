# Streaming rollouts to a trainer

`bench train stream <job>` and `bf.stream_rollouts(job_dir)` hand each finished rollout of a job to a trainer while the job is still running: its reward, its group, and, when BenchFlow's gateway captured them, the prompt token ids, sampled token ids and logprobs of every model call. This is the data side of online RL. BenchFlow does not train, sync weights or host a policy; you point it at your own OpenAI-compatible policy server (vLLM or SGLang), run tasks, and consume the stream.

## Quick start

Terminal 1, the job (the policy server is yours; see [Self-hosted policy servers](#self-hosted-policy-servers)):

```bash
bench eval run --tasks-dir tasks/ --agent claude-agent-acp \
  --model vllm/my-policy --sandbox docker --jobs-dir jobs/ \
  --agent-env BENCHFLOW_PROVIDER_BASE_URL=http://my-gpu-host:8000/v1 \
  --agent-env BENCHFLOW_PROVIDER_API_KEY=... \
  --agent-env BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1
```

Terminal 2, the trainer's feed (the job folder may not exist yet; the stream waits for it):

```bash
bench train stream jobs/ --format jsonl > rollouts.jsonl
```

Python, in another process:

```python
import benchflow as bf

for rollout in bf.stream_rollouts("jobs/", group_size=8):
    for seq in rollout.sequences:          # empty unless training-grade
        trainer.add(seq["prompt_ids"], seq["completion_ids"],
                    seq["completion_mask"], seq["completion_logprobs"],
                    advantage=rollout.advantage)
```

Or in one process, next to the job:

```python
evaluation = bf.Evaluation(tasks_dir, jobs_dir, config=config)
job = asyncio.create_task(evaluation.run())
async for rollout in bf.astream_rollouts(evaluation.job_dir):
    ...
await job
```

(`Evaluation.stream()` also yields each task's `RolloutResult` in process; the rollout stream adds token data and groups, and works from any process that can read the job folder.)

## Behaviour

- A rollout is emitted once, when its `result.json` exists. BenchFlow writes `result.json` atomically and last, after the trajectory and the gateway log, so a half-written rollout is never read. Rollouts already finished when the stream starts come first; then new ones in the order they finish.
- The stream ends when the job finishes: the job lock (`.evaluation.lock`) is gone and `summary.json` exists. The argument may be the job folder or a `--jobs-dir` holding one job. `bench eval run` and `bf.Evaluation` write both; `bf.run` and `bf.run_batch` write neither, so for their folders use `--no-follow` (`follow=False`) or `--timeout`.
- `--no-follow` (`follow=False`) scans once and stops.
- `--group-size N` (`group_size=N`) holds rollouts until N final rollouts of a group have finished, then emits them together with a GRPO advantage, `(reward - mean) / (std + 1e-4)` over the group's scored rollouts (sample std). The group is `--group-by` (default `task,agent,model,job`, same keys as `bench train convert`: a group never spans jobs unless `job` is left out). Groups still short when the stream ends are emitted with `advantage: null` and `group_complete: false`. Without `--group-size`, `advantage` and `group_complete` are null and the trainer computes its own baseline.
- Retries. In an Evaluation job, a rollout that the job's retry policy (`evaluation.json`) would retry is not final: with `--group-size` it counts toward no group until the job ends (it was then its trial's last attempt) or a retry replaces it, when it is emitted with `retried: true`, no advantage and no group. Every rollout that replaces an earlier attempt of its trial names it in `replaces` (its `rollout_path`), with or without `--group-size`, so a trainer computing its own baseline can drop the replaced one.
- Unscored rollouts (verifier error, crash) have `reward: null`, never 0, and never enter a baseline.

| Situation | CLI | Python |
| --- | --- | --- |
| Job finished | exit 0 after the last rollout | iterator ends |
| Job's process died first (lock names a dead process on this host) | every finished rollout, then exit 1 | `JobProcessGone` after the last rollout |
| `--timeout S` reached | every finished rollout, then exit 3 | `StreamTimeout` |
| Folder missing with `--no-follow`; bad `--format`, `--group-size` or `--group-by` | exit 2 | `JobNotFound` / `ValueError` |
| Unreadable `result.json` | skipped, one warning on stderr | skipped, `on_warning` callback |

Records go to stdout, one JSON object per line; status and warnings go to stderr.

## The record: `benchflow.rollout-stream.v1`

JSON Schema: [`schemas/benchflow-rollout-stream.v1.schema.json`](schemas/benchflow-rollout-stream.v1.schema.json) (generated from `benchflow.trajectories.rollout_stream.SCHEMA`). A new optional field keeps `benchflow.rollout-stream.v1`, and the schema is open (no `additionalProperties: false`), so a trainer validating records keeps working when one is added; removing or changing a field is a new version. An illustrative record in the shape the end-to-end scenario (`tests/e2e/test_rl_self_hosted.py`: `claude-agent-acp` on a vLLM-shaped policy server in a sandbox) produces, with synthetic ids and token lists shortened:

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

- `calls`: every captured model call, in the order of `trajectory/llm_trajectory.jsonl` (`index` is the line). Ids and logprobs are the server's own ([token capture](token-capture.md)); fields it could not return are listed in `unavailable` with a reason.
- `token_capture`: the `bench train token-coverage` summary of the rollout plus `path` (route provider of the captured calls: `vllm`, `sglang`, …). `status` is `captured`, `capture_off` (gateway used, capture not enabled) or `no_gateway_capture` (subscription auth, oracle, or an agent the gateway cannot route).
- `token_capture.threads`: the conversations inside the rollout. One agent run holds several: the tool loop, tool-less helper calls (Claude Code sends short separate prompts, e.g. for a session title), and subagents with their own tool set. When any call offers tools, calls are grouped by tool set (`agent`) and each tool-less call is its own `helper`; a rollout with no tool calls at all is one `chat` conversation. Token-in/token-out is checked within each conversation.
- `sequences`: present only when the rollout is training-grade (every call complete, with one logprob per sampled token id, and no prefix break); a rollout with a conversation that cannot be merged is not training-grade and says why in `token_capture.reason`, so a sequence is never dropped silently. One per conversation: `prompt_ids` is its first prompt; `completion_ids` is everything after it, the sampled tokens of each call (`completion_mask` 1, their logprobs) and the tokens the next prompt adds after them, such as tool results (mask 0, logprob 0.0). This is the shape multi-turn RL trainers use (prompt ids, completion ids, completion mask, completion logprobs); only the first choice of each call is merged.

## Self-hosted policy servers

Use a `vllm/<model>` or `sglang/<model>` model with `BENCHFLOW_PROVIDER_BASE_URL` (the server's `/v1` URL), `BENCHFLOW_PROVIDER_API_KEY` (its `--api-key`, any string if it has none) and `BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1`. The gateway asks vLLM for ids with `return_token_ids` and SGLang with its `sglext` flags, and every call asks for logprobs. `bench train token-coverage <job>` then says per rollout whether the run is training-grade and on which route.

What a run on a real policy server needs (not validated on a GPU; the contract is exercised end to end against a local server that returns the same fields):

- **vLLM.** A release that has `return_token_ids` on the chat endpoint (vllm-project/vllm#22587) and the fix for missing ids on streamed tool calls (#29074); agents stream and call tools. Serve with tool calling on, for example `vllm serve Qwen/Qwen3-8B --served-model-name my-policy --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 65536 --api-key $KEY`. Logprobs of sampled tokens need no flag; `BENCHFLOW_CAPTURE_TOP_LOGPROBS=k` needs `k <= --max-logprobs` (default 20).
- **SGLang.** A release with the response-level `sglext` ids (sgl-project/sglang#34488); for example `python -m sglang.launch_server --model-path Qwen/Qwen3-8B --served-model-name my-policy --tool-call-parser qwen25 --api-key $KEY`. The per-request flags are sent by BenchFlow; `--return-input-ids --return-output-ids` force them server-wide instead.
- **GPU.** Sized by the model and the agent's context, not by BenchFlow. Coding agents send long prompts (Claude Code's first prompt is several thousand tokens before any tool output), so plan for 32k to 64k tokens of context per sequence. As an estimate: an 8B model in bf16 fits one 40 to 80 GB GPU with room for KV cache at that length (a 24 GB GPU needs fp8 weights or KV cache, or a shorter context); larger policies need tensor parallelism.
- **Reachability.** On `--sandbox docker` the gateway runs on your machine, so `http://localhost:8000/v1` works. On Daytona and other sandbox-proxy providers the gateway runs inside the sandbox, so the server must be reachable from there (a public or tunnelled URL, protected by `--api-key`).
- **Token-in/token-out on a real template.** A run is training-grade only if each prompt re-renders the history exactly as the model sampled it. Chat templates that rewrite past turns (for example dropping earlier reasoning) or an agent that compacts its context break this; `bench train token-coverage` lists the breaking calls, and such rollouts still stream with `calls` but no `sequences`.
