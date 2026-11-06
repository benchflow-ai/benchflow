# Token ids and logprobs for RL

Turn on token capture for a run, then check that the rollouts are complete enough to train on:

```bash
bench run tasks/ -a claude-agent-acp -m vllm/my-policy --sandbox docker --job capture-demo \
  --env BENCHFLOW_PROVIDER_BASE_URL=http://localhost:8000/v1 \
  --env BENCHFLOW_PROVIDER_API_KEY=<the server's --api-key, any string if it has none> \
  --env BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1
bench train token-coverage jobs/capture-demo
```

`vllm/my-policy` is a model your own vLLM server serves as `my-policy`. Setting up the server is covered in [Self-hosted policy servers](./rollout-stream.md#self-hosted-policy-servers).

Every model call an agent makes goes through BenchFlow's LiteLLM gateway, one per rollout. With capture on, the gateway asks the model server for the logprob of each sampled token. When the server supports it, it also asks for the prompt and completion token ids.

The gateway stores them with each call in `trajectory/llm_trajectory.jsonl`. Because this happens in the gateway, not in the agent, it works for any agent routed through it: Claude Code, Codex, OpenHands, OpenCode and the rest.

What reads the capture:

| Command | What it does with it |
| --- | --- |
| `bench train token-coverage` | Says, per rollout, whether the capture is training-grade ([Checking coverage](#checking-coverage)). |
| `bench train stream` | Hands each finished rollout, tokens included, to a trainer while the job runs ([rollout stream](./rollout-stream.md)). |
| `bench train convert` | Does not read token capture yet. |

## Turning it on

Pass each variable to `bench run` with `--env`.

| Variable | Values | Effect |
| --- | --- | --- |
| `BENCHFLOW_CAPTURE_TOKEN_LOGPROBS` | `1` (or `true`, `yes`, `on`) | Turns capture on. A run that skips the gateway (a subscription or OAuth login, an agent that can't be routed) records nothing and logs a warning. |
| `BENCHFLOW_CAPTURE_TOKEN_IDS` | `auto` (default), `1`, `0` | `auto`: ask for token ids on `vllm/...` and `sglang/...` routes only. `1`: ask on every OpenAI-compatible chat route, for other servers that implement vLLM's `return_token_ids`. `0`: never ask. |
| `BENCHFLOW_CAPTURE_TOP_LOGPROBS` | integer | Also ask for this many alternatives per sampled token. Off by default. |

## What the gateway requests

| Call the agent makes | Upstream | What is added to the request |
| --- | --- | --- |
| `/v1/chat/completions` | any | `logprobs: true`; plus, when token ids are on, `extra_body.return_token_ids: true`, or on `sglang/...` routes `extra_body.return_input_ids_in_sglext: true` and `return_output_ids_in_sglext: true` |
| `/v1/messages` (Claude Code) | OpenAI-compatible chat backend (e.g. vLLM), reached through LiteLLM's Messages-to-chat bridge | same as chat |
| `/v1/messages` | Anthropic API or Bedrock | nothing; these APIs have no logprobs or token ids |
| `/v1/responses` (Codex) on a `-responses-bridge` alias | OpenAI-compatible chat backend | same as chat |
| `/v1/responses` | native Responses API (OpenAI, Azure OpenAI) | `include: ["message.output_text.logprobs"]` |

Streamed calls are captured too. LiteLLM drops per-chunk logprobs and token ids when it assembles a stream, so the gateway patches LiteLLM at startup to keep them.

## Provider coverage

| Provider / server | Prompt token ids | Completion token ids | Logprobs |
| --- | --- | --- | --- |
| vLLM (`vllm/...` routes), streamed or not | yes | yes | yes |
| SGLang (`sglang/...` routes), streamed or not | yes | yes | yes |
| OpenAI chat completions | `not_requested` | `not_requested` | yes, for models that support logprobs |
| OpenAI / Azure Responses API | `provider_api_unsupported` | `provider_api_unsupported` | yes, for models that support logprobs |
| Anthropic API, Bedrock (Claude) | `provider_api_unsupported` | `provider_api_unsupported` | `provider_api_unsupported` |
| Gemini through chat completions | `provider_api_unsupported` | `provider_api_unsupported` | requested; untested, may come back as `not_returned` |
| Gemini native `generateContent` (Gemini CLI) | `provider_api_unsupported` | `provider_api_unsupported` | `not_requested` |

SGLang returns ids in its `sglext` response extension, not in each choice. BenchFlow reads them from there. It does not send SGLang's older `return_token_ids`, which SGLang refuses on streamed chat.

The vLLM and SGLang paths are tested against a mock server that returns the same fields, not yet against a live server.

## Schema `benchflow.token_capture.v1`

Each line of `llm_trajectory.jsonl` is one call. With capture on, `metadata.token_capture` holds:

```json
{
  "schema_version": "benchflow.token_capture.v1",
  "wire": "openai-chat",
  "provider": "vllm",
  "requested": {"logprobs": true, "top_logprobs": null, "token_ids": true},
  "prompt_token_ids": [151644, 872, 198],
  "completions": [
    {
      "index": 0,
      "token_ids": [9707, 0],
      "tokens": ["Hello", "!"],
      "logprobs": [-0.02, -0.4],
      "top_logprobs": null
    }
  ],
  "unavailable": {}
}
```

| Field | Meaning |
| --- | --- |
| `wire` | The API the upstream call used: `openai-chat`, `openai-responses`, `anthropic-messages`, `gemini` or `other`. |
| `requested` | What the gateway actually asked the server for. |
| `prompt_token_ids` | The prompt as the server tokenized it, chat template included. Train on these: re-tokenizing the messages can give different ids. |
| `completions` | One entry per choice. `token_ids`, `tokens`, `logprobs` and `top_logprobs` are parallel lists, one entry per sampled token. `top_logprobs` is null unless you asked for alternatives. |
| `unavailable` | Each of `prompt_token_ids`, `completion_token_ids` or `logprobs` that is missing for any choice, as `{"reason", "detail"}`. A field is either present or listed here. |

Reason codes:

| Reason | Meaning |
| --- | --- |
| `provider_api_unsupported` | The upstream API has no such output. |
| `not_requested` | Capture is on, but the gateway did not ask this route for the field. The detail says how to request it. |
| `not_returned` | Requested, but the response did not include it (the server ignored the parameter or the model does not support it). |
| `request_failed` | The call failed, so there is no response. |

Token ids are stored only in `token_capture`: they are removed from the raw `response.body` to avoid storing them twice. Provider logprobs stay in the raw body as before.

## Reading it

```python
import json

for line in open("trajectory/llm_trajectory.jsonl"):
    call = json.loads(line)
    capture = call["metadata"].get("token_capture")
    if not capture or capture["unavailable"]:
        continue  # skip calls with no capture or a missing field
    prompt_ids = capture["prompt_token_ids"]
    for completion in capture["completions"]:
        sampled_ids = completion["token_ids"]
        sampled_logprobs = completion["logprobs"]
        print(len(prompt_ids), len(sampled_ids), sum(sampled_logprobs))
```

## Checking coverage

```bash
bench train token-coverage jobs/<job>          # one line per rollout, then a total
bench train token-coverage jobs/<job> --json
```

The argument is a job folder or one rollout folder. Each line starts with `yes` or `no` (training-grade or not), then the rollout and the route its calls took (`[vllm]`, `[sglang]`; `path` in `--json`). Missing fields are counted by field and reason code.

A rollout is **training-grade** when both hold:

1. Every call has prompt token ids, sampled token ids, and one logprob per sampled token. A count mismatch is reported as `logprobs:length_mismatch`.
2. Within each conversation, each call's prompt is the previous call's prompt plus its sampled tokens. This is the token-in/token-out property an RL trainer needs. A break means the agent re-rendered its history or compacted its context.

**Conversations** (`threads` in `--json`) group the calls of one rollout:

- When any call offers tools, calls with the same tool set form one conversation. The main agent loop is one; a subagent with other tools is another.
- Each call without tools, such as Claude Code's short helper prompts, is its own conversation.
- A rollout with no tool calls at all is one conversation.

A rollout with no `llm_trajectory.jsonl` is reported as `no_gateway_capture`, with its `usage_tracking.endpoint_kind`. Subscription-login runs (`agent_native`) never pass through the gateway, so they have no token data.

From Python, call `benchflow.trajectories.token_capture.summarize_rollout_token_capture(rollout_dir)`. `bench train stream` attaches the same summary to every rollout. When the rollout is training-grade, it adds one merged token sequence per conversation.

## Limits

- Each call stores its full prompt token ids, so the file grows with turns × context length.
- Root-agent and subagent calls share one `llm_trajectory.jsonl` and are not labelled.
- Streamed OpenAI Responses API calls are untested.
- Token ids decode back to the prompt text. Treat `llm_trajectory.jsonl` with the same care as the prompts.

## Next steps

- [Stream rollouts to a trainer](./rollout-stream.md) while the job runs.
- [Add reward vectors and group advantages](./training-signal.md) to converted rows.
- [`bench train` commands](./cli.md#bench-train) in the CLI reference.
- [Train with TRL GRPO](../training/grpo-recipe.md), an end-to-end recipe.
