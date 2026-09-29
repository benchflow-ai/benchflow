# Gateway token capture

BenchFlow's per-rollout LiteLLM gateway sees every model call an agent makes. With token capture on, it asks the model server for sampled-token logprobs and, where the server supports it, prompt and completion token ids, and stores them with each call in `trajectory/llm_trajectory.jsonl`. This works for any agent that talks to the gateway (Claude Code, Codex, OpenHands, OpenCode, ...), because it happens at the gateway, not in the agent.

Capture is opt-in. `bench train convert` does not read it yet; `bench train stream` ([rollout stream](rollout-stream.md)) hands it to a trainer per rollout while a job runs. `bench train token-coverage <job or rollout> [--json]` reports, per rollout, whether the capture is training-grade (below).

## Turning it on

| Variable | Values | Effect |
| --- | --- | --- |
| `BENCHFLOW_CAPTURE_TOKEN_LOGPROBS` | `1` | Enables capture. Pass it with `--agent-env`; it reaches the gateway process on host and sandbox proxies. Runs that skip the gateway (subscription/OAuth auth, agents that cannot be routed) record nothing and log a warning saying so. |
| `BENCHFLOW_CAPTURE_TOKEN_IDS` | `auto` (default), `1`, `0` | `auto` requests token ids on `vllm/...` routes (`return_token_ids`) and `sglang/...` routes (SGLang's `sglext` flags) only. `1` requests them on every OpenAI-compatible chat route, with vLLM's `return_token_ids` except on `sglang/...` routes; use it for other servers that implement the field. `0` never requests them. |
| `BENCHFLOW_CAPTURE_TOP_LOGPROBS` | integer | Also request that many alternatives per sampled token. Off by default. |

## What the gateway requests

| Call the agent makes | Upstream | What is added to the request |
| --- | --- | --- |
| `/v1/chat/completions` | any | `logprobs: true`; plus, when token ids are on, `extra_body.return_token_ids: true`, or on `sglang/...` routes `extra_body.return_input_ids_in_sglext: true` and `return_output_ids_in_sglext: true` |
| `/v1/messages` (Claude Code) | OpenAI-compatible chat backend (e.g. vLLM), reached through LiteLLM's Messages-to-chat bridge | same as chat |
| `/v1/messages` | Anthropic API or Bedrock | nothing; these APIs have no logprobs or token ids |
| `/v1/responses` (Codex) on a `-responses-bridge` alias | OpenAI-compatible chat backend | same as chat |
| `/v1/responses` | native Responses API (OpenAI, Azure OpenAI) | `include: ["message.output_text.logprobs"]` |

Streamed calls are covered too. LiteLLM's stream assembly drops per-chunk `logprobs`, `token_ids` and the first chunk's `prompt_token_ids`, so a startup patch in the gateway process (`src/benchflow/providers/litellm_token_capture_patch.py`) keeps them from the raw chunks.

## Provider coverage

| Provider / server | Prompt token ids | Completion token ids | Logprobs |
| --- | --- | --- | --- |
| vLLM (`vllm/...` routes), streamed or not | yes | yes | yes |
| SGLang (`sglang/...` routes), streamed or not | yes | yes | yes |
| OpenAI chat completions | `not_requested` | `not_requested` | yes, for models that support logprobs |
| OpenAI / Azure Responses API | `provider_api_unsupported` | `provider_api_unsupported` | yes, for models that support logprobs |
| Anthropic API, Bedrock (Claude) | `provider_api_unsupported` | `provider_api_unsupported` | `provider_api_unsupported` |
| Gemini through chat completions | `provider_api_unsupported` | `provider_api_unsupported` | requested; LiteLLM's Gemini mapping was not verified, so expect `not_returned` if it drops them |
| Gemini native `generateContent` (Gemini CLI) | `provider_api_unsupported` | `provider_api_unsupported` | `not_requested` |

SGLang returns ids in its response-level `sglext` extension (`input_ids`, and `output_ids` with one list per choice), in the body of a plain call and in one final `choices: []` chunk of a streamed call; the gateway reads them from there and drops that chunk before LiteLLM's Anthropic Messages stream adapter sees it (the adapter fails on a chunk without choices). SGLang's older `return_token_ids` is refused on streamed chat, so BenchFlow does not send it on `sglang/...` routes. Only the vLLM and SGLang field names were checked against their source (vLLM and SGLang `main`); both shapes are exercised end to end by `tests/test_gateway_token_capture.py` and `tests/test_gateway_sglang_capture.py` against a mock server (`tests/fixtures/mock_token_logprobs_server.py`, `--flavor vllm|sglang`), and through `bench eval run` by the scenario in `tests/e2e/`. No live vLLM, SGLang or OpenAI server was used.

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

- `wire` is the API the upstream call used: `openai-chat`, `openai-responses`, `anthropic-messages`, `gemini` or `other`.
- `requested` is what the gateway actually sent upstream, read from the recorded request.
- `prompt_token_ids` is the prompt as the server tokenized it, chat template included. Use these ids instead of re-tokenizing the messages; re-tokenizing can differ from what the model saw.
- `completions` has one entry per choice. `tokens`, `logprobs` and `top_logprobs` are parallel lists, one entry per sampled token. `top_logprobs` is null unless alternatives were requested.
- `unavailable` maps each of `prompt_token_ids`, `completion_token_ids` and `logprobs` that was not captured for every choice to `{"reason", "detail"}`. A field is either captured or listed here; it is never silently missing.

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
        continue
    prompt_ids = capture["prompt_token_ids"]
    for completion in capture["completions"]:
        sampled_ids = completion["token_ids"]
        sampled_logprobs = completion["logprobs"]
```

## Checking coverage

`bench train token-coverage <job or rollout>` (Python: `benchflow.trajectories.token_capture.summarize_rollout_token_capture`) prints one line per rollout, with the route its calls took (`[vllm]`, `[sglang]`, … as `path` in `--json`), and a total. A rollout is training-grade when every call has prompt token ids, sampled token ids and logprobs for every choice, and, within each conversation of the rollout, each call's prompt starts with the previous call's prompt followed by its sampled tokens (the token-in/token-out property an RL trainer needs; a break means the history was re-rendered or the context was compacted). Conversations (`threads` in `--json`, `benchflow.trajectories.token_capture.conversation_threads`): when any call offers tools, calls are grouped by their tool set (the agent loop; a subagent with other tools is its own), and each tool-less call, such as Claude Code's short helper prompts, is its own one-call conversation; a rollout without tool calls is one conversation. A rollout without `llm_trajectory.jsonl` is reported as `no_gateway_capture` with its `usage_tracking.endpoint_kind`: subscription-auth runs (`agent_native`) never pass through the gateway, so they carry no token data. Missing fields are counted by field and reason code. `bench train stream` ([rollout stream](rollout-stream.md)) carries the same summary with every rollout, plus one merged token sequence per conversation when the rollout is training-grade.

## Limits

- Each call stores its full prompt token ids, so the file grows with turns times context length, like the stored request messages already do.
- `llm_trajectory.jsonl` mixes root-agent and subagent calls; token capture does not label them.
- The OpenAI Responses API streams through a path this capture has not been tested on.
