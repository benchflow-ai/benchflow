# Usage and telemetry

For every trial, BenchFlow records the tokens and cost the agent used, how long each step took, and, for each model call, what was sent and what came back. It reads these numbers at the model boundary, not from the agent's own telemetry, and records where each number came from.

## What a trial records

`result.json` sums the trial's usage in its `agent_result` block, with where each number came from. From the quick start's Claude Code trial on MiMo:

```json
{
  "n_input_tokens": 499812,
  "n_output_tokens": 11934,
  "n_cache_read_tokens": 453760,
  "total_tokens": 511746,
  "cost_usd": null,
  "usage_source": "provider_response",
  "price_source": null
}
```

and from an OpenCode trial on DeepSeek, a model BenchFlow has a price for:

```json
{"total_tokens": 247840, "cost_usd": 0.00857596, "usage_source": "provider_response", "price_source": "litellm"}
```

`timing.json` has seconds per step, including how much of the agent's turn was spent waiting for the model (`provider_wait`) and how much doing anything else (`agent_without_provider_wait`); [How BenchFlow works](../start/how-it-works.md#one-rollout-step-by-step) shows a whole one.

Each model call is a line in `trajectory/llm_trajectory.jsonl`: the request as sent, the response, its HTTP status and how long it took. The first call of the same Claude Code trial:

```text
path: /v1/messages | duration_ms: 8025.3 | status: 200
request: model "mimo-v2.6-flash", reasoning_effort, stream true, tools [...]
metadata: {"request_model": "mimo-v2.6-flash", "provider_model": "mimo-v2.6-flash", "call_type": "anthropic_messages", "call_purpose": "agent"}
usage: {"completion_tokens": 169, "prompt_tokens": 15230, "total_tokens": 15399, "reasoning_tokens": 13, "cached_tokens": 15168}
```

That record is what to check when a setting matters: it shows the effort, sampling and model each request carried, not just what you asked for.

## Where each number comes from

`usage_source` says how the trial's tokens were counted:

| `usage_source` | Means |
|---|---|
| `provider_response` | read from each model response, as the calls passed through BenchFlow's model proxy |
| `agent_native_acp` | reported by the agent itself over ACP, as with a subscription login, which the proxy doesn't route |
| `unavailable` | not recorded; treat token and cost figures for the trial as missing, not zero |

`price_source` says where the price came from: `litellm`, the price table bundled with BenchFlow's model proxy; `agent_session_log`, an estimate from the agent's own session log, as for Claude Code on a subscription login; or `null` when there's no price for the model. The proxy is always used for agents it can route, so usage and `llm_trajectory.jsonl` are always captured for them; `--usage-tracking required` additionally makes trusted usage (`provider_response` or `agent_native_acp`) a requirement of the run.

## OpenTelemetry

BenchFlow 0.9.1 and 0.10.0rc1 neither send nor receive OpenTelemetry. Usage, cost and timing come from the model proxy and the trial's own records, so they don't depend on whether an agent exports telemetry of its own. The gap is a subscription login, which bypasses the proxy: then only what the agent reports over ACP is recorded (`agent_native_acp`), or nothing (`unavailable`).

An agent can still send its own telemetry to a receiver you run. Claude Code's is off until you turn it on with its variables, which `--agent-env` passes into the sandbox. The receiver must listen on an address the sandbox can reach: for local Docker on Linux, Docker's bridge, which the first line below reads. [`receive.py`](../examples/otel/receive.py) is a receiver of about 30 lines that keeps every export as a JSON line:

```bash
BRIDGE=$(ip -4 -o addr show docker0 | awk '{print $4}' | cut -d/ -f1)
python3 receive.py $BRIDGE 4318 otlp.jsonl &
bench run tasks/primes -a claude -m xiaomi/mimo-v2.6-flash \
  --agent-env CLAUDE_CODE_ENABLE_TELEMETRY=1 \
  --agent-env OTEL_LOGS_EXPORTER=otlp \
  --agent-env OTEL_METRICS_EXPORTER=otlp \
  --agent-env OTEL_EXPORTER_OTLP_PROTOCOL=http/json \
  --agent-env OTEL_EXPORTER_OTLP_ENDPOINT=http://$BRIDGE:4318 \
  --agent-env OTEL_LOGS_EXPORT_INTERVAL=1000
```

The trial passed in 31 seconds, and the receiver got five log exports and one metrics export. Claude Code logs an `api_request` event for each model call, with its tokens. [`calls.py`](../examples/otel/calls.py) prints them beside the same calls in the proxy's record:

```bash
python3 calls.py otlp.jsonl jobs/2026-11-05__18-12-19/primes__c02bbd43
```

```text
call  otel: input  cache_read  output   proxy: prompt_tokens  completion_tokens
   1           35       14720     169                  14755                169
   2          215       14720      91                  14935                 91
   3          125       14912      47                  15037                 47
```

The two records agree call by call once you know how each counts. Claude Code's `input_tokens` leave out the tokens read from the cache, and the proxy's `prompt_tokens` include them, so input plus cache_read is the prompt. Output tokens are the same number in both. Each event also carries a `cost_usd`: that is Claude Code's own estimate from its price table, which isn't MiMo's price, so price a non-Claude model's tokens yourself, as below.

The trial's `config.json` keeps every variable you pass with `--agent-env`, values included, under `agent_env`. BenchFlow leaves out a variable whose name looks secret (it contains `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `AUTH` or `SESSION`, among others) or whose value looks like a credential.

## Cost for models without a price

BenchFlow prices tokens with the table bundled in its model proxy. A model missing from it, such as MiMo or a model released after the table, gets `cost_usd: null`, and the run summary says so:

```text
Cost:      not reported in USD (subscription logins and unpriced models report tokens only), 511,746 tokens
```

For such models, compare runs by tokens, cap them with `--max-tokens` rather than `--max-cost-usd`, and price the tokens yourself from the provider's rate card when you report a cost.
