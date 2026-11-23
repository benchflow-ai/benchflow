# Usage and telemetry

For every trial, BenchFlow records the tokens and cost the agent used, how long each step took, and, for each model call, what was sent and what came back. It reads these numbers at the model boundary, not from the agent's own telemetry, and records where each number came from.

Runs here use MiMo (`-m xiaomi/mimo-v2.6-flash`): export its key and base URL first ([Quick start](../start/quick-start.md#on-another-provider)), or use `-m sonnet` with a Claude login.

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

A task graded by an LLM judge (an `llm-judge` or `agent-judge` verifier, or a task.md judge) records the judge's calls apart, in `judge_usage`, with the same fields: its tokens as each provider response reported them (`usage_source` `provider`) and their cost at list prices (`cost_source` `list_price_estimate`, with `price_table_date`), plus `calls`, `calls_without_usage` and the `models` that answered. It is null when no judge ran. The judge's tokens are never added to `agent_result`. A response that reported no usage leaves the judge's tokens and cost null, since a sum without it would read as the whole.

`timing.json` has seconds per step, including how much of the agent's turn was spent waiting for the model (`provider_wait`) and how much doing anything else (`agent_without_provider_wait`); [How BenchFlow works](../start/how-it-works.md#one-rollout-step-by-step) shows a whole one.

Each model call is a line in `trajectory/llm_trajectory.jsonl`: the request as sent, the response, its HTTP status and how long it took. The first call of the same Claude Code trial:

```text
path: /v1/messages | duration_ms: 8025.3 | status: 200
request: model "mimo-v2.6-flash", reasoning_effort, stream true, tools [...]
metadata: {"request_model": "mimo-v2.6-flash", "provider_model": "mimo-v2.6-flash", "call_type": "anthropic_messages", "call_purpose": "agent"}
usage: {"completion_tokens": 169, "prompt_tokens": 15230, "total_tokens": 15399, "reasoning_tokens": 13, "cached_tokens": 15168}
```

That record is what to check when a setting matters: it shows the effort, sampling and model each request carried, not just what you asked for. A line holds `request.body` (the request as sent: `model`, `reasoning_effort`, `temperature`, `top_p`, `max_tokens` when set), `response.status_code`, `response.body` (`model` as answered, `choices[0].finish_reason`, `usage`) and `metadata.call_purpose`. [`requests.py`](../examples/audit/requests.py) prints those for every request of a job; for the Claude Code trial on MiMo in [OpenTelemetry](#opentelemetry) below:

```bash
python3 requests.py jobs/2026-11-05__18-12-19
```

```text
trial             call  purpose  model            answered         effort  temperature  top_p  max_tokens  status  finish
primes__c02bbd43  1     agent    mimo-v2.6-flash  mimo-v2.6-flash  high    -            -      -           200     tool_calls
primes__c02bbd43  2     agent    mimo-v2.6-flash  mimo-v2.6-flash  high    -            -      -           200     tool_calls
primes__c02bbd43  3     agent    mimo-v2.6-flash  mimo-v2.6-flash  high    -            -      -           200     stop
```

Claude Code sent effort `high` on each call and no sampling settings; an OpenCode trial on DeepSeek V4 Flash sent neither, and its first call has the purpose `title`, a request OpenCode makes for itself.

## Where each number comes from

`usage_source` says how the trial's tokens were counted:

| `usage_source` | Means |
|---|---|
| `provider_response` | read from each model response, as the calls passed through BenchFlow's model proxy |
| `agent_native_acp` | reported by the agent itself over ACP, as with a subscription login, which the proxy doesn't route |
| `unavailable` | not recorded; treat token and cost figures for the trial as missing, not zero |

codex-acp reports only its last model call's tokens over ACP. For a codex-acp trial the proxy doesn't meter, the tokens come from Codex's own session log instead, copied into the trial's `agent/codex-sessions/`: every response's usage, summed. `usage_details.token_count` then names the log and keeps what ACP reported; a codex-acp trial without it reports its last call's tokens only.

A count that misses calls says so. When a turn ends without its usage report (a timeout, a cancel, a lost sandbox) and no log of the agent's can count it, the trial's tokens are those of the turns that reported: result.json's `agent_result` then has `usage_incomplete: true` with `usage_incomplete_reason`, and `cost_incomplete: true` beside a cost priced from that count. ATIF's `final_metrics.extra` says the same, and summary.json and the run card count such trials as `trials_usage_incomplete` and `trials_cost_incomplete`, so a job's totals say when they are short. On the ACP path the agent's session log is copied out of the sandbox every 30 seconds while a turn runs and once more when it ends. A sandbox lost before a copy takes the calls since the last one with it; the count from what was copied is marked `usage_incomplete`. That gap is accepted. The trial export (`bench results --json`) carries the same three fields, and the `bench results` table shows such a trial's cost with a trailing `+`: a floor. The copy is redacted in a worker thread, so it never holds up the turn. The native harness keeps Claude Code's output stream on the host, so a lost sandbox costs it no tokens.

`price_source` says where the price came from: `litellm`, the price table of LiteLLM, the library BenchFlow's model proxy runs on; `agent_session_log`, an estimate from the agent's own session log, as for Claude Code on a subscription login; or `null` when there's no price for the model. `bench results` shows every cost whose `cost_source` is not `provider` as `~$…`: an estimate, not a bill. The proxy is always used for agents it can route, so usage and `llm_trajectory.jsonl` are always captured for them; `--usage-tracking required` additionally makes trusted usage (`provider_response` or `agent_native_acp`) a requirement of the run.

## What a job's summary counts

`summary.json` totals the trials two ways:

| Keys | Count |
|---|---|
| `total_input_tokens` ... `total_tokens`, `total_cost_usd`, `avg_cost_per_trial_usd`, `usage_sources`, `cost_sources` | the agents' usage over each task's final attempt (the trials the score reads) and the trials stopped before they finished |
| `judge_totals` | the LLM judges' usage over the same trials (`trials`, `calls`, the same token and cost totals, `usage_sources`, `cost_sources`) |
| `spend_all_attempts` | what the job spent over every attempt it recorded, retries and resumed runs included: `cost_usd` and `tokens` (agents' and judges' together), `judge_cost_usd` and `judge_tokens` (the judges' part), `attempts`, `attempts_without_cost` and `attempts_without_tokens` (unknown, never 0), `attempts_without_model_call` (an oracle, a nop, a script), and `stopped_attempts` (stopped by Ctrl-C, SIGTERM or a budget before they finished: what each one's receipt recorded counts; `stopped_attempts_without_usage` recorded nothing, `stopped_attempts_without_cost` tokens with no cost), `attempts_usage_salvaged` (stopped attempts that left no result and no receipt because their run stopped mid-turn (a host reset or a killed process), counted from the model proxy's log or the agent's session log they left: a floor, since a call still running then has no line), and `attempts_cost_incomplete` (a cost priced from some of the attempt's calls only, so `cost_usd` is a floor) |

A task that failed twice on infrastructure and then passed counts once in `total_cost_usd` and three times in `spend_all_attempts.cost_usd`. `spend_all_attempts` counts each attempt by the rule a job budget uses, so with a budget it agrees with `budget.spent` as the run recorded it. `bench backfill-cost` updates `spend_all_attempts` with the trials it prices, not `budget.spent`, which stays what the cap saw while the job ran. A total no trial reported is null, never 0. A run split across worker processes (`--worker-concurrency`) adds the workers' totals into the same keys. `bench eval metrics` shows the two as labelled rows, `Cost (total_cost_usd)` ("kept trials: …") and `Spend (spend_all_attempts)` ("every attempt: …"), each with how many trials or attempts reported no cost, and how many counted only some of their calls, which makes the total a floor.

## Refused requests and reasoning effort

Each trial's `model-calls.json` also says what BenchFlow's model proxy refused and the reasoning effort each request ran at, read from the proxy's record of every request it served (`trajectory/provider_requests.jsonl`, refused ones included):

| Key | Means |
|---|---|
| `refused_calls` | every refused call: the calls the model refused (a refusal or a content filter, `model_refused_calls`) and the requests the proxy refused before any provider saw them |
| `refusals_by_reason` | those by reason: `model_not_served` (a request for a model the run doesn't serve, such as a sub-agent's `haiku`), `trial_budget_spent`, `refusal`, `content_filter` |
| `models_refused` | the models a `model_not_served` refusal asked for, with how many requests each |
| `efforts_used` | the requests by the reasoning effort each went upstream at: Anthropic's `output_config.effort`, OpenAI's `reasoning_effort` and the Responses API's `reasoning.effort` are BenchFlow's level names already; an Anthropic thinking budget with no level is the agent's own (Claude Code's `MAX_THINKING_TOKENS`) and counts as `budget_tokens=N` |
| `efforts_requested` | the same by the effort each asked for; it differs from `efforts_used` only with the effort pin |
| `effort_mismatches` | the requests that went upstream at another level than the run's (`--effort`); a thinking budget with no level is not compared; null when the run asked for none |
| `effort_pin`, `effort_rewrites` | the level the pin held each request to, and how many requests it rewrote (below) |

A request that carried no effort is in none of the effort counts. A refused request used no effort and is in none of them either. These need the proxy: a Claude login or an endpoint the run names leaves them null. The job summary's Model calls row names the proxy's refusals and the models they asked for, and an Efforts used row appears when the job's requests used more than one level.

**Sub-agents.** Claude Code's Agent tool takes a `model` and an `effort` of its own. BenchFlow sets `CLAUDE_CODE_SUBAGENT_MODEL` to the run's model and `CLAUDE_CODE_SUBAGENT_MODEL_FORCE=1` in Claude Code's environment on every route, so a sub-agent runs on the run's model (the run card's Sub-agents line); a run that sets either variable itself (`--agent-env`) keeps its own and BenchFlow sets neither. An effort argument still applies: the sub-agent's requests carry it, and `efforts_used` shows it. `--effort-pin` (`effort_pin: true` in a run config; off by default, listed by `bench run --help-all`) has the proxy rewrite each request's effort level to the run's, a sub-agent's or Codex's `spawn_agent`'s included, on all three wires, and counts each rewrite in `effort_rewrites`; a thinking budget with no level is left as sent, since it is the agent's own budget, its main loop's too. It needs `--effort` and the proxy: a run without either is refused before it starts. Codex's `spawn_agent` with another `model` is refused by the proxy as `model_not_served` and counted. Off the proxy (a Codex login) nothing holds it: in the pinned Codex 0.159.3, `agents.default_subagent_model` is only a default that a `spawn_agent` call's own `model` overrides, and its multi-agent setting for those arguments only hides them from the model (Codex still applies them when sent), so run Codex on a route the proxy serves (an API key) to keep its sub-agents on the run's model.

## OpenTelemetry

BenchFlow 0.9.2 neither sends nor receives OpenTelemetry. Usage, cost and timing come from the model proxy and the trial's own records, so they don't depend on whether an agent exports telemetry of its own. The gap is a subscription login, which bypasses the proxy: then only what the agent reports over ACP is recorded (`agent_native_acp`), or nothing (`unavailable`).

0.10.0rc2 writes each trial's trace as OTLP/JSON to `trajectory/otel-spans.json`, sends it to the collector `OTEL_EXPORTER_OTLP_ENDPOINT` names where `bench` runs, and keeps the exports of an agent whose telemetry you turn on in `trajectory/agent-otel.jsonl` ([OpenTelemetry in the model-call records](../reference/model-calls.md#opentelemetry)). On 0.9.2, an agent can still send its own telemetry to a receiver you run. Claude Code's is off until you turn it on with its variables, which `--agent-env` passes into the sandbox. The receiver must listen on an address the sandbox can reach: for local Docker on Linux, Docker's bridge, which the first line below reads. Docker Desktop on macOS or Windows has no `docker0`; there a container reaches the host as `host.docker.internal`, so run the receiver on `0.0.0.0` and use that name in the endpoint (we ran this on Linux). [`receive.py`](../examples/otel/receive.py) is a receiver of about 30 lines that keeps every export as a JSON line:

```bash
git clone --depth 1 https://github.com/benchflow-ai/benchflow.git
mkdir -p tasks && cp -r benchflow/docs/examples/primes tasks/
cp benchflow/docs/examples/otel/receive.py benchflow/docs/examples/otel/calls.py .
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

The trial's `config.json` keeps every variable you pass with `--agent-env`, values included, under `agent_env`. BenchFlow leaves out a variable whose name looks secret (it contains `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `AUTH` or `SESSION`, among others), unless the name is a setting's such as `MAX_OUTPUT_TOKENS` or `SESSION_TIMEOUT`, and one whose value looks like a credential.

## Cost, and models without a price

BenchFlow prices tokens with LiteLLM's table. In 0.9.2 the proxy downloads that table from GitHub when it starts, if it can, and otherwise reads the copy installed with it, so the same run can be priced differently on another day. 0.10.0rc2 reads only the installed copy and carries vendors' own rates for 18 current models, MiMo V2.6 and DeepSeek V4 among them. Two consequences for 0.9.2:

- **DeepSeek V4's dollar figures are placeholders.** Its routes are priced at an unverified $0.28 and $0.42 per million input and output tokens, not DeepSeek's rates, so the DeepSeek costs in these docs, such as the $0.00857596 above, are that placeholder's. Report DeepSeek's tokens, or price them from DeepSeek's rate card with the date you read it.
- **A model missing from the table gets no price**, such as MiMo or any model newer than the table: `cost_usd: null`, and the run summary says so:

```text
Cost:      not reported in USD (subscription logins and unpriced models report tokens only), 511,746 tokens
```

For such models, compare runs by tokens, cap them with `--max-tokens` rather than `--max-cost-usd`, and price the tokens yourself from the provider's rate card when you report a cost.

**Claude Code on another provider's model.** Claude Code prices Claude models only; on an Anthropic-compatible endpoint (MiMo, DeepSeek, GLM, Kimi) its own figure is at its default Claude rate, 18 to 20 times MiMo's. Such a session is priced from Claude Code's own token counts at BenchFlow's list price (`usage_details.cost_estimate.method` `claude-code-tokens-at-list-price`, Claude Code's figure kept as `claude_code_usd`), and a model no list price covers has no cost, named in `unpriced_models`. A Claude Code trial's tokens count its sub-agents: `usage_details.token_count` names the `scope` and splits `main` from `subagents`. A session log over the 64 MiB copy limit is not copied, so such a trial keeps no cost.
