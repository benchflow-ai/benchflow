# Model calls

Every trial writes one record per model call to `trajectory/model-calls.jsonl`, and the trial's totals to `model-calls.json`, from the first of these sources the trial has:

| Source | Where it reads | When |
|---|---|---|
| `proxy` | `trajectory/llm_trajectory.jsonl`, BenchFlow's model proxy | the agent's calls went through the proxy (an API key) |
| `claude-code-otel` | `trajectory/agent-otel.jsonl`, the OpenTelemetry Claude Code sent to BenchFlow's receiver | the run turned Claude Code's telemetry on (below) |
| `claude-code-session-log` | `agent/claude-sessions/`, Claude Code's own `~/.claude/projects` | Claude Code went around the proxy (a Claude login) |
| `codex-session-log` | `agent/codex-sessions/`, Codex's own `$CODEX_HOME/sessions` | Codex went around the proxy (a ChatGPT login) |

An agent that calls its model around the proxy may export no OpenTelemetry either; Claude Code often doesn't. Its session log still holds each response, so BenchFlow copies the log, with credentials redacted, whenever the proxy saw none of the trial's calls. Set `BENCHFLOW_KEEP_AGENT_SESSION_LOGS=1` to keep it for proxied runs too, and check the proxy's records against the agent's own.

## A record (`benchflow.model-call.v2`)

| Field | Meaning | `proxy` | session logs |
|---|---|---|---|
| `source` | where the record came from | ✓ | ✓ |
| `index`, `started_at`, `duration_ms` | order, start time, time to the answer | ✓ | from the log's timestamps |
| `http_status` | the provider's answer status | ✓ | – |
| `purpose` | `agent`, a reviewer's or a judge's call | ✓ | `agent` |
| `model_requested`, `model_served`, `route` | the model asked for, the one that answered, the proxy's route | ✓ | `model_served` |
| `effort`, `thinking`, `temperature`, `top_p`, `max_output_tokens` | what the request sent | ✓ | Codex: `effort` |
| `tools` | how many tools the request offered the model (new in v2: 0.10.0rc1's `benchflow.model-call.v1` records have no `tools`) | ✓ | – |
| `seed` | the sampling seed the request carried as it went upstream: a route whose parameters drop it sends none (new in v2, as `tools`) | ✓ | – |
| `finish_reason`, `truncated`, `refused` | why the answer ended; cut at the output cap; a refusal or content filter | ✓ | Claude Code |
| `input_tokens` | every prompt token, cached ones included | ✓ | ✓ |
| `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `reasoning_tokens` | the rest of the usage | ✓ | ✓, as the log has them |
| `cost_usd` | the call's price, where a source has one | when the proxy prices the model | – |
| `subagent` | a Claude Code subagent's call | – | Claude Code |

A field a source cannot see is `null`, never a guess. `truncated` is true for `length`, `max_tokens` and `max_output_tokens`; `refused` for `refusal` and `content_filter`, and for an answer that is itself a refusal however it ended: a Chat Completions message's `refusal` or a Responses `refusal` part (both can end `stop` or `completed`). The proxy reads cached and reasoning tokens from the Chat Completions usage details (`prompt_tokens_details`, `completion_tokens_details`) or the Responses ones (`input_tokens_details`, `output_tokens_details`).

## The totals (`benchflow.model-calls.v1`)

`model-calls.json` counts the calls, the models served and requested, the effort, thinking and output caps sent, the finish reasons, the truncated calls and the truncation rate, the refused calls, the provider errors (`http_errors`), the tokens, the cost and the model time. `http_status_counts` counts every call by its HTTP status, and `agent_retries` the agent's failed calls it followed with another; `upstream_attempts`, `upstream_retries` and `upstream_errors` count the proxy's own attempts upstream, the attempts it made after one of the same request failed, and the attempts that failed (`trajectory/provider_requests.jsonl`; null without it). A request can make more than one call upstream without a failure (a lookup before the model call); those are not retries. `agent_effort` counts the effort of the agent's own calls alone (the calls `last_agent_call` reads, below): a session title's, a reviewer's or a judge's call can carry another. `input_files` says, for each file the agent read that it had not written (a task input), what its result carried in the next request: `kind` (`document`, `image` or `text`), `text_chars` and `text_sha256` of its text, `reads`, the proxy's `route`, `stub` and `recovered` (below); null when the agent read none or without the proxy. `first_agent_tools` holds the tools the agent's first request that offered any had, as the proxy saw it: `count`, `names` in order (a Responses namespace's tools as `namespace.name`, the tools a Codex code-mode request sends in an `additional_tools` input item among them) and `sha256` of the names, one per line; null without the proxy. `sources` names where each figure came from: the source, `none sent` for a setting the proxy saw no request carry, or `not recorded by <source>`.

`result.json`'s `settings_sent` counts what the agent's main calls carried: the calls to the run's own model, apart from side calls (the agent's calls to another model, such as Claude Code's background calls; a call to its own model that offers no tools while its other calls do, such as the title Claude Code's ACP session asks for; and a subagent's calls: `side_calls` and `side_models`). For `effort`, `max_output_tokens`, `temperature`, `top_p` and `seed` it has each value and the calls that carried it, `without` the calls that carried none, and the served models and routes; the job compares them with what the run asked for ([The run card](../evaluate/before-you-run.md#the-run-card-010)).

`last_agent_call` holds the `http_status`, `refused` and `finish_reason` of the agent's own last call, or null when there is none: a call made for a reviewer, a judge or a helper, a Claude Code subagent's call, or a side call to the agent's own model as above, is not the agent's, and a call with no purpose recorded is. `result.json`'s `agent_outcome` reads it ([Model outcomes](./model-outcomes.md)).

On the same trial, Claude Code's session log gives the proxy's figures: in a 30-call run under a 400-token output cap, both count 10 calls cut at the cap, the same input, output and cache-read tokens, and model time within 0.2%.

## OpenTelemetry

**Sent.** Every trial writes `trajectory/otel-spans.json`, an OTLP/JSON trace any OpenTelemetry Collector reads (`otlpjsonfile`) or takes on `POST /v1/traces`: a `trial <task>` span (task, agent, model, reward, error, and each phase's seconds as `benchflow.timing.*`), and one `chat <model>` span per model call with the GenAI attributes (`gen_ai.request.model`, `gen_ai.response.finish_reasons`, `gen_ai.usage.input_tokens`, ...) and `benchflow.model_call.*` (its `source` above among them). Set `OTEL_EXPORTER_OTLP_ENDPOINT` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) where you run `bench`, and each trial's spans also go there as it finishes, as OTLP/HTTP JSON, with `OTEL_EXPORTER_OTLP_HEADERS`. A collector that is down never fails a trial.

**Received.** Turn the agent's exporter on and name no collector, and BenchFlow receives its telemetry: `--agent-env CLAUDE_CODE_ENABLE_TELEMETRY=1` (or any `OTEL_LOGS_EXPORTER`, `OTEL_METRICS_EXPORTER` or `OTEL_TRACES_EXPORTER` naming `otlp`). A loopback OTLP/HTTP receiver runs as root in the agent's sandbox, the agent's `OTEL_EXPORTER_OTLP_*` point at it, and what arrives is copied to `trajectory/agent-otel.jsonl`, one export per line, as sent. It needs Python 3 in the task image; without it the run goes on and says why. Name your own `OTEL_EXPORTER_OTLP_ENDPOINT` in `--agent-env` and the telemetry goes there instead. Claude Code's `claude_code.api_request` events become `claude-code-otel` records: in a two-call trial through the proxy they gave the proxy's calls, tokens and effort, model time 2% longer (timed from the client), and Claude Code's own cost estimate, which is its price table's figure, not the provider's bill.

JSON Schemas: [`benchflow-model-call.v2.schema.json`](./schemas/benchflow-model-call.v2.schema.json), [`benchflow-model-call.v1.schema.json`](./schemas/benchflow-model-call.v1.schema.json) for 0.10.0rc1's records, and [`benchflow-model-calls.v1.schema.json`](./schemas/benchflow-model-calls.v1.schema.json).
