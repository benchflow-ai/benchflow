# Model outcomes beside verifier scores

Each trial's `result.json` has an `agent_outcome` object, and the job's `report.md` shows it beside the verifier's reward. It says how the agent's model ended. It never changes the reward, the pass verdict or the counts a score is taken over by itself; a trial whose agent's turn ended on a failed call is an error of its own (below).

| Field | Meaning |
|---|---|
| `status` | `refusal`, `provider_error`, `completed` or `unknown` (below) |
| `stop_reason` | How the agent's last prompt ended, as the agent reported it (`end_turn`, `refusal`, `max_tokens`, ...); null when the prompt did not end, as on a timeout |
| `stop_reasons` | How every prompt turn of the trial ended, in order, as `stop_reason` spells it; null for a turn that did not end. Empty when no prompt was sent (the oracle, `nop`) |
| `finish_reasons` | Each finish reason the trial's model calls ended with, and how many calls ended with it (`model-calls.json`); null when no call was recorded |
| `truncated` | The trial was cut at the output cap: the agent's last call ended at the cap (`length`, `max_tokens` or `max_output_tokens`), its last prompt ended `max_tokens`, or its error says the output cap was exceeded (Claude Code's "exceeded the 32000 output token maximum"). False when the trial ended some other known way; null when nothing says how it ended |
| `truncated_by` | What said so: `last_call`, `stop_reason` or `error`; null when `truncated` is not true |
| `ended_by`, `harness_cause` | `harness` when BenchFlow ended the prompt: its idle watchdog stopped a silent agent (`harness_cause` `idle_timeout`) or the agent's time limit ran out (`timeout`), the causes [When a trial fails](../evaluate/failed-trials.md) names; `agent` when the agent's last prompt ended on its own (a prompt BenchFlow cancelled did not); null when nothing says |
| `http_errors` | Calls that got an HTTP status of 400 or more; null when no recorded call has a status (an agent's session log has none) |
| `http_status_counts` | Every call by its HTTP status (`{"200": 528, "500": 11}`), the failures the agent survived among them; empty when no recorded call has a status |
| `agent_retries` | The agent's failed calls it followed with another call (Claude Code and Codex retry a 429 or a 5xx themselves, each try a call of its own); null without statuses |
| `upstream_retries` | The attempts the model proxy itself sent upstream after an attempt of the same request failed (`trajectory/provider_requests.jsonl`); null without that record |
| `refused_calls` | Calls that were refusals or stopped by a content filter; null when no recorded call has a finish reason |
| `terminal_provider_error` | The agent's last call failed with HTTP 429 or a 5xx status |
| `last_http_status` | The HTTP status of the agent's last call, or null |

The status is the first of these that applies:

1. `refusal`: the last prompt ended with `refusal`.
2. `provider_error`: the agent's last call failed with HTTP 429 (rate limited) or a 5xx status.
3. `refusal`: the agent's last call was a refusal: a `refusal` or `content_filter` finish reason, a Chat Completions message's `refusal`, or a Responses `refusal` part.
4. `completed`: the last prompt ended with `end_turn`. When the agent reported no stop reason, the trial also counts as completed if it recorded no error or timeout and the agent's last call answered 200 with a finish reason that ends an answer (`stop`, `end_turn` or `stop_sequence`). A Responses API call ends `completed` whether it answered or asked for a tool, so it is not enough.
5. `unknown`: anything else. A trial that timed out, or whose agent failed after a successful call, is `unknown` unless one of the first three applies.

An error or a refusal the agent recovered from stays in `http_errors` or `refused_calls` but does not set the status: a 503 followed by a successful answer is `completed`. A call cut at the cap that the agent recovered from stays in `finish_reasons` and does not make the trial `truncated`; only how the trial ended does. Calls made for a reviewer, a judge or a helper, and a Claude Code subagent's calls in its session log, count in the totals but are never the agent's last call. A call with no purpose recorded counts as the agent's. A refusal is never inferred from the agent's text.

Beside `agent_outcome`, `result.json` has `settings_sent`, what the agent's main calls carried ([Model calls](./model-calls.md)), and `compactions`, Claude Code's compactions in the trial ([The run card](../evaluate/before-you-run.md#the-run-card-010)).

The counts and the last call come from the trial's `model-calls.json` (see [Model calls](./model-calls.md)), written once when the trial's sandbox is torn down. A trial with no recorded calls has null counts, which means unknown, not zero.

## A turn that ended on a failed call

When the agent's turn ended with no error of its own and its last call had failed with HTTP 429 or a 5xx (`terminal_provider_error`), the model never answered that call, so the verifier's score of the workspace is not the model's answer. The trial is an `api_error`, `provider api error [provider_error/transient] HTTP 500 (the agent's last model call) on 11/539 requests` (`rate_limit` for a 429): unscored, the verifier's reward kept as `api_error_info.reward_withheld`, and retried as a transient provider error. When the agent's turn ended in an error of its own after a 5xx (500, 502, 503, 504 or 529), the error names the provider (`... | provider unavailable (HTTP 500)`), and the trial is retried as infrastructure; `bench explain provider_unavailable` explains it. Failures the agent survived stay counted and never change the score.

## The effort that ran

`result.json`'s `effort` puts the reasoning effort the run asked for beside the effort that ran, and `config.json` keeps `reasoning_effort_applied` (and `reasoning_effort_route` when the route sets one) beside `reasoning_effort`.

| Field | Meaning |
|---|---|
| `requested` | The `--reasoning-effort` the run asked for, or null |
| `applied` | The agent session's own word: the `model[effort]` id codex-acp took, the value an adapter answered an effort option with (the Antigravity shim answers `high` for `max`: agy has three levels), the effort a native CLI was given. With no effort asked for, the agent's default where the session names one |
| `route` | An effort the model proxy's route sets on each request itself (Bedrock's `high` ceiling for some Claude models), or null |
| `sent` | The agent's own model calls by the effort each carried (`agent_effort` in `model-calls.json`). The proxy sees every request, so `{}` from it means none carried an effort (a provider route that drops it); an agent's session log records only what it records, so null when it recorded none |
| `sent_source` | Where `sent` came from |
| `matches` | False when `applied`, `route` or any agent call's effort differs from `requested`; null when there is nothing to compare |

When `matches` is false the trial's `warnings` name what differed. It never changes the score. An effort the agent's session does not offer at all is refused before the agent starts: `agent integration failure [agent_effort]` (see [Agents and models](../evaluate/agents.md#choose-the-model-and-effort)).

## The tools the agent was offered

`result.json`'s `tool_surface` is the tools the agent's first request offered (`first_agent_tools` in `model-calls.json`: `count`, `names`, `sha256`), or null without the proxy. For Codex, `fallback` says whether they are Codex's fallback surface: Codex takes a model's tools from its model catalog, and runs a model the catalog lacks on fallback metadata, with plain function tools and neither `apply_patch` nor code mode's `functions.exec` (every model the pinned Codex catalogues has one of them). That trial measured another agent than the model is used with, so it fails: `agent integration failure [agent_model_catalog]`, unscored and not retried, whether Codex warned (`Model metadata for ... not found`, which stops an ACP trial at once) or not. `fallback` is null for other agents.

## Whether the task's inputs reached the model

`result.json`'s `input_delivery` is `model-calls.json`'s `input_files`: for each file the agent read with a read tool (Claude Code's `Read`, or a tool named `read_file` or `view_file`) that it had not written itself, what the result the model received in the next request carried, as the proxy saw it: a `document` or `image` block, or `text` with its length (`text_chars`) and hash (`text_sha256`). The proxy records the agent's request, from before LiteLLM's bridge to chat completions translates it, where a read still holds its document; for a result that held a document or an image, the proxy also runs that translation and records what the bridge made of it (`metadata.bridge_tool_results` in `trajectory/llm_trajectory.jsonl`: its parts, length, hash and first 240 characters), and that is what counts.

A result is a `stub` when it cannot hold the file: Claude Code's `PDF file read: <path> (47KB)` line with no document beside it, or fewer than 200 characters of text for a PDF, an image or an Office file. Each stub is a trial warning naming the file and the route; `recovered` says a later call that named the file (`pdftotext`, a script) brought back a long text. A stub does not fail the trial: agents often read the file another way after one (in one published run's sample, 2 of 8, 4 of 8 and 7 of 7 trials of three such rows did), and BenchFlow's bridge now turns a document into its text, so what is left is mostly a provider dropping a block the request still carried, which no record of the request can show. LiteLLM's own bridge, without BenchFlow's patch, leaves only the read's text line: that is a stub on the record. A row with stubs shows in the job's summary (below).

## In the job's summary

`summary.json`'s `model_calls` totals these over the job's trials: `http_errors`, `http_status_counts`, `trials_with_http_errors`, `agent_retries`, `upstream_retries` and `terminal_provider_errors` over the trials that recorded a status (`trials_with_status`); `tool_surfaces` lists each set of tools the agents' first requests offered, by `sha256`, with its `count`, `names`, `fallback` and how many `trials` had it (a job whose trials ran on two surfaces lists two); `fallback_tool_trials` counts the trials on a fallback surface, and `effort_mismatches` the trials whose `effort.matches` is false; `input_reads`, `input_stubs` and `input_stub_trials` count the agents' reads of task inputs, the files that reached the model as a stub, and the trials with one. It is absent when no trial recorded any of it.
