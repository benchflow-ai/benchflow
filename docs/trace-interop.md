# Trace interoperability

BenchFlow writes agent execution to disk in several shapes. This document
describes **the ACP-session capture-event format** — the records BenchFlow's
ACP-session emitter writes into `acp_trajectory.jsonl`, which BenchFlow both
produces and consumes — and records the current state of the two interchange
formats it is adjacent to, ATIF and OpenTelemetry.

Everything below is marked as one of:

- **FACT** — verified against the referenced implementation and tests when this
  document was introduced. Every FACT carries a symbol or `file:line` reference.
- **PROPOSAL** — a suggestion under discussion. Nothing marked PROPOSAL has been
  agreed, scheduled, or implemented.

The companion JSON Schema lives at
[`src/benchflow/trajectories/schemas/acp-capture-event-v1.schema.json`](../src/benchflow/trajectories/schemas/acp-capture-event-v1.schema.json)
and is exercised by `tests/trajectories/test_acp_capture_event_schema.py`.

---

## 1. Two different things are called "ACP"

**FACT.** The name covers two distinct objects, and only one of them has an
external specification.

| | ACP, the protocol | The BenchFlow ACP *trajectory* |
|---|---|---|
| What | Agent Client Protocol — a JSON-RPC wire protocol between a client and an agent | `trajectory/acp_trajectory.jsonl`, a BenchFlow artifact |
| Specified by | The `agent-client-protocol` SDK, re-exported in `src/benchflow/acp/types.py` | Nothing. This document specifies its ACP-session records only (§2.1) |
| Shape | Request/response and `session/update` notifications | A flat JSONL event log |
| Produced by | The agent | `benchflow.trajectories._capture`, plus the sources in §2.4 |

The ACP-session records are **derived from** the protocol, not equal to it. They
are a lossy, flattened projection: `ACPSession.handle_update` consumes protocol
notifications into in-memory state, and `_events_to_trajectory` projects a
subset of that state into the on-disk event records.

**FACT.** Fields the protocol carries and those records do not include the tool
call's `rawInput`, `rawOutput`, `locations` and `_meta`: `handle_update` reads
only `toolCallId`, `title`, `kind`, `status` and `content`. Verified by driving
a live `ACPSession` with all four present and observing their absence from the
captured events.

---

## 2. The ACP-session capture-event format, as emitted today

**FACT.** `trajectory/acp_trajectory.jsonl` is JSON Lines: one JSON object per
line, one object per event, in chronological order. There is **no envelope, no
header line, and no version field** — the file is the bare event sequence.

An empty trajectory is written as a **zero-byte file**, not as a blank line
(`TrajectoryWriter.write_final`). The serializer emits no trailing newline
(`redact_acp_trajectory_jsonl`); some callers append one (`hosted_env.py`), so
readers must tolerate both.

### 2.1 Scope of the schema

**FACT.** `TrajectoryWriter` performs **no validation**. It will serialize any
JSON-compatible dict handed to it, including records with an unknown `type` or
no `type` at all — verified by writing both and observing them persisted
unchanged.

The schema therefore describes **the normative output of the ACP-session capture
emitter** — the records constructed by `_events_to_trajectory` and by the
`ACPSession` legacy fallback in `_capture_session_trajectory` — and **not** the
set of every JSON value `TrajectoryWriter` is technically willing to serialize.
A file that fails the schema is not necessarily a file the writer would have
rejected; it is a file the ACP-session emitter would not have produced.

**The schema's scope is narrower than the file.** Distinguish between:

| | Covered by the schema? |
|---|---|
| The **ACP-session capture-event vocabulary** — records built by `_events_to_trajectory` and the `ACPSession` legacy fallback | **Yes.** This is exactly what the schema and the conformance suite pin |
| The **complete `acp_trajectory.jsonl` artifact** — everything that ends up in the file on disk | **No.** Two other sources contribute records; see §2.4 |

So **validating every line of a final trajectory against this schema is not
equivalent to validating the artifact** — a failing line may come from one of
those other sources rather than be malformed.

An artifact-level contract does not exist in this repository, and this document
does not create one. Whether `oracle` belongs inside the ACP trajectory contract
or is a separate downstream concern is open question 2 in §6.

### 2.2 Event types

**FACT.** `_events_to_trajectory` — the serializer for ACP sessions — emits
exactly **five** event types.

| `type` | Emitted by | Meaning |
|---|---|---|
| `user_message` | `ACPSession.record_user_prompt` | A prompt handed to the agent |
| `agent_message` | `agent_message_chunk` / `text_update` notifications, via `handle_update` | Agent text visible to the user |
| `agent_thought` | `agent_thought_chunk` / `agent_thought` notifications, via `handle_update` | Agent internal reasoning |
| `tool_call` | `tool_call` / `tool_call_update` notifications, via `handle_update` | One tool invocation and its captured output |
| `agent_timeout` | `ACPSession.record_agent_timeout` | BenchFlow's wall-clock timeout marker — not an ACP notification |

Consecutive text events of the same type are **merged into one record** before
serialization (`ACPSession._flush_agent_text`), so a single
`agent_message` may span many wire notifications.

**FACT.** `handle_update` recognizes six `sessionUpdate` values
(`ACPSession._RECOGNIZED_UPDATE_TYPES`) and returns early for anything else. It
does set `_events_active` before that check — so the session stays on the
event-log capture branch — but records no event and fires no snapshot or change
notification. Unknown update types from a future ACP version are therefore
dropped, not persisted.

### 2.3 Fields per event type

**FACT.** Every listed field is **always present** on records from the
ACP-session emitter — `_events_to_trajectory` builds these dicts as literals, so
none of them is conditionally omitted. "Required" below means required by the
schema; a missing key means the record did not come from that emitter.

#### `user_message` · `agent_message` · `agent_thought`

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | string | yes | One of the three values |
| `text` | string | yes | The merged text |

No other field is emitted.

#### `tool_call`

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | `"tool_call"` | yes | |
| `tool_call_id` | string | yes | May be `""` — `handle_update` defaults to the empty string rather than synthesizing an id |
| `kind` | string | yes | **Open vocabulary**, see below |
| `title` | string | yes | May be `""`. For ACP `execute` calls, conventionally the command line |
| `status` | enum | yes | `pending` · `in_progress` · `completed` · `failed` · `cancelled` |
| `content` | array | yes | Captured tool output; `[]` when there was none |

**FACT — `kind` is not an enum.** `_canonical_tool_kind`
passes the agent-supplied string through unchanged, so production values are not
limited to the vendored `benchflow.acp.types.ToolKind` members
(`other` · `bash` · `search` · `browser` · `read` · `write` · `skill`). Values
documented as production kinds in a code comment at
`tests/integration/agent_judge.py:118-124` include `execute`, `edit`, `delete`,
`move`, `fetch`, `think` and `switch_mode`, none of which are `ToolKind`
members. Independently verified: `handle_update` writes the literal `"tool"` as
the fallback kind when a `tool_call_update` arrives for an id that was never
opened. Constraining `kind` to `ToolKind` in the schema would reject data the
emitter can produce.

**FACT — `status` *is* closed.** The serialized value is always
`ToolCallStatus(...).value`, and a status the enum cannot parse falls back to
`in_progress`, so no out-of-vocabulary value reaches disk.

**FACT — `content` blocks are pass-through.** BenchFlow stores the wire value
verbatim and never reshapes it. Two shapes are known to be consumed downstream by
`content_blocks_to_text`: the nested ACP shape
`{"type": "content", "content": {"type": "text", "text": "..."}}` and a flat
`{"text": "..."}` form. The protocol also defines file-edit and terminal block
variants, which BenchFlow persists but does not interpret. The schema keeps
content blocks permissive for this reason.

#### `agent_timeout`

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | `"agent_timeout"` | yes | |
| `reason` | `"wall_clock_timeout"` | yes | Single value, hardcoded by `record_agent_timeout` |
| `timeout_sec` | number | yes | The budget that was exceeded |
| `pending_tool_call_ids` | array of string | yes | Tool calls not in a terminal status when the timeout fired |
| `terminal_trajectory_complete` | boolean | yes | Whether the capture is considered complete anyway |

### 2.4 Records in the file that the ACP-session emitter did not produce

**FACT.** **Three** other sources can put records into `acp_trajectory.jsonl`.
None goes through `_events_to_trajectory`, and the schema models none of them.
Note that the artifact *path* is the same in every mode, so the same filename
can hold different record shapes depending on how the rollout ran.

**Session-factory Sessions.** `_snapshot_session_trajectory` duck-types on the
ACP streaming attributes; a `benchflow.agents.protocol.Session` has none, so the
live `on_change` sink writes `session.steps` **unchanged**
(`_capture.py:116-117`), bypassing the ACP-session emitter entirely. Whatever
that plane puts in `steps` is what reaches the streaming file; what survives the
final write depends on the trajectory list the caller passes to
`TrajectoryWriter.write_final`.

**The `oracle` record — a whole alternative trajectory, not an addition.** In
`oracle_mode` the rollout does not run an agent at all: `_run_oracle`
(`rollout/_setup.py`) builds a **new, oracle-only** trajectory carrying `type`,
`command`, `return_code` and `stdout`, and the rollout **assigns**
`self._trajectory` to that list. ACP rollouts and oracle rollouts are therefore
mutually exclusive today, and **no production path emits a mixed ACP + oracle
artifact**.

**`hosted_env`, which writes the file directly.** `_reconstruct_trajectory`
(`hosted_env.py:655`) rebuilds a trajectory from a vf-eval `results.jsonl` and
`hosted_env.py:552` writes it to the artifact path **without going through
`TrajectoryWriter`**. Its own docstrings call the output "ACP-shaped" and
"ACP-style session events", and the resemblance is real but partial: the records
reuse the `user_message` / `agent_message` type strings while carrying the text
under **`content`** rather than `text`, adding `ts` and `example_index`, and
introducing a **fourth type, `reward`**, with `value` and `source`
(`hosted_env.py:720-767`). Every one of those records fails the §2.1 schema —
on the required field, on `additionalProperties`, and for `reward` on the type
enum. This is provenance-tagged: `trajectory_source="hosted_env"`.

**A fourth record family is recognized by consumers, and nothing in this
repository writes it.** `viewer.py:573` branches on
`{session_meta, response_item, event_msg, turn_context}`, and
`publish/traj_report.py:243-246` reads the same set. No site under `src/` emits
those types. They are an **externally produced** shape: `cli/traj.py:549`
identifies `session_meta` as a **Codex** payload, alongside Claude's `cwd`
events, and that CLI uploads sessions recorded by those harnesses. So this is a
foreign trace format the consumers tolerate, not a BenchFlow producer — worth
separating, because "a consumer knows how to read it" and "something here writes
it" are different facts and only the first is shown.

So the file is not a superset of the ACP-session emitter's output; it holds one
of several possible shapes, and no single site defines its full vocabulary.

**There is, however, a partial typed contract nobody has connected to this
schema**: `TrajectorySource = Literal["acp", "scraped", "partial_acp",
"hosted_env"]` (`models.py:17`) records *which lineage produced the file* and
reaches `result.json`. It has four members and covers neither the oracle records
nor the family above — in oracle mode `"oracle"` is the **agent name**, not the
trajectory source. So the artifact's provenance is typed while its *content* is
not, which is the gap §2.1 describes from the other side.

Whether `oracle` belongs in the ACP trajectory contract or is a separate
downstream concern is an open question (see §6), and this document does not
settle it.

---

## 3. Producers and consumers

### 3.1 Producers

**FACT.**

| Stage | Site |
|---|---|
| Wire → session state | `ACPSession.handle_update` |
| Session state → events | `_events_to_trajectory` |
| Live streaming to disk | `TrajectoryWriter.flush`, wired as `session.on_change` |
| Multi-scene cumulative streaming | `make_trajectory_sink` |
| Final authoritative write | `TrajectoryWriter.write_final`, called from `rollout/_results.py` |
| Session-factory passthrough (§2.4) | `_snapshot_session_trajectory` (`_capture.py:116-117`) |
| Agent-native fallback (Gemini CLI) | `_scrape_agent_trajectory` / `_parse_gemini_trajectory` |
| Provider-evidence repair | `_reconcile_tool_evidence` / `_parse_provider_tool_evidence` |
| Oracle mode — replaces the trajectory (§2.4) | `_run_oracle` |
| Copy into the sandbox for verifiers | `_publish_trajectory_for_verifier` |

### 3.2 Consumers

**FACT.** Each consumer below parses the file independently; there is no shared
reader.

| Consumer | Site |
|---|---|
| Trajectory viewer (`bench eval view`) | `trajectories/viewer.py:161-164, 283` |
| Skill-eval / GEPA export | `skill_eval/gepa_export.py:28-53` |
| In-sandbox judge | `adapters/resources/mcp_atlas_judge.py:27` |
| Task verifiers (declarable input) | `docs/task-standard.md:332, 587` |
| Integration rubric gate | `tests/integration/rubric_checks.py:565-599` |
| Harbor parity check | `tests/integration/check_skillsbench_harbor_parity.py:113-145` |
| Experiment-review skill | `.agents/skills/benchflow-experiment-review/scripts/validate_run_artifacts.py:767` |

---

## 4. State of the adjacent formats

### 4.1 ATIF

**FACT.** ATIF (Agent Trajectory Interchange Format) support is one module,
`src/benchflow/trajectories/export_atif.py`, pinned to `ATIF-v1.7`
(`ATIF_SCHEMA_VERSION`). It writes `trainer/atif.json` per scored rollout, from
`_write_trainer_artifact`.

- **No production reader.** Nothing under `src/` reads it back; the only code
  that loads or validates ATIF is test/integration code.
- **No models and no schema.** Documents are assembled as plain `dict` literals;
  the specification is prose in the module docstring, pinned against upstream
  Harbor models and the ATIF RFC, **neither of which is vendored or checked by
  any test**.
- Its validator is `atif_issues` (`tests/integration/scenarios.py:262-290`),
  which checks that `schema_version` starts with `ATIF-v1` and that each
  `step.source` is in `{user, agent, oracle}`.
- The trajectory viewer explicitly does not read it (`viewer.py:3`: *"No ATIF
  conversion."*).

**FACT.** The same module family also emits ADP (`export_adp.py`, version
`1.3.1`) and the Verifiers/ORS record (`export.py`). All three walk the same ACP
event list, from the same call site, `_write_trainer_artifact`.

### 4.2 OpenTelemetry

**FACT.** There is **no OpenTelemetry representation in the codebase**. There is
no OTel dependency in `pyproject.toml`, and no OTLP or `gen_ai.*` handling in
`src/`.

`src/benchflow/trajectories/otel.py` — recorded in `CHANGELOG.md` under
*Removed*, "Removed the unwired `OTelCollector`" — was deleted as *"a
designed-but-never-wired OTLP receiver … never instantiated, never tested, and
not part of any run path"*. It converted **inbound only** (OTLP/JSON spans into
`LLMExchange` records) and mapped no tool calls, no span status and no
parent/child relationships. It contained no emitter, and no other OTel-named
module appears in the repository history.

**What *is* pinned, and why it matters.** `uv.lock` resolves
`opentelemetry-proto` **1.41.1**, `opentelemetry-sdk` **1.41.1** and
`opentelemetry-semantic-conventions` **0.62b1**, transitively, as dependencies of
`daytona` — the `sandbox-daytona` extra (`pyproject.toml:69-75`). None of them is
a BenchFlow dependency, none is installed in the default dev environment, and
nothing in `src/` imports them. They matter for one reason: they are artifacts
this repository already names, at a version, with a hash. So a question about the
OTLP wire shape or about a `gen_ai.*` attribute name has a *checkable* answer
here, instead of an answer from recollection. §8.3's OTel edge is written against
those two artifacts and records both versions in code.

Two facts worth stating up front, because both bound what any OTel mapping can
claim:

- **The GenAI conventions are experimental even at the pinned version.** The
  package puts every `gen_ai.*` name under `opentelemetry.semconv._incubating`,
  and several are already marked deprecated in 0.62b1 — `gen_ai.system`,
  `gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens`,
  `gen_ai.prompt`, `gen_ai.completion`.
- **The deleted collector's attribute names do not match it.** `otel.py` read
  `gen_ai.usage.total_tokens`, which does not exist at 0.62b1, and
  `gen_ai.usage.cache_read_input_tokens` / `…cache_creation_input_tokens`, which
  are spelled `gen_ai.usage.cache_read.input_tokens` and
  `gen_ai.usage.cache_creation.input_tokens`. Its `_parse_attributes` also read
  `stringValue or intValue or doubleValue or boolValue`, which turns an observed
  `""`, `0` or `false` into "no value".

---

## 5. Information loss in today's conversions

**FACT.** All losses below are observable in the current code. They describe
`ACP → ATIF` and `ACP → ADP` as implemented; nothing here is hypothetical.

| # | What is lost | Where | Effect |
|---|---|---|---|
| 1 | **Tool arguments** | `acp_events_to_atif_steps` emits `"arguments": {}`; `acp_events_to_adp_content` emits `"kwargs": {}` | Exported trajectories record *that* a tool ran, not *with what*. The arguments are absent because the capture path drops `rawInput` (§1), not because the target formats lack a field |
| 2 | **Tool status** | ATIF demotes it into the non-standard `extra`; ADP drops it outright, as its module docstring states | A failed tool call is not machine-distinguishable from a successful one in ADP |
| 3 | **All timestamps** | `ToolCallRecord.started_at` / `.finished_at` exist and are not serialized by `_events_to_trajectory` | No per-event timing survives in any format. Only rollout-level wall clock exists, in `result.json` / `timing.json` |
| 4 | **`agent_timeout` events** | Emitted by `_events_to_trajectory`; no branch handles the type in `acp_events_to_atif_steps`, `acp_events_to_adp_content` or `acp_events_to_messages` | The timeout marker is absent from every exported document. In ADP, where `status` is dropped too (#2), no signal that the rollout timed out survives at all |
| 5 | **Non-text content blocks** | `content_blocks_to_text` skips them | File-edit diffs and terminal output are dropped from ATIF and ADP |
| 6 | **Per-step token usage** | `trajectory_to_atif_record` | Only four trajectory-level totals survive, and they are sourced from the LLM proxy capture (`rollout/_results.py:445-448`), not from ACP |
| 7 | **Agent version** | Hardcoded `"unknown"` by `trajectory_to_atif_record` | Documented in the code as deliberate rather than fabricated |
| 8 | **ACP session token usage** | `ACPSession.usage_snapshots` is routed to `result.json`, never into the trajectory | The trajectory carries no usage at all |
| 9 | **`stop_reason`** | Captured as `ACPSession.stop_reason`, never exported | Why the agent stopped is not in any trajectory format |
| 10 | **`agent_thought` boundaries** | `ThoughtBuffer.take` joins buffered thoughts with a blank line | One thought containing a blank line and two consecutive thought events produce the same `reasoning_content`; the number of thought events is not recoverable. Reachable in production — `_parse_gemini_trajectory` appends one event per entry of a message's `thoughts` list |

**FACT.** Loss #4 is worth noting on its own: the timeout marker is written to
the trajectory so downstream consumers can see it, and every exporter drops it.
It is not lost at rollout level — `trajectory_summary.event_type_counts` in
`result.json` counts every event type including `agent_timeout`
(`_utils/result_metadata.py`), and the `agent_timeout_info` diagnostic records
it separately. The loss is confined to the exported trajectory documents.

### 5.1 `ACP → ATIF`, field by field

**FACT.** Every row below is asserted by
`tests/trajectories/test_atif_preservation.py` against records built through
the production capture path. This section describes what the conversion does
today; it proposes nothing.

Four classes are used, and the distinction between the last two is where a fix
would have to land:

- **preserved** — reaches the ATIF document unchanged.
- **normalized** — reaches it relocated or reshaped, so a consumer must know
  BenchFlow's convention to read it.
- **dropped** — the capture events carry it and the converter discards it.
  Fixable in `export_atif.py` alone.
- **unsupported** — the capture events never carried it. `export_atif.py`
  cannot fix these; the loss is upstream, at `handle_update` or in
  `_events_to_trajectory`.

| ACP capture field / event | In ATIF today | Class |
|---|---|---|
| `user_message.text`, `agent_message.text` | `step.message`, verbatim | preserved |
| text-empty message or thought event | no step emitted | dropped |
| `agent_thought.text` | joined into the next agent step's `reasoning_content` | normalized (loss #10) |
| `tool_call.tool_call_id` | `tool_calls[].tool_call_id`; synthesized `call_{n}` when empty | preserved |
| `tool_call.kind` | `tool_calls[].function_name`; `"tool"` when empty | normalized — an ACP kind is a category, not a function name |
| tool arguments | `"arguments": {}`, always | unsupported — see §1 |
| `tool_call.content`, text blocks | `observation.results[].content` | preserved |
| `tool_call.content`, non-text blocks | absent | dropped (loss #5) |
| `tool_call.status`, `tool_call.title` | `tool_calls[].extra`, stringified | normalized (loss #2) — `extra` is ATIF-v1.7 and non-standard |
| `rawInput` / `rawOutput` / `locations` / `_meta` | absent | unsupported — dropped at the wire, never captured |
| `ToolCallRecord.started_at` / `.finished_at` | absent | unsupported — never serialized (loss #3) |
| event order | `step_id`, dense from 1 | preserved |
| per-step token usage | no `metrics` on any step | unsupported (loss #6) |
| `ACPSession.usage_snapshots`, `stop_reason` | absent | dropped (losses #8, #9) |
| `agent_timeout` | absent | dropped (loss #4) |
| unknown event types, non-dict entries | skipped silently, no gap in `step_id` | dropped, deliberate |
| `oracle` record | `source: "agent"`, `message: "[oracle: <cmd>]"` | normalized — **ambiguous**, see below |
| agent version | `"unknown"` | unsupported, deliberate (loss #7) |
| `prompts` argument | leading `user` steps | addition — not an ACP event |

**FACT — the `oracle` source is a live divergence.** The in-repo validator
(`tests/integration/scenarios.py`) accepts `source` in
`{"user", "agent", "oracle"}`; the emitter produces only `user` and `agent`,
rendering oracle activity as an `agent` step with an `[oracle: …]` prefix. The
produced set is therefore a strict subset of the accepted set, and a consumer
cannot tell an oracle step from an agent step except by matching that string.
Whether the emitter should produce the third value or the validator should drop
it is open question 3 below; this document does not settle it, and the test
suite asserts the divergence rather than either resolution.

**FACT — an ATIF document is not a function of the capture file alone.**
`prompts` adds steps with no corresponding ACP event, and text-empty events add
none, so the event list cannot be reconstructed from the document. Any future
round-trip claim has to account for both.

### 5.2 Confirmed against real rollouts

**FACT.** Two rollouts of the `gemini` agent were run through the production
path — docker sandbox, ACP transport, the standard artifact writers — and their
`acp_trajectory.jsonl`, `trainer/atif.json` and `result.json` were inspected by
hand. Everything §5.1 describes held. This section records only what those
artifacts showed.

In both rollouts:

- every `tool_call` capture record carried exactly `type`, `tool_call_id`,
  `kind`, `title`, `status` and `content` — the six fields §5.1 predicts, and
  nothing else;
- every `tool_call_id` reached the ATIF document unchanged;
- textual tool output reached `observation.results[].content`;
- `kind` became `function_name`, with observed values `execute`, `read` and
  `think` — none of them `ToolKind` members, corroborating §2.3;
- `title` and `status` appeared only inside `tool_calls[].extra`;
- every `arguments` was `{}`;
- no ISO-8601 value appeared in either artifact;
- `rawInput`, `rawOutput`, `locations` and `_meta` appeared in neither the ACP
  capture nor the ATIF document.

**FACT — the tool inputs existed and were dropped, not unavailable.** In the
same rollouts the proxy capture (`llm_trajectory.jsonl`) carried non-empty
tool-call `arguments` payloads — in one case keyed `command` and `description`,
holding the shell command the agent ran — while the ATIF document recorded `{}`
for that same call. The inputs were therefore present in traffic BenchFlow
already captures, and absent from the exported trajectory.

This is **not** evidence that the ACP `rawInput` family was on the wire. Those
four fields occurred nowhere in the captured artifacts, the proxy capture
included, which is expected because that capture is not ACP. Their absence from
the ACP capture is consistent with `handle_update` never reading them (§1), but
no observation in this repository shows them arriving.

**FACT — the prompt-derived step duplicates the first user message.** Both ATIF
documents opened with two `user` steps carrying identical text: one built from
the `prompts` argument, one from the captured `user_message` event. A consumer
counting user turns from an ATIF document over-counts by one.

**FACT — `agent_timeout` is written to the capture and exported nowhere.** A
rollout driven into its wall-clock budget recorded

```json
{"type": "agent_timeout", "reason": "wall_clock_timeout", "timeout_sec": 90.0,
 "pending_tool_call_ids": [], "terminal_trajectory_complete": true}
```

in `acp_trajectory.jsonl`, and `result.json` counted it under
`trajectory_summary.event_type_counts.agent_timeout`. The ATIF document for that
same rollout carried no representation of it — confirming loss #4 end to end,
including that the signal survives at rollout level (§5, note after the table).

Note the shape observed: the budget expired between tool calls, so the pending
list was empty and the capture was marked terminally complete. The other branch
— a timeout with tool calls still in flight — is covered by the test suite but
was not observed in a real rollout.

**Not exercised by these rollouts**, and still resting on code reading plus the
synthetic tests: the non-text content-block loss (#5), because neither agent
emitted a file-edit or terminal block; and the `oracle` source divergence,
because neither rollout ran in oracle mode.

---

## 6. Open questions

These are unresolved in the repository today; this document does not answer
them.

1. Should `acp_trajectory.jsonl` carry a `schema_version` field? It has none
   today, and the consumers listed in §3.2 parse it without one.
2. Is `oracle` part of the ACP trajectory contract, or a separate downstream
   record type that happens to share the file? (§2.4)
3. Should the ATIF validator's `oracle` source be produced by the emitter, or
   removed from the validator? (§5)
4. Which definition of `input_tokens` is canonical — the cross-provider
   normalized one in `_exchange_token_usage`, or the unnormalized ACP snapshot
   in `normalize_acp_usage`?
5. Is OpenTelemetry wanted as an inbound receiver, an outbound emitter, or
   both? The only implementation ever written was inbound, and it was removed
   deliberately. **Still unanswered by anyone who owns the answer.** §8.3
   implements the inbound half (`OTel → IR`); §8.12 records *our* decision to
   scope this implementation to ingest and to defer the emitter, and is careful
   to say that the decision is ours rather than a reading of this question. The
   emitter is not written, and §8.11 lists the decisions it would depend on.

**Still open.** §8 takes a *provisional* position on question 1 — a canonical
hub — and implements it in isolation so it can be reviewed as code. That is a
proposal, not an answer: no maintainer has agreed it, nothing depends on it, and
questions 2–5 are untouched by it.

---

## 7. Known divergences from this schema

**FACT.** Files and fixtures in this repository that use the ACP trajectory
filename or shape but do **not** conform to the format described above. They are
**excluded from the conformance corpus**. Their current non-conformance is
recorded here as an observation, not asserted as a contract — a future change
that brings any of them into conformance is an improvement, not a regression.

| Location | Divergence |
|---|---|
| `.agents/skills/benchflow-experiment-review/evals/files/*/trajectory/acp_trajectory.jsonl` (5 files) | Synthetic eval-harness fixtures that reuse the filename without following the production writer. They use `phase`, `tool`, `args` and `reward` keys, and `type` values `final`, `score`, `stderr`, `timeout` — none of which the writer emits. They pass their own validator, which requires only that `type` be a non-empty string (`validate_run_artifacts.py:232-244`) |
| `tests/test_judge_robustness.py:32` `_tc()` | A nested `{"source": ..., "tool_calls": [...]}` shape, documented in place as the "synthetic/deepagents" form — not ACP |
| `tests/test_judge_robustness.py:38` `_native()` | A `tool_call` with `type`, `tool_call_id`, `kind` and `title` only; the scanner under test reads no other field |
| `tests/test_integration_check_results.py:197` | A `tool_call` with `type`, `kind` and `title` only |
| `tests/test_integration_suite.py:242` | `{"role": "assistant"}` — no `type` |
| `tests/acceptance_live_harness.py:210` and `tests/test_acceptance_live_execution.py` | `{"type": "oracle", ...}` and `{"type": "agent", ...}` records; `oracle` is the oracle-mode type of §2.4; no producer of an `agent`-typed record was found under `src/` |

The common pattern is that these are **inputs to a consumer under test**, minimal
by design because the consumer reads only a few fields. That is a legitimate
testing practice and not a claim about the file format.

---

## 8. Canonical Trace IR v0 — provisional

> **PROPOSAL — not approved.** This section describes a design decision taken
> *without* maintainer sign-off, so the task could continue while the questions
> in §6 stayed open. It is implemented in
> [`src/benchflow/trajectories/ir.py`](../src/benchflow/trajectories/ir.py) so it
> can be reviewed as code rather than as a sketch. **Nothing imports it, nothing
> writes it to disk, and no existing format, artifact or code path changes
> because it exists** — a property pinned by a test
> (`test_no_runtime_module_imports_the_ir`). Deleting the module and its test
> returns the tree to its previous behaviour.
>
> Statements about the *current code* below remain FACT, with references, as
> everywhere else in this document. Statements about the IR are PROPOSAL.

### 8.1 Why a hub, and why not pairwise converters

**FACT.** Four trace-shaped representations already exist in this repository —
the ACP-session capture events (§2), ATIF (§4.1), ADP and the Verifiers/ORS
record — and all three exporters walk the same ACP event list from the same call
site, `_write_trainer_artifact`. OpenTelemetry (§4.2) would be a fifth.

Pairwise conversion costs `N*(N-1)` directed edges, but the cost that matters is
not the edge count. It is that **each edge answers the same questions
privately**, and in this codebase those answers already diverge:

**FACT.** For the identical ACP `tool_call` event, ATIF emits
`"arguments": {}` while ADP emits `"kwargs": {}`; ATIF preserves the tool status
in a non-standard `extra` while ADP drops it (§5, losses #1–#2). Both join
thought boundaries irreversibly through the same `ThoughtBuffer` (loss #10), and
neither represents `agent_timeout` at all (loss #4) — three independent
decisions about the same event, taken three times, recorded nowhere.

A hub turns each format into one edge against a written contract, and — the
actual point — makes the loss a **typed value** rather than a comment in a
module docstring. See `LossReport` / `LossRecord` in the module.

The alternative shapes considered and not taken:

| Option | Why not |
|---|---|
| **Direct pairwise converters** | What exists today. Each new format multiplies the decisions above, and none of them is recorded anywhere a consumer can read |
| **Promote ATIF to the hub** | ATIF has no reader, no models and no vendored schema in this repository (§4.1); it cannot represent `agent_timeout` or per-event timestamps, so it would bake today's losses into the hub |
| **Promote the ACP capture events to the hub** | It is a lossy projection of the protocol by construction (§1), has no artifact-level contract (§2.1), and its file can hold non-ACP records (§2.4) |
| **Extend the capture format instead** | Changes the on-disk artifact that the consumers in §3.2 already parse — a compatibility decision, and precisely open question 3 |

### 8.2 What the IR carries, and on what evidence

**PROPOSAL.** The rule the module is built on: *the IR is a pragmatic superset
of what BenchFlow can observe, not a model of what an agent trace could contain*.
Every field is in one of four states, and the distinction is deliberate:

| Class | Meaning | Fields |
|---|---|---|
| **Supported today** | Some source in this repository carries the value now | event order · event kind · role · text · reasoning · tool call id / kind / title / status · content blocks · outcome status · trace-level usage · session id · agent + model name |
| **Optional** | Carried when the source has it, absent without loss when it does not | trace id · provider · reward · error category · stop reason · per-event usage |
| **Needs enrichment first** | The slot exists because the value demonstrably exists *upstream* of the capture, and is dropped before disk | tool `arguments` (the ACP `rawInput` family, §1) · per-event `started_at` / `finished_at` (`ToolCallRecord` tracks both, loss #3) |
| **Must not be invented** | The IR deliberately has no way to fabricate these | agent version (`"unknown"` is ATIF's requirement, not a fact) · synthetic tool-call ids (`call_{n}`, `call_NNNNNN`) · timestamps for sources that carry none · OTel span/trace ids |

Three design choices carry most of the weight:

1. **Tri-state optionality.** A value, `None` ("this source never carried it"),
   and an empty value ("carried, and empty") are three different facts.
   `arguments={}` and `arguments=None` are the case that matters: every
   ACP-derived tool call is the second, and both ATIF and ADP serialize the
   first, which is why their documents read as though every tool was called with
   no arguments.
2. **Absence must be declared.** A `None` argument map without a matching
   `LossRecord` is an *invalid* trace (`validate_trace`, invariant 7). This is
   what makes the loss report a contract instead of documentation.
3. **Normalization is never destructive.** `TraceEvent.source_type` keeps the
   source's own type string next to the normalized `kind`, and
   `ToolCall.name_semantics` records that an ACP `kind` is a category rather
   than a function name — the normalization §5.1 currently performs silently.
4. **The canonical JSON encoding keeps the nulls.** A trace serializes with
   `model_dump(mode="json")`; **`exclude_none=True` is not a valid encoding of a
   Trace IR document.** `None` is a positive statement — *the source did not
   carry this* — and the `LossRecord` that legalizes it addresses the field **by
   path**. Drop the key and that address stops resolving, so the declaration
   becomes unverifiable inside the document that carries it. Both encodings
   re-validate to an equal pydantic model, which is exactly why the rule lives
   in a test rather than in the type; the audience of an interchange format
   reads the JSON. The corollary is that a record names the outermost absent
   node — a conversion with no usage declares `usage`, not `usage.input_tokens`
   — and that sections every conversion has an opinion about (`agent`,
   `outcome`) are always present, with `None` fields inside them. The normative
   statement lives in the `ir.py` module docstring.

   *This rule was written after the fact: §8.4 was first published with
   `exclude_none=True`, and a human end-to-end read of a converted rollout found
   the loss records pointing at keys the document did not contain.*
5. **A loss record declares which document its path addresses.** Not every
   record is about an IR node: an inbound edge can read an input element that
   becomes no IR node at all, and an outbound edge can emit a value the IR never
   held — one its target format requires, or one supplied by the conversion
   context. `LossRecord.space` is `hub` · `source` · `target`, defaulting to
   `hub`, and `field` is the path *inside* that space with no prefix repeating
   it. Three spaces cover every direction, because **every edge has the IR on
   exactly one side** and therefore exactly one non-hub space: OTel will add
   none.

   Only `hub` records compose across edges — the IR is the output of an inbound
   conversion and the input of an outbound one, so `events[1].tool_call.arguments`
   denotes the same field in both reports and the records join on it: `acp -> ir`
   declares it `unsupported`, `ir -> atif` declares it `synthesized`, and read
   together they are the whole history of that field along the pipeline.
   `source` and `target` records are terminal by construction. There is no
   unified report; composition happens at read time over
   `(direction, field, space)`.

   **Which side owns the report** follows from the same asymmetry. A trace is
   built exactly once, so an inbound conversion may attach its report to the
   trace (`CanonicalTrace.losses`). A trace may be converted to many targets, so
   an outbound conversion returns its report alongside its document and leaves
   `trace.losses` untouched.

### 8.3 Implemented mapping

**`ACP → IR`** ([`ir_from_acp.py`](../src/benchflow/trajectories/ir_from_acp.py)),
**`IR → ATIF`** ([`ir_to_atif.py`](../src/benchflow/trajectories/ir_to_atif.py)),
**`ATIF → IR`** ([`ir_from_atif.py`](../src/benchflow/trajectories/ir_from_atif.py)),
**`OTLP/JSON → IR`**
([`ir_from_otel.py`](../src/benchflow/trajectories/ir_from_otel.py)) and
**`IR → ACP capture events`**
([`ir_to_acp.py`](../src/benchflow/trajectories/ir_to_acp.py)) **are
implemented**, all unwired and still provisional. `IR → OTel` is not, and is
deliberately not sketched — §8.11 lists the decisions it depends on.

With both ATIF edges in place the loop `ACP → IR → ATIF → IR′` closes, and §8.10
reports what it measures. The OTel edge is inbound only, so no loop closes
through it and §8.11 reports what it showed instead.

#### ACP-session capture events → IR

**FACT** for the implemented rows — asserted by
`tests/trajectories/test_ir_from_acp.py` against events produced by driving a
real `ACPSession` through the production capture path.

| Capture field / event (§2.2) | IR | Class | Loss recorded |
|---|---|---|---|
| event order | `events[].index`, dense from 0 | preserved | — |
| `type` | `kind` **and** `source_type` verbatim | preserved | — |
| `user_message.text` | `text`, `role=user` | preserved | — |
| `agent_message.text` | `text`, `role=agent` | preserved | — |
| text-empty message (`""`) | `text=""`, event kept | preserved | — (both exporters drop the event) |
| `agent_thought.text` | `reasoning` **and** `reasoning_segments=[text]` | preserved | — the segment list is what avoids loss #10 |
| `tool_call.tool_call_id` | `tool_call.call_id`, `""` kept as `""` | preserved | — (ATIF/ADP synthesize here) |
| `tool_call.kind` | `tool_call.name` + `name_semantics="acp_kind"` | preserved | — the semantics is recorded instead of assumed |
| `tool_call.title` | `tool_call.title` | preserved | — |
| `tool_call.status` | `tool_call.status` | preserved | — |
| `tool_call.content`, text blocks | `ContentBlock(kind=text)` + `raw` | preserved | — |
| `tool_call.content`, other blocks | `ContentBlock(kind=opaque, raw=…)` | preserved | — carrying the block verbatim is how loss #5 stops being a loss |
| tool arguments | `arguments=None` | unsupported | `events[i].tool_call.arguments`, per call (#1) |
| `ToolCallRecord.started_at`/`.finished_at` | `None` | unsupported | `events[].tool_call.*`, once (#3) |
| `agent_timeout` | `kind=timeout`, `outcome=reason`, rest in `extensions`; trace `outcome.status=timeout` | preserved | — loss #4 becomes representable |
| `oracle` record (§2.4) | `kind=oracle`, `role=oracle`, fields in `extensions` | preserved | — no `[oracle: …]` prefix, so no string matching |
| unrecognized `type` | `kind=unknown`, `source_type` verbatim, record in `extensions` | preserved | — every exporter skips these today |
| unrecognized extra field on a known record | `extensions` | preserved | — |
| per-event usage | `None` | unsupported | `events[].usage`, once (#6, #8) |
| agent version | `None` | unsupported | `agent.agent_version`, once (#7) |
| `stop_reason` | `None` | unsupported | `outcome.stop_reason`, once (#9) |
| non-object entry in the list | *(no IR event)* | dropped | `source[i]` |
| non-string where a string is expected | coerced with `str()` | normalized | `events[i].<field>` |
| status outside the ACP vocabulary | `unknown`, original in `extensions.source_status` | normalized | `events[i].tool_call.status` |

Two rows are deliberately absent. The converter does **not** prepend the
`prompts` argument as leading `user` events, though `acp_events_to_atif_steps`
and `acp_events_to_adp_content` both do: those steps are not ACP events, and
§5.2 records what they cost — an ATIF document that opens with two identical
`user` steps, so user turns over-count by one. A target that wants them adds
them at its own edge as `SYNTHESIZED`. And it reads no other artifact:
`result.json`, `timing.json` and the proxy capture are not consulted, so a value
that lives only there is a declared loss rather than a silent enrichment.

**FACT — measured on two real rollouts.** The `gemini` rollouts of §5.2,
converted through this path: H1 (5 events, 2 tool calls) produces **7** loss
records, H2 (4 events, 1 tool call, one real wall-clock timeout) produces **6**.
All `unsupported`. In both, 5 records are systemic and the rest are the
per-call `arguments`, so the report is `n_tool_calls + 5` and does not grow with
trace length.

#### IR → ATIF

**Implemented** (Slice D,
[`src/benchflow/trajectories/ir_to_atif.py`](../src/benchflow/trajectories/ir_to_atif.py)),
unwired: `export_atif.py` is untouched and still the only writer of
`trainer/atif.json`.

**FACT — the hub reproduces the direct exporter on everything measured.**
`ir_to_atif(acp_events_to_ir(events), prompts=P)` produces the same document as
`trajectory_to_atif_record(events=events, prompts=P)`, with one deliberate
exception (oracle, below). **Measured, not proved for all inputs**: the evidence
is the corpus named next, and no argument here establishes the property for
capture inputs outside it. Asserted by
`tests/trajectories/test_ir_to_atif.py` over events driven through the
production capture path and nine further shapes, and confirmed against the two
real rollouts of §5.2: for both, the document produced through the hub is
identical to the `trainer/atif.json` those rollouts actually wrote.

This is the evidence that the IR is sufficient for this format — a hub that lost
something the direct path carried would fail that equality.

| IR | ATIF | Class | Loss recorded |
|---|---|---|---|
| `events[].text` | `step.message` | preserved | — |
| `reasoning` / `reasoning_segments` | `reasoning_content`, joined by blank line | normalized | `events[].reasoning_segments`, once (#10) |
| `events[].index` | `step_id`, dense from 1 over emitted steps | normalized | `events[].index`, once |
| `tool_call.name` | `function_name` | preserved | — the ACP-kind semantics is what is lost, below |
| `tool_call.name_semantics` | *(no slot)* | dropped | `events[].tool_call.name_semantics`, once |
| `tool_call.call_id` empty or absent | `call_{n}` | **synthesized** | `events[i].tool_call.call_id` |
| `tool_call.name` empty or absent | `"tool"` | **synthesized** | `events[i].tool_call.name` |
| `tool_call.arguments = None` | `{}` | **synthesized** | `events[i].tool_call.arguments`, per call (#1) |
| `tool_call.arguments` present | passed through | preserved | — nothing declared |
| `tool_call.status` / `title` | `tool_calls[].extra`, stringified | normalized | — (shape unchanged from the direct path) |
| text content blocks | `observation.results[].content` | preserved | — |
| opaque content blocks | *(no slot)* | dropped | `events[i].tool_call.content` (#5) |
| `kind=timeout` | *(no slot)* | dropped | `events[i]` (#4) |
| `kind=unknown` | *(no slot)* | dropped | `events[i]` |
| text-empty event | *(no step)* | dropped | `events[i]` |
| `kind=oracle` | `source: "oracle"`, command as `message` | normalized | `events[i].extensions` — **deviation, see below** |
| `agent.agent_name` absent | `"unknown"` | **synthesized** | `agent.agent_name` |
| `agent.agent_version` absent | `"unknown"` | **synthesized** | `agent.agent_version` (#7) |
| `usage.input/output/cache_read` | `final_metrics.total_prompt/completion/cached_tokens` | preserved | — |
| `usage.cost_usd` | `final_metrics.total_cost_usd` | preserved | — |
| `usage.cache_creation_tokens`, `.reasoning_tokens`, `.total_tokens`, `.source`, `.price_source` | *(no slot)* | dropped | `usage.<field>` |
| `outcome.*` | *(no slot)* | dropped | `outcome`, once |
| `trace_id`, `started_at`, `finished_at`, `provenance`, `extensions` | *(no slot)* | dropped | once each |
| `agent.provider` | *(no slot)* | dropped | `agent.provider` |
| `events[].provenance`, `.source_type`, `.extensions`, `.outcome`, `.usage` | *(no slot)* | dropped | once each |
| `events[].started_at` / `.finished_at` | *(no slot)* | dropped | once each |
| `events[].tool_call.started_at` / `.finished_at` | *(no slot)* | dropped | once each, **under the tool call** |
| `events[].tool_call.content[].raw` on text blocks | *(only the rendered text survives)* | dropped | once |
| `events[].role`, when it disagrees with the source the kind implies | *(no slot)* | dropped | `events[i].role`, per event |
| *(the `prompts` argument)* | leading `user` steps | **synthesized** | `steps[i]`, **target space** |
| *(none)* | `steps[].message` on tool / flushed-thought steps | **synthesized** | `steps[].message`, **target space**, once |
| *(none)* | `final_metrics.total_steps` | **synthesized** | `final_metrics.total_steps`, **target space** |

**FACT — the one deliberate deviation is `oracle`.** `acp_events_to_atif_steps`
renders an oracle record as a `source: "agent"` step prefixed `[oracle: …]`,
recoverable only by string matching; §5.1 records this as a live divergence,
since the in-repo validator already accepts `source: "oracle"` while no emitter
produces it. The IR carries the role, so this edge emits `source: "oracle"` with
the command as the message and no prefix. A test asserts that this is the *only*
step that differs on a trajectory containing every capture event type plus an
oracle record.

**FACT — the outbound report describes the trace it received, not ACP's
habits.** A loss is declared only when the IR actually carries the value. An
ACP-derived trace has no per-event timestamps or usage, and the inbound report
already declared those absences `unsupported`; re-declaring them here would
count one fact twice and misdescribe an edge that loses nothing it was given.
The same conversion run over a trace that *does* carry them declares every one.
Both halves are asserted, the second as a complete set, and a companion test
derives the field list from the IR models themselves — so a field added to the
IR that ATIF cannot represent fails the suite until its fate is decided.

**FACT — measured on the same two real rollouts.** H1 produces **14** outbound
records (6 synthesized, 6 dropped, 2 normalized; 11 hub, 3 target); H2, whose
trajectory contains a real wall-clock timeout, produces **16** (5 synthesized,
9 dropped, 2 normalized), the extra drops being the timeout event, its
`extensions` and the trace `outcome`.

**FACT — the two reports compose.** For H1, `events[2].tool_call.arguments`
carries `unsupported` in the `acp -> ir` report and `synthesized` in the
`ir -> atif` one: the source never had arguments and the target demanded them
anyway, joined on one hub path (§8.2, choice 5).

**Cost.** `TraceUsage` carries `cost_usd` and `price_source`, so
`final_metrics.total_cost_usd` survives the hub.

**FACT — the writing path exists.** The LiteLLM callback log's per-entry `cost`
is summed into `Trajectory.metadata["cost_usd"]`
(`providers/litellm_logging.py:618-623`), surfaces as `AgentResult.cost_usd`
with `price_source: "litellm"` (`extract_usage_from_trajectory`), and is handed
to the ATIF writer by `rollout/_results.py:448`. The agent-native ACP path sets
it to `None` explicitly (`rollout/__init__.py:1737`).

**FACT — and it has not been observed producing a value here.** Every rollout
artifact on the machine this was developed on carries `cost_usd: null`,
including four whose `usage_source` is `provider_response` — they went through
the proxy, and their gateway log simply carried no per-entry cost, so
`price_source` stayed `None` too. So the field's *production* is established by
reading the code, not by observation, and parity with a cost is asserted on
synthetic input on both sides. Without the field, though, the hub would be
unable to carry a value the direct exporter's own API accepts — a gap in the
contract regardless of how often it is exercised.

`price_source` has no ATIF slot and is declared dropped: BenchFlow computes no
prices of its own, so a cost without the table that produced it is not
comparable. Per-call cost stays out of scope.

#### ATIF → IR

**Implemented** ([`ir_from_atif.py`](../src/benchflow/trajectories/ir_from_atif.py)),
unwired. Asserted by `tests/trajectories/test_ir_from_atif.py` against documents
from *both* writers — the direct exporter and the hub — plus malformed input.

This edge reads a document that is itself the output of a lossy conversion, and
it is built on one rule: **read what the document says, never what it probably
meant.** Several values in an ATIF document were fabricated by the converter that
wrote it, and nothing in the document marks them as such. Reading them back as
absences would be guessing which ones were invented — a guess that happens to be
right is still a guess, and it would make the round trip below report a
preservation that did not happen.

| ATIF | IR | Class | Loss recorded |
|---|---|---|---|
| `session_id` | `session_id` | preserved | — |
| `schema_version` | `extensions.schema_version` | preserved | — no IR field names a source format's version |
| unknown document key | `extensions` | preserved | — |
| `agent.name` / `.version` / `.model_name` | `agent.agent_name` / `.agent_version` / `.model` | preserved | — **including the literal `"unknown"`**, see below |
| step order | `events[].index`, dense from 0 | preserved | — |
| `step.source` | `role` **and** `source_type` verbatim | preserved | — |
| `step.source` outside the vocabulary | `kind=unknown`, `source_type` verbatim | preserved | — every exporter drops these today |
| `step.message` | `text`, **including `""`** | preserved | — |
| `step.reasoning_content` | `reasoning` + `reasoning_segments=[joined]` | normalized | — one step is one segment; the boundaries are already gone |
| `step_id` | `events[].extensions.step_id` | preserved | — not mapped onto `index`, which invariant 2 makes a dense position |
| `tool_calls[].tool_call_id` | `tool_call.call_id` | preserved | — |
| `tool_calls[].function_name` | `tool_call.name` + `name_semantics="function_name"` | preserved | — the ACP-kind semantics is not in the document to recover |
| `tool_calls[].arguments` | `arguments`, **verbatim including `{}`** | preserved | — see below |
| `tool_calls[].extra.title` / `.status` | `title` / `status` | preserved | — |
| `.extra.status` outside the vocabulary | `unknown`, original in `extensions` | normalized | `events[i].tool_call.status` |
| `observation.results[].content` | `ContentBlock(kind=text)`, `raw=None` | preserved | — the source block is not in the document |
| a result matching no call in the step | `extensions.unmatched_observation_results` | normalized | `steps[i].observation.results`, **source space** |
| `step.metrics` | `extensions.metrics`, uninterpreted | normalized | `events[i].usage` — no ATIF schema is vendored here to map it |
| a step with *n* tool calls | *n* IR events | normalized | `steps[i]`, **source space** |
| `final_metrics.total_prompt/completion/cached_tokens`, `total_cost_usd` | `usage.input/output/cache_read_tokens`, `.cost_usd` | preserved | — |
| `final_metrics.total_steps` | *(no IR field)* | dropped | `final_metrics.total_steps`, **source space** |
| other `final_metrics` keys | *(no IR field)* | dropped | `final_metrics`, **source space** |
| a non-object step | *(no IR event)* | dropped | `steps[i]`, **source space** |
| non-string where ATIF specifies a string | coerced with `str()` | normalized | the field, hub space |
| *(none)* | `trace_id`, `started_at`, `finished_at`, `agent.provider`, `outcome`, `events[].started_at`/`.finished_at`/`.outcome`/`.usage`, `events[].tool_call.started_at`/`.finished_at`/`.content[].raw`, the five unmapped `usage` fields | **unsupported** | one record each |

**Everything ATIF does not carry is `UNSUPPORTED`, never `DROPPED`.** There is
nothing in the document to drop, and the distinction is what says where a fix
would have to land — the same line `ACP → IR` draws, which is what makes the two
inbound reports comparable. Every `DROPPED` record in this direction addresses
the source document, in the source path space.

**Three values are deliberately read as observed, and this is the whole point of
the edge.** `agent.version: "unknown"` is what `ir_to_atif` writes when the trace
has no version *and* a legal observed value; `arguments: {}` is what it writes
for every ACP-derived call; `message: ""` is what it writes on a step carrying
only a tool call. All three come back as observations. §8.10 measures what that
costs.

**One shape is deliberately not undone.** A step with both `message` and
`reasoning_content` becomes **one** event carrying both, not a reasoning event
followed by a message event: the blank-line join that produced it is not
injective (§5 loss #10), so splitting it would invent a boundary. The fusion is
reported by the event count instead.

#### OTLP/JSON → IR

**Implemented** ([`ir_from_otel.py`](../src/benchflow/trajectories/ir_from_otel.py)),
unwired, inbound only. Asserted by `tests/trajectories/test_ir_from_otel.py`
against a payload produced by `opentelemetry-proto` 1.41.1 itself — see §8.11 —
plus documents no conformant producer writes.

The mapping is written against the two artifacts §4.2 names, at the versions
`uv.lock` pins, and the module records both in `OTLP_PROTO_VERSION` and
`SEMCONV_VERSION`. It takes **no OTel dependency**: it reads JSON dictionaries,
so `uv.lock` is untouched. Nothing here is written against the published
specification text, which is not vendored.

The rule the edge is built on: **a span is evidence of an operation, not a
statement about an agent.** OTLP is a general tracing format; it says what was
instrumented, when, and under what identifiers, and it does not say who spoke or
what a turn was. So exactly one span shape maps onto a typed IR kind — the one
the pinned vocabulary defines as a tool execution — and everything else becomes
`UNKNOWN` with its whole content carried. What a plausible reading would have
added instead is in §8.11 as an open decision.

| OTLP | IR | Class | Loss recorded |
|---|---|---|---|
| span order in the payload | `events[].index`, dense from 0 | preserved | — **never sorted by time**; see below |
| `span.traceId` | `trace_id`, **verbatim, never re-encoded** | preserved | — |
| `span.spanId` / `.parentSpanId` / `.traceState` / `.flags` / `.kind` | `events[].extensions.otel.span`, verbatim | preserved | — the IR models no parent link |
| `span.name` | `source_type` | preserved *(value)* | `events[].source_type` — the **value** is carried verbatim; what is unsupported is the absent-versus-empty *distinction*, because the field has no presence |
| `span.attributes[]` | `extensions.otel.attributes`, decoded to a map | **normalized** | `events[].extensions` — values are preserved, the wire form is not; see below |
| `span.events[]` / `span.links[]` | `extensions.otel.span`, verbatim | preserved | — |
| `span.status` | `extensions.otel.span.status`, verbatim | normalized | `events[i].outcome` — the IR slot is free text with no vocabulary a status code maps into |
| `span.startTimeUnixNano` / `.endTimeUnixNano` | `events[].started_at` / `.finished_at` | preserved | — unless not a whole microsecond, then normalized (see below) |
| `resource` / `scope`, with schema URLs | `extensions.otel`, per event | preserved | — per event, not per trace: one payload mixes them |
| envelope position (`resourceSpans[i].scopeSpans[j].spans[k]`) | `extensions.otel.envelope` | preserved | — absent when the caller did not read the span out of a payload |
| `scope.name` | `events[].provenance.producer` | preserved | — |
| `span.dropped*Count > 0` | *(nothing to carry)* | **unsupported** | `events[i].extensions` — the SDK discarded it before export |
| `gen_ai.operation.name == "execute_tool"` | `kind = tool_call` | preserved | — the only typed mapping |
| any other span | `kind = unknown`, everything carried | preserved | — |
| `gen_ai.tool.name` | `tool_call.name` + `name_semantics="gen_ai.tool.name"` | preserved | — |
| `gen_ai.tool.call.id` | `tool_call.call_id` | preserved | `events[i].tool_call.call_id` when absent — **no id is synthesized** |
| `gen_ai.tool.call.arguments` (kvlist) | `tool_call.arguments`, **including `{}`** | preserved | — |
| `gen_ai.tool.call.arguments` (JSON string) | *(kept in the attribute map)* | normalized | `events[i].tool_call.arguments` — **not parsed** |
| `gen_ai.tool.call.result` (string / object) | `ContentBlock` `text` / `opaque` | preserved | `events[i].tool_call.content[0].raw` for a text block (the attribute *is* the text) · `…content[0].text` for an object block (the document holds no rendering) · `…tool_call.content` when the attribute is absent |
| the span's own extent | `tool_call.started_at` / `.finished_at` | preserved | `events[i].tool_call.started_at` / `.finished_at` when the span carries no readable instant, or when nanoseconds are truncated — the first inbound edge that can *fill* these (§5 loss #3), and it declares them at their own path when it cannot |
| `gen_ai.usage.input_tokens` / `.output_tokens` / `.cache_read.input_tokens` / `.cache_creation.input_tokens` | `events[].usage.*`, `source="otel_gen_ai_usage"` | preserved | — |
| `gen_ai.usage.prompt_tokens` / `.completion_tokens` | the same fields | normalized | `events[i].usage.*` — read from a spelling the pinned package marks deprecated |
| `gen_ai.agent.name` / `.version`, `gen_ai.request.model`, `gen_ai.provider.name` | `agent.*` | preserved | — |
| `gen_ai.system` | `agent.provider` | normalized | `agent.provider` — deprecated spelling |
| spans disagreeing on an `agent.*` value | the first is kept | dropped | `agent.<field>` |
| `gen_ai.input.messages` / `.output.messages` / `gen_ai.prompt` / `.completion` | `extensions.otel.attributes` | normalized | `events[i].text` — **not read as text** |
| *(none)* | `session_id`, `started_at`, `finished_at`, `outcome`, `usage`, `events[].role`, `.reasoning`, `.reasoning_segments`, `.source_type` (presence), `events[].tool_call.title`, the usage fields with no attribute | **unsupported** / **normalized** | one record each; see below |
| a non-object entry at any envelope level | *(no span)* | dropped | `resourceSpans[i]…`, **source space** |
| a non-string `traceId` | *(not read)* | dropped | `spans[i].traceId`, **source space** |

**Order is preserved; causality is not inferred from it.** Document order is
kept exactly and spans are never sorted by start time — siblings overlap, a
batch may omit a parent, and two spans can share a start instant. The real
structure is the `parentSpanId` edge set, and it is preserved per span. Because
the IR has no parent field, it lives in `extensions`; adding one is a change to
the hub and therefore a maintainer's decision (§8.11).

**Two losses are structural and unavoidable, and both are declared.**
`datetime` resolves to microseconds while OTLP counts nanoseconds, so a
timestamp that is not a whole microsecond is truncated in the IR field — the
exact integer stays in `extensions.otel.span`, so the value survives, but the
canonical field no longer holds it. And OTLP models `name`, the timestamps,
`parentSpanId`, `kind` and the `dropped*Count`s as protobuf scalars **without
presence**, so an unset field and one holding the default are the same document:
the IR's absent-versus-empty distinction is simply not recoverable for them.

**Semantic preservation is not wire preservation, and the table says which one
it means.** Attribute values are decoded out of their `AnyValue` wrappers into a
plain map, and the canonical protobuf JSON mapping writes an `int64` as a
*string* while much of the ecosystem writes it as a number. `{"intValue": "7"}`
and `{"intValue": 7}` are therefore the same `7` afterwards, and the wrapper
type is gone with them. No value is lost, so `attributes_raw` is *not* kept —
that copy exists for the cases where a value would be, listed above — but the
two payloads are indistinguishable, and an `IR → OTel` emitter could not
reproduce either byte-for-byte. Declared once per conversion at
`events[].extensions`, because it is a property of the decoding rather than of
any one span.

**The envelope partition is preserved as coordinates, not as structure.** One
payload nests spans under `resourceSpans[] → scopeSpans[] → spans[]`, and
reading it into one list of events flattens that: two spans the producer batched
under *different* `ScopeSpans` objects that happen to carry an equal `scope`
would otherwise become indistinguishable. `extensions.otel.envelope` keeps the
three indices, so the grouping stays reconstructible without the IR growing a
concept of an envelope. Spans a caller assembled itself carry no coordinates —
there is no envelope position to record, and inventing one would claim a payload
that never existed.

**Trace-level absences carry the class that says where a fix would land.**
`outcome`, `session_id` and `events[].role` are `UNSUPPORTED` — OTLP has nothing
to read. `started_at`, `finished_at` and `usage` are `NORMALIZED` when spans
carry the information per span: the values are preserved on the events, and
deriving a run extent or a token total from them would assume this payload holds
the whole run, which a batch does not promise. When no span carries them at all,
the same fields become `UNSUPPORTED` instead. Two different reasons for the same
empty field, kept apart.

#### IR → ACP capture events

**Implemented** ([`ir_to_acp.py`](../src/benchflow/trajectories/ir_to_acp.py)),
unwired, in memory only.

**The target is the ACP-session capture event format, not the
`acp_trajectory.jsonl` artifact.** §2.1 and §2.4 are the reason: the file holds
records from four sources — the ACP-session emitter, `_run_oracle`,
`hosted_env`, and whatever a session-factory `Session` puts in `steps` — plus a
fifth family the viewer reads and nothing here writes. **No artifact-level
contract exists**, so there is nothing for an edge to target at that level. The
capture format *is* defined, by the §2.1 schema, and that is what this edge
writes. Whether the artifact should acquire a contract is untouched by this
slice.

| IR | ACP capture | Class | Loss recorded |
|---|---|---|---|
| `USER_MESSAGE` / `AGENT_MESSAGE` / `AGENT_REASONING` + text | `text_event` with the matching `type` | preserved | — including an observed `""` |
| `TOOL_CALL` with every required value | `tool_call_event` | preserved | — |
| `TIMEOUT` with `outcome == "wall_clock_timeout"` and its three extension keys | `agent_timeout_event` | preserved | — recovered by name from `extensions` |
| `tool_call.title` absent | `""` | **synthesized** | `events[i].tool_call.title` — the contract's own representation of an absent title |
| `content[].raw` | the block, verbatim | preserved | `…content[i].kind` — the IR's text/opaque classification is not part of the wire shape |
| several `reasoning_segments` | one `agent_thought` | dropped | `events[i].reasoning_segments` — §5 loss #10, in the writing direction |
| `arguments`, `name_semantics`, per-event and per-call timestamps, `role`, `source_type`, `usage`, `extensions` | *(no slot)* | dropped | one record each, when the trace carries them |
| `trace_id`, `session_id`, `started_at`, `finished_at`, `agent.*`, `usage`, `outcome.*`, trace `extensions` | *(no envelope at all)* | dropped | one record each |

**Five things make an event unrepresentable, and each refuses rather than
invents.** A tool call with no ACP `status`, no `tool_call_id` or no `kind`; a
content block with no source block to write; a text event with no text; a
timeout missing or mistyping any of its four required values; and an `ORACLE` or
`UNKNOWN` event, which have no record shape in the contract at all.

**`status` is the one that matters.** The ACP vocabulary is closed and every
member asserts a lifecycle state, so there is no neutral value and the IR's
`ToolStatus.UNKNOWN` has no counterpart. A synthesized status would reach the
viewer and be displayed to a person as an observation. It is never written.

**Fail closed.** A trace with one unrepresentable event yields **no** events —
`AcpCaptureNotRepresentable`, a `ValueError` carrying the blocking
`LossRecord`s. The target has no envelope, no count and no marker for "some
events are missing", so a partial list is indistinguishable from a complete one
the moment it leaves the converter.

**Representability is decided by the data, never by the provenance.** An
OTel-derived trace that carries every required value exports; an ACP-derived
trace missing one does not. `Provenance` is not read, and a test asserts the
module never references it. Today ACP-derived traces usually export and
OTel-derived ones usually do not — that is an observation about what those edges
currently carry, not a rule.

**`name_semantics` is part of that data, and it gates the `kind` slot.** ACP's
`kind` is a *category* — `ToolKind` calls itself "Category tag for tool calls",
`_canonical_tool_kind` defaults an absent one to `other`, and the production
values (`execute`, `edit`, `fetch`, `think`) are categories. ATIF's
`function_name` and OTel's `gen_ai.tool.name` name *particular tools*. Writing
one of those into `kind` would not be a normalization a reader could undo; it
would assert a category nobody observed. So only `name_semantics == "acp_kind"`
may be written, and **the string is never inspected** — a `function_name` of
`read` coincides with a real `ToolKind` member and is refused anyway.

**Why `content[].raw` is required, and what it costs.** The schema leaves
`content_block` permissive (`type: object`), so this is **not** a constraint the
contract imposes — it is a conservative choice not to invent wire structure. ACP
stores `session/update.content` exactly as the wire delivered it, and two shapes
are known to be consumed: the nested `{"type": "content", "content": {...}}` and
the flat `{"text": …}`. A block that carries rendered text but no source block
gives no way to choose between them.

The cost is concrete and worth stating plainly: **`ATIF → IR` and `OTLP → IR`
both produce content blocks with `text` and no `raw`, so a tool result that did
not come from ACP is normally not representable today.** Relaxing this would
require defining a canonical ACP content-block construction policy — which shape
to write, and on whose authority. **No such policy is defined here**, and adding
one is a decision rather than an implementation detail.

**What the round trip has actually been measured on.** `ACP → IR → ACP`
reproduces its input with **structural equality** on three corpora: the two
captured rollouts of §5.2 — H1, five records, and H2, four records ending in a
real wall-clock timeout — and the suite's conformant fixture of five records
together with its five drop-one subsets. **Key-order preservation** and **schema
validity** are checked separately, the second over six further shapes. Byte
equality through `redact_acp_trajectory_jsonl` also holds, but adds nothing
beyond key order: the serializer redacts both sides, so redaction cancels.

**No universality is claimed or demonstrated.** There is no property test, and
three corpora are three corpora: nothing here establishes the behaviour for
conformant inputs outside them. What *is* structural is the alternative — the
edge either reproduces its input or refuses, so an unmeasured input cannot
silently produce a degraded record. It produces none.

#### Human verification of the ACP edge

**HUMAN E2E VERIFIED, 2026-08-18, G1–G10 all pass**, against
`e2e-g/PROCEDURE.md`. Every step was carried out and judged by a person rather
than by the suite.

| | what was verified |
|---|---|
| G1 | H1 and H2 are real BenchFlow rollouts — `result.json` carries `trajectory_source: "acp"`, the typed lineage of `models.py:17`, and each record's keys are the emitter's literal writes |
| G2 | `ACP → IR → ACP` read event by event: structural equality **and** key-order preservation on H1 (5 records) and H2 (4 records) |
| G3 | every loss path resolves in the canonical encoding; **no unexpected synthesis**; `name_semantics: "acp_kind"` recorded `NORMALIZED`, not `DROPPED` |
| G4 | the laundering path is closed — a `function_name` is refused even when its value spells `read`, `write`, `bash` or another real `ToolKind` member; the OTel fixture shows 8 blockers in four independent kinds |
| G5 | fail-closed atomicity: the blocker is on the last event and **no partial prefix** is returned |
| G6 | `title=None` → `""` with `SYNTHESIZED`; an observed empty title synthesizes nothing; `tool_call_id=None` is refused |
| G7 | representability and output are independent of provenance, and an ACP-derived trace with an `UNKNOWN` status is still refused |
| G8 | 12 emitted records validated, 0 errors, and all four negative controls rejected |
| G9 | 27/27 mutations caught, no `MISSED`, no `SKIP`, exit 0; the files restored, proven by a green `tests/trajectories` |
| G10 | no importer outside the IR family, no wiring, no file I/O; `uv.lock`, `ir.py`, the schema and the viewer unchanged |

**The verification found two defects, both in the procedure, neither in the
converter.** G4 described the OTel fixture's blockers as three independent kinds
when there are four, and the inspector's G7 header said "two traces" while
printing four. Both are corrected in the scaffolding, which lives outside the
repository and is not part of this change.

**What the sign-off does not extend to.** Each of these is unchanged by it:

- **Not** that every conformant ACP document round-trips. The measured corpora
  are H1, H2 and the suite fixture with its subsets.
- **Not** that the `acp_trajectory.jsonl` **artifact** is supported. The target
  is the capture *event format* of §2.1; that file also holds oracle records,
  `hosted_env` records and session-factory `steps`, and §2.4 records that no
  artifact-level contract exists.
- **Not** that `ORACLE` or `UNKNOWN` events are exportable — they have no record
  shape in the contract and are refused.
- **Not** that ATIF or OTel tool results are generally exportable. Their content
  blocks carry `text` and no `raw`, and their names are not ACP kinds.
- **Not** that `OTel → Viewer` works. The OTel fixture is four independent kinds
  of refusal away from exporting, and nothing here changes that.
- **Not** that anything is wired. There is no runtime path, and this edge writes
  no file — it returns records.

#### IR → OpenTelemetry

**Not implemented, and deliberately not sketched further.** The inbound edge
above establishes what OTLP can be read *as*; writing spans raises a different
and larger set of questions — what a span tree for a BenchFlow rollout should
look like, which ids to mint, which encoding to write them in, and whether
emitting `gen_ai.*` attributes commits the project to an `_incubating`
vocabulary. Those are in §8.11.

**This is a deferral we chose, not one we were given.** §8.12 records it as a
scope decision with its reasoning and its reversal conditions, so a reader can
tell it apart from the maintainer decision that is still outstanding.

### 8.4 A worked example

**FACT.** The document below is produced by the models in `ir.py` and compared
against this block by `test_the_documented_example_matches_the_models`, so it
cannot drift from what the code emits. It is written in the **canonical
encoding** — nulls retained (§8.2). An earlier revision published it with
`exclude_none=True`, which dropped `arguments` from the document while the loss
report kept addressing it; the two together are the point of the example, so
they are now shown together.

It shows the four properties the design exists for: an unavailable value that is
`null` *and* declared (`arguments`, beside its `LossRecord`), a thought whose
boundaries survive next to the joined form, a non-text content block carried as
`opaque` instead of skipped, and a timeout that is representable at all — with
its source-specific fields in `extensions` rather than as four new IR fields.

<!-- ir-example -->
```json
{
  "ir_version": "bf-trace-ir-v0",
  "trace_id": null,
  "session_id": "rollout-7f3a",
  "agent": {
    "agent_name": "gemini",
    "agent_version": null,
    "model": "gemini-2.5-flash",
    "provider": null
  },
  "started_at": null,
  "finished_at": null,
  "events": [
    {
      "index": 0,
      "kind": "user_message",
      "source_type": "user_message",
      "role": "user",
      "text": "Count the rows in data.csv",
      "reasoning": null,
      "reasoning_segments": null,
      "tool_call": null,
      "started_at": null,
      "finished_at": null,
      "outcome": null,
      "usage": null,
      "provenance": {
        "source_format": "acp-capture-v1",
        "producer": "_events_to_trajectory",
        "captured_at": null
      },
      "extensions": {}
    },
    {
      "index": 1,
      "kind": "tool_call",
      "source_type": "tool_call",
      "role": "agent",
      "text": null,
      "reasoning": "Check the file first.\n\nThen count.",
      "reasoning_segments": [
        "Check the file first.",
        "Then count."
      ],
      "tool_call": {
        "call_id": "tc_1",
        "name": "execute",
        "name_semantics": "acp_kind",
        "title": "wc -l data.csv",
        "status": "completed",
        "arguments": null,
        "content": [
          {
            "kind": "text",
            "text": "42 data.csv",
            "raw": {
              "type": "content",
              "content": {
                "type": "text",
                "text": "42 data.csv"
              }
            }
          },
          {
            "kind": "opaque",
            "text": null,
            "raw": {
              "type": "diff",
              "path": "/w/data.csv",
              "oldText": "a",
              "newText": "b"
            }
          }
        ],
        "started_at": null,
        "finished_at": null
      },
      "started_at": null,
      "finished_at": null,
      "outcome": null,
      "usage": null,
      "provenance": {
        "source_format": "acp-capture-v1",
        "producer": "_events_to_trajectory",
        "captured_at": null
      },
      "extensions": {}
    },
    {
      "index": 2,
      "kind": "timeout",
      "source_type": "agent_timeout",
      "role": null,
      "text": null,
      "reasoning": null,
      "reasoning_segments": null,
      "tool_call": null,
      "started_at": null,
      "finished_at": null,
      "outcome": "wall_clock_timeout",
      "usage": null,
      "provenance": {
        "source_format": "acp-capture-v1",
        "producer": "_events_to_trajectory",
        "captured_at": null
      },
      "extensions": {
        "timeout_sec": 90.0,
        "pending_tool_call_ids": [],
        "terminal_trajectory_complete": true
      }
    }
  ],
  "usage": {
    "input_tokens": 1180,
    "output_tokens": 96,
    "cache_read_tokens": null,
    "cache_creation_tokens": null,
    "reasoning_tokens": null,
    "total_tokens": 1276,
    "source": "llm_proxy_normalized",
    "cost_usd": null,
    "price_source": null
  },
  "outcome": {
    "status": "timeout",
    "stop_reason": null,
    "reward": null,
    "error_category": null
  },
  "provenance": {
    "source_format": "acp-capture-v1",
    "producer": "_events_to_trajectory",
    "captured_at": null
  },
  "extensions": {},
  "losses": {
    "direction": "acp->ir",
    "ir_version": "bf-trace-ir-v0",
    "records": [
      {
        "field": "events[1].tool_call.arguments",
        "space": "hub",
        "loss_class": "unsupported",
        "detail": "ACPSession.handle_update reads five fields and rawInput is not one of them, so no ACP-derived tool call carries arguments.",
        "doc_ref": "\u00a75 loss #1"
      },
      {
        "field": "events[1].tool_call.started_at",
        "space": "hub",
        "loss_class": "unsupported",
        "detail": "ToolCallRecord tracks started_at/finished_at in memory and _events_to_trajectory serializes neither.",
        "doc_ref": "\u00a75 loss #3"
      }
    ]
  }
}
```

Read it beside the `losses.records` at the bottom: every `LossRecord.field` is a
path into this same document, and following it lands on a key that is present
and `null`. That resolvability is what `exclude_none` broke, and what
`test_every_concrete_loss_path_resolves_in_the_canonical_encoding` now guards.

### 8.5 Invariants

**PROPOSAL.** Checked by `validate_trace`, which returns one string per
violation rather than raising, and asserted in
`tests/trajectories/test_trace_ir.py` with both a violating and a clean trace
for each:

1. `ir_version` equals `bf-trace-ir-v0` — v0 defines no migration path.
2. `events[i].index == i` — dense and ordered, so a dropped event leaves a hole
   rather than disappearing.
3. `kind == tool_call` if and only if a `tool_call` payload is present.
4. A `text` block carries text; an `opaque` block carries `raw`. An opaque block
   with no payload would be a dropped block claiming preservation.
5. When both are present, `"\n\n".join(reasoning_segments) == reasoning` — the
   segments are a strictly richer encoding of the same value, never a second
   divergent one.
6. An `agent_reasoning` event carries reasoning in one of the two fields.
7. **Every `arguments is None` has a matching loss record.** Absence is declared,
   never silent.
8. A `user_message` is attributed to `user` or to nobody. Agent-side attribution
   is genuinely ambiguous today (§5.1, the `oracle` divergence) and the IR does
   not pretend to settle it.

Three further properties are pinned by tests rather than by `validate_trace`:
the IR's tool-status vocabulary is a superset of the ACP `ToolCallStatus` enum,
read off the enum itself; every event type `_events_to_trajectory` emits maps to
an `EventKind`, with the producer's vocabulary read from source by AST — the
same mechanism the Slice A conformance suite uses (§2.2); and **every concrete
`LossRecord.field` resolves to a key that is present in the canonical encoding**
(§8.2, choice 4). The last one asserts in the same test that the discarded
`exclude_none` encoding *fails* to resolve those paths, so it cannot pass for
both encodings at once.

### 8.6 What is provisional, and what a review can still change

Everything in §8 is provisional. In particular, a review can reject any of the
following without any other work having to be undone, because nothing depends on
them:

- **the hub itself** — if direct converters are preferred (open question 1), the
  module is deleted and the loss taxonomy survives as documentation;
- **the tri-state / declared-absence contract** — the strongest opinion here,
  and the one most likely to feel heavy in a converter;
- **`name_semantics` and `reasoning_segments`** — both exist to preserve a
  distinction the current exporters discard; if that distinction is not wanted,
  both fields go and losses #10 and the `kind`→`function_name` normalization
  stay as they are;
- **`extensions` as the escape hatch** — the alternative is a field per source
  quirk, which is how a pragmatic superset becomes a spec;
- **`TraceUsage.source`** — it exists only because open question 4 is open. If a
  canonical definition of `input_tokens` is chosen, the field can go;
- **the version string and the absence of migration machinery**;
- **the module's location** (`benchflow.trajectories.ir`) and every name in it.

What a review cannot change by rejecting the IR: the losses in §5 are properties
of the current code, not of this proposal, and they remain whatever happens
here.

### 8.7 Status

- **Implemented:** the IR types, the loss model with its path spaces, the
  invariants, the validation suite, this section, six converters — `ACP → IR`,
  `IR → ATIF`, `ATIF → IR`, `OTLP/JSON → IR`, `IR → ACP capture events`,
  `IR → viewer trace steps` (§8.14) — each with its loss report; the page edge
  that renders those steps through the current viewer's own cards (§8.15); the
  round-trip
  measurement over the ATIF pair (§8.10), and the ACP pair's round-trip evidence
  in §8.3 and its test suite; and the loss-bounded conformance gate that joins
  the two halves (§8.13). There is no
  §8.10-style measurement for the ACP pair: `ACP → IR → ACP` either reproduces
  its input exactly or refuses, so there is no partial preservation to quantify.
- **Deferred by a decision of ours, reversibly:** `IR → OTel` — see §8.12 for
  the reasoning and the conditions that end the deferral. The OTel direction is
  *implemented inbound and deferred outbound*, which is one state and not two
  half-answers.
- **Wired, opt-in and reversibly:** one branch in `render_rollout`, off unless
  `BENCHFLOW_VIEWER_TRACE_IR` is set, with a lazy import inside it (§8.15).
  That is the family's only reach into a run path.
- **Not implemented:** any on-disk artifact, any capture-layer enrichment, any
  change to a default code path.
- **Unchanged:** every existing format, exporter, artifact and code path.
  `export_atif.py` in particular is untouched and remains the only writer of
  `trainer/atif.json`.

The isolation property is unchanged in substance and restated at a new boundary:
`ir.py`, `ir_from_acp.py`, `ir_to_atif.py`, `ir_from_atif.py`,
`ir_from_otel.py`, `ir_to_acp.py`, `_otlp_anyvalue.py`, `ir_round_trip.py`,
`ir_conformance.py`, `ir_to_view.py` and `ir_to_view_html.py` form a closed
family that may import each other, and
`test_only_the_ir_family_imports_the_ir` asserts that nothing else in
`src/benchflow` imports any of them **except the one site named in
`WIRING_SITES`** — `viewer.py`, whose opt-in branch is described in §8.15.

`ir_to_view_html` is the one member that imports a runtime module — it renders
through `viewer`'s card builders rather than copying their markup — and the
dependency runs one way only: no part of `viewer` imports any part of the
family.

Every converter reads or writes its own format **as data**, never through the
module that already handles it: `ir_to_atif` does not import `export_atif` and
pins the shared schema version by test instead, and `ir_from_otel` imports no
`opentelemetry` package at all. Each imports the IR, and `ir_from_otel` also
imports `_otlp_anyvalue` — the decoder split out of it, which knows nothing
about the hub and is the only family member that imports none of the others.

### 8.8 What the first converter showed

**FACT.** Writing `ACP → IR` — and then reading a converted real rollout by hand
— was the first stress test of §8.2's declared-absence rule. Three things came
out of it that a design document could not have settled:

- **The rule is affordable, because the report is bounded by tool calls rather
  than by trace length.** Systemic absences — timestamps, per-event usage, agent
  version, stop reason — are declared once each under an unindexed
  `events[].…` path; only `arguments`, which `validate_trace` requires per
  event, scales. A 50-tool-call trace declares 55 records that carry one
  sentence of distinct information between them, which is why the converter
  ships `loss_summary`.
- **The per-event requirement is the part to review.** It is what makes an
  undeclared absence a test failure rather than a habit, and it is also the
  reason 50 records say the same thing. An `events[*].…` wildcard would collapse
  them at the cost of making a single missed call invisible. That trade is open;
  §8.6 already lists this contract as the most likely thing to change.
- **A loss report addressed by path constrains the encoding, which the design
  had not noticed.** Reading a converted rollout by hand — not running the test
  suite, which was green and self-consistent — showed the §8.4 example published
  with `exclude_none=True`, its loss records pointing at keys the document did
  not contain. The rule in §8.2 choice 4 and the guard in §8.5 are the result.
  Applying that guard immediately found a second instance of the same class:
  `outcome.stop_reason` could not resolve in any trace that did not time out,
  because the section itself was `None`, which is why `outcome` is now always
  present.

**Worth stating plainly, because it is a limit and not a fix.** The invariant
forces a converter to *declare* an absence; it cannot stop one from writing
`arguments: {}` instead of `null`. A trace with a fabricated empty argument map
and no loss record is valid. Verified by hand. Closing that would mean the IR
taking a position on what an empty map means for each source, which §8.2's
tri-state rule deliberately leaves to the converter.

### 8.9 What the first outbound converter showed

**FACT.** `IR → ATIF` was the first edge that had to *fabricate* rather than
declare an absence, and it settled three things the inbound edge could not:

- **`SYNTHESIZED` earns its place.** Seven values ATIF requires and the IR does
  not carry are now produced and recorded — an agent version, a tool-call id, a
  function name, an empty argument map, an empty step message, a step count, and
  the prompt-derived steps. Without the class, each would be an invented value
  indistinguishable from an observed one, which is exactly how the `{}` in
  today's documents reads.
- **The hub is sufficient for this format, demonstrably.** Parity with the
  direct exporter holds byte-for-byte on real captured rollouts, so the
  round trip through the IR costs nothing that ATIF was previously getting.
- **The report's ownership had to be settled**, and target-only values forced
  `PathSpace` into existence (§8.2, choice 5). A prompt-derived step has no IR
  antecedent at all, so it cannot be addressed in the hub vocabulary — and
  before this edge, `LossRecord` had no way to say so.

### 8.10 What the round trip measures

With both ATIF edges implemented the loop `ACP → IR → ATIF → IR′` closes, and
the question it answers is a harder one than parity: **how much of a trace is
still there after a trip through the interchange format?**
[`ir_round_trip.py`](../src/benchflow/trajectories/ir_round_trip.py) answers it
as a measurement. It compares the two *traces* and never reads the loss reports,
so a converter that lost something without declaring it is caught rather than
confirmed.

#### Why not a percentage

A single number would merge two things that have nothing to do with each other,
so the report crosses an **observed** axis with a **declared** one:

| observed (`RoundTripOutcome`) | meaning |
|---|---|
| `preserved` | same values, same count |
| `transformed` | values on both sides, not the same ones — `matched=0` means *replaced* |
| `lost` | values in, none out |
| `fabricated` | none in, values out |

| declared (`Representability`) | meaning |
|---|---|
| `representable` | ATIF has a slot — a loss here is a gap in **our** edge, and fixable |
| `non_representable` | ATIF has no slot — a loss here is a cost of the **format** |

The declared half is a table with one entry per IR field, and a test derives the
field list from the models, so a new IR field cannot reach the round trip
without a disposition. Comparison is by canonical path — `events[0].text` and
`events[7].text` are one path with two values — because after a conversion that
fuses and drops events there is no recoverable correspondence between event *i*
and event *j*, and inventing an alignment would be the same guessing §8.3's
inbound edge refuses.

#### Measured on the two real rollouts of §5.2

**FACT — measured by the harness, and verified by hand.** The rollouts are real
captures; the loop was run over their `acp_trajectory.jsonl`, and its output was
then checked against the raw artifacts by a person: document parity, the
conversion read step by step in both directions, each fabricated value confirmed
present in `trainer/atif.json` and absent from the capture, each unrepresentable
value confirmed absent from the document and present in the capture, and a
negative control that corrupts a value and confirms the comparison notices.
`with prompts` reproduces what a real export does, and its document is
byte-identical to the `trainer/atif.json` each rollout actually wrote.

The limits of that verification are in the closing paragraph of this section and
are not narrowed by it.

| | H1 fields | H2 fields | H1 values | H2 values |
|---|---|---|---|---|
| `preserved` | 15 | 15 | 35 of 46 | 25 of 37 |
| `transformed` | 5 | 5 | | |
| `lost` (fixable) | **0** | **0** | | |
| `non_representable` | 1 | 6 | | |
| `fabricated` | 4 | 4 | | |

H1 is a 5-event tool-use rollout; H2 is 4 events ending in a real wall-clock
timeout. Three results, in the order they matter:

- **Within the declared field mapping, nothing representable is lost.** Every
  value the loop drops from these rollouts is dropped because ATIF has nowhere
  to put it, so the remaining loss is a property of the format and not of this
  implementation. `test_nothing_representable_is_lost_on_a_captured_rollout`
  pins it, and a failure there would name a converter bug.

  **This is a result, not a definition**, and the distinction matters because
  the table declaring what is representable is written by hand: a field wrongly
  marked unrepresentable would move a real, fixable loss into the format's
  column and shrink `lost` to zero for free. Two tests close that. On a trace
  populating *every* IR field, `lost` is **2** — `ir_to_atif` writes no per-step
  metrics, so per-event usage is lost through a slot ATIF actually has — and no
  field the table calls unrepresentable comes back with any value intact. So
  `lost = 0` on these rollouts says something about them, not about the table.
- **The trace comes back with *more* values than it left with** — 46 in, 56 out
  for H1 — while having lost information. Four fields are fabricated on every
  run: `agent.agent_version` (`"unknown"`), `events[].tool_call.arguments`
  (`{}`), `events[].extensions.step_id`, and `extensions.schema_version`. The
  first two are the ones that matter: on the way out `ir_to_atif` declares both
  `SYNTHESIZED`, and on the way back nothing in the document marks them as
  invented, so the reconstructed trace asserts — with the full authority of the
  tri-state contract of §8.2 — that the agent's version was observed to be
  `"unknown"` and that every tool was observed to be called with no arguments.
  **The information was not lost so much as overwritten with a plausible value
  of the same shape.** A consumer reading only the second trace cannot tell.
  With `prompts` the same laundering happens one level up: steps that are not
  trace data at all return as captured user messages, and the event count grows
  by exactly the number of prompts.
- **A timeout costs five fields and the event carrying them.** H2 differs from
  H1 only in ending in a timeout, and that single fact takes with it
  `events[].outcome` (`"wall_clock_timeout"`), `outcome.status` (`"timeout"`),
  and the three extension fields the marker carried — `timeout_sec`,
  `pending_tool_call_ids`, `terminal_trajectory_complete` — along with the event
  itself. §5 loss #4, measured rather than asserted. The IR made all of it
  representable; the trip through ATIF makes the run look like one that simply
  ended. (H1's single unrepresentable field, the source content block, is the
  sixth entry in H2's column and is not a cost of the timeout.)

`transformed` with `matched=0` is worth reading as its own category:
`events[].source_type` and `events[].tool_call.name_semantics` are not degraded,
they are **replaced** — ATIF's step `source` and its `function_name` slot sit
down where the ACP type string and the `acp_kind` semantics used to be.

#### What this does and does not argue

It argues for the hub in the one way an assertion cannot: the pair of loss
reports carries exactly the facts the second trace has lost the ability to
state, and the measurement shows there is no third place to get them from. It
does **not** argue that the round trip is safe, or that ATIF should be used as a
storage format — the opposite, if anything.

**Not measured, and not claimed:** oracle rollouts and non-text content blocks
appear only in constructed test input, never in a captured rollout here; no
artifact on hand carries a non-null cost, so `usage.cost_usd` round-trips in
tests only; and the loop has been run over two rollouts from one agent, which is
a demonstration, not a survey.

### 8.11 What the OTel edge showed

#### Where its ground truth comes from

**FACT.** The three edges before this one could be checked against a producer in
this repository. There is no OpenTelemetry producer here at all (§4.2), so the
alternative to guessing was the lock file. The suite's fixture is the output of
`google.protobuf.json_format.MessageToJson` over an `ExportTraceServiceRequest`
built with `opentelemetry-proto` **1.41.1** — the version `uv.lock` pins, whose
wheel hash matches the lock entry — and every `gen_ai.*` constant in the module
is copied from `opentelemetry-semantic-conventions` **0.62b1**, likewise pinned.
Neither package is imported, neither is added as a dependency, and `uv.lock` is
untouched.

Reading them settled four things that a mapping written from memory gets wrong,
and the deleted `OTelCollector` got three of them wrong:

- `intValue` is a JSON **string** in the canonical mapping (`"1204"`), not a
  number, and `startTimeUnixNano` likewise. Both spellings are accepted on
  parse, so a reader has to handle either.
- An `AnyValue` writes its member explicitly even when the value is falsy:
  `{"stringValue": ""}`, `{"boolValue": false}`, `{"doubleValue": 0.0}`. The
  deleted collector's `stringValue or intValue or doubleValue or boolValue` read
  all three as "no value".
- `gen_ai.usage.total_tokens` **does not exist** at 0.62b1, and the cache
  counters are spelled `gen_ai.usage.cache_read.input_tokens` /
  `gen_ai.usage.cache_creation.input_tokens` — one dot away from the collector's
  spelling, and never a match.
- Enum fields serialize as member **names** by default (`"STATUS_CODE_ERROR"`)
  and as integers under `use_integers_for_enums`. Both are accepted on parse, so
  both are recognized here.

#### The finding that decides how identity is handled

`traceId` and `spanId` are `bytes` in the proto, and the protobuf canonical JSON
mapping encodes bytes as **base64** — a 16-byte trace id becomes 24 characters
ending in `==`. Much of the ecosystem writes lowercase hex instead. The two are
**not reliably distinguishable**, and this is checkable rather than a worry:
feeding the 32-character hex string `4bf92f35…` to the pinned JSON parser is
accepted *as base64* and yields **24 bytes**, silently.

So the edge carries identifiers **exactly as written** and re-encodes nothing.
Any normalization would make identity depend on a heuristic, and a heuristic
that is right most of the time is the worst possible property for an identifier.

#### Two limits that are properties of OTLP, not of this converter

- **Absent and default are the same document.** `name`, both timestamps,
  `parentSpanId`, `kind` and the `dropped*Count`s are protobuf scalars *without
  presence* — verified against the pinned descriptors — so a producer that never
  set the field and one that set it to the default write identical JSON. The
  IR's absent/`None`/observed-empty tri-state (§8.2) is therefore real on this
  edge only for the fields OTLP models with presence, and the limit is declared
  once per conversion rather than left to a reader to discover.
- **Nanoseconds do not fit a `datetime`.** OTLP counts nanoseconds; `datetime`
  resolves to microseconds. A span ending at `…1900000375` loses 375 ns from the
  IR field. The exact integer stays in `extensions.otel.span`, so the value is
  preserved — but the canonical field no longer holds it, which is a
  normalization and is recorded as one.

#### `UNSUPPORTED` and `NORMALIZED` for the same empty field

The trace-level `started_at`, `finished_at` and `usage` are empty after every
OTel conversion, and the class says *why*, which is where a fix would land:

- when spans carry timestamps or token counters, the records are `NORMALIZED` —
  the information is in the trace, per event, and deriving a run extent or a
  token total from it would assume the payload holds the whole run. **An OTLP
  export request is a batch**: it may carry several traces, it need not contain
  the root span, and one trace may arrive across several requests.
- when no span carries them, the same fields are `UNSUPPORTED` — there was
  nothing to aggregate.

The same reasoning keeps this edge from mapping a span status onto
`outcome.status` or onto `ToolStatus`: a status is per span, and choosing one
span's status as the run's needs a root the payload does not promise.

#### What the report costs

**MEASURED**, on the producer-derived fixture — five spans: an `invoke_agent`
root, a `chat` child with usage, two `execute_tool` children, and a plain HTTP
span.

| | count |
|---|---|
| records total | 31 |
| `unsupported` | 19 |
| `normalized` | 12 |
| `dropped` | **0 — see below** |
| `synthesized` | **0 — see below** |
| systemic (declared once) | 19 |
| per-event | 12 |
| envelope (source space) | 0 |

**The two zeros are not the same kind of zero, and neither should be read as a
property of the edge.**

`synthesized` is **structurally unreachable**: the class does not appear
anywhere in `ir_from_otel.py`, and it could not. It means "the *target* required
a value the source did not have", and on an inbound edge the target is the
canonical IR, whose only required fields are `provenance`, an event's `index`
and `kind`, and a content block's `kind` — each derived from the input rather
than invented. There is no slot this edge could be forced to fill. `ACP → IR`
and `ATIF → IR` have zero sites for the same reason; `ir_to_atif`, which is
outbound, has eight.

`dropped` is **a property of this payload**, not of the edge. Four inputs reach
a `DROPPED` site, and two of them are fully conformant OTLP *and* fully
conformant semantic conventions:

| input | validity | record |
|---|---|---|
| two spans with different `gen_ai.request.model` | valid OTLP, valid semconv | `agent.model` |
| `gen_ai.usage.input_tokens` and the deprecated `gen_ai.usage.prompt_tokens` disagreeing | valid OTLP, both names in 0.62b1 | `events[i].usage.input_tokens` |
| a token counter carried as a `doubleValue` or `bytesValue` | valid OTLP, violates semconv | `events[i].usage.<field>` |
| a non-list `scopeSpans`, a non-object span entry, a non-string `traceId` | wire-invalid | source-space records |

The fixture is a single-model trace with no deprecated spellings, so it reaches
none of them. Reading its `0` as "this edge never drops anything" would be
exactly the inference the loss model exists to prevent.

Repeating the same five spans *k* times gives **31, 43, 67, 139** records for 5,
10, 20 and 50 spans — `19 + 2.4n` for this payload's span mix. The systemic half
is constant; only the per-span half grows, which is the same affordability
property `ACP → IR` has (§8.8).

#### The edge reaches two of the seven event kinds

**FACT.** `gen_ai.operation.name` has eight values at `SEMCONV_VERSION`, read
out of the pinned wheel: `chat`, `generate_content`, `text_completion`,
`embeddings`, `retrieval`, `create_agent`, `invoke_agent`, `execute_tool`. Only
the last becomes a typed IR kind. Everything else — including a span with no
`gen_ai` attribute at all — becomes `UNKNOWN`, so of the IR's seven
`EventKind` members this edge can emit exactly **`tool_call` and `unknown`**.
`user_message`, `agent_message`, `agent_reasoning`, `timeout` and `oracle` are
unreachable from OTLP.

That is worth stating plainly, because it means **an OTel-derived trace is
structurally poorer in the hub than an ACP-derived one**, and a consumer that
assumes otherwise will be wrong.

`UNKNOWN` does not mean nothing was read. A `chat` span still fills
`agent.model`, `events[].usage.*`, both timestamps, the decoded attribute map
and the whole span in `extensions`; what `UNKNOWN` withholds is the assertion of
an *event type*. The reason splits in two:

- for `invoke_agent`, `create_agent`, `embeddings` and `retrieval` there is **no
  candidate**: the IR has no member that means any of them, so `UNKNOWN` is
  forced rather than chosen;
- for `chat`, `generate_content` and `text_completion` there *is* a candidate —
  `agent_message` — and the blocker is checkable rather than a matter of taste.
  Filling it means reading `gen_ai.input.messages` / `gen_ai.output.messages`,
  whose structure the pinned package defines by reference to a JSON schema it
  does not ship: the wheel contains 129 entries, none of them a `.json` file.
  Mapping them would be writing against a document nobody in this repository
  can check.

#### What the guards caught, and what caught the guard

**FACT.** Three rounds, each finding something the round before could not.

**The contract guard, first version.** It derives the IR field list from the
models and requires every field to be filled or declared. It found two **real
undeclared absences** in the first working converter:
`events[].reasoning_segments` and `events[].usage.cache_creation_tokens`. The
second produced a distinction worth keeping — a usage field the edge *can* fill
but this payload lacks reads differently from one no OTLP payload can carry, and
the records now say which.

**Twenty targeted mutations, twenty caught** — sorting spans by time,
attributing every span to the agent, synthesizing a tool-call id, parsing a
serialized arguments string, deriving the run extent, summing usage, normalizing
a hex id, reproducing the deleted collector's falsy-value bug, among others. Two
were green on the first run: one mutation was a no-op and was rewritten, and the
other exposed a **real gap in the suite** — nothing asserted that
`gen_ai.conversation.id` is not read as a session id, so a converter that read
it stayed green. That test exists now.

**A structural review of the finished slice found four more, and one of them was
in the guard itself.** All four were invisible to a suite that was green
throughout, which is the same lesson `ATIF → IR` produced (§8.10) in a different
place.

- **The guard was field-level, not instance-level.** It asked whether a path was
  filled *somewhere* or declared *somewhere* in the trace. So a field the
  fixture happens to fill was satisfied on every payload — including one where
  it is empty and nothing declares it. `events[i].tool_call.started_at` was
  exactly that: an `execute_tool` span with no readable instant left both
  tool-call timestamps `None` with no record at any path beneath `tool_call`,
  while `ACP → IR` and `ATIF → IR` both declare those two paths
  unconditionally. The guard now checks **one instance at a time** — per event,
  per content block — with two exemptions that are properties of the IR rather
  than of a converter: a `tool_call` payload that invariant 3 *forbids* on a
  non-tool event, and a field covered by a record on an outer node.
- **The content blocks were outside the contract entirely.** The walker did not
  descend into `list[ContentBlock]`, so `content[].kind`, `.text` and `.raw`
  were never asked about. A text block read from a string result has no `raw`
  and an object block has no rendered `text`; neither absence is structural, and
  both are now declared at their own concrete path — the same path
  `ATIF → IR` uses for `content[].raw`.
- **The envelope partition was being flattened silently.** Two spans batched
  under *different* `ScopeSpans` objects with an equal `scope` became
  indistinguishable once the payload was read into one list. `§8.3` now carries
  the three coordinates in `extensions.otel.envelope`.
- **"Faithful" decoding was described as preserving the attributes.** It
  preserves the *values*; the wire form goes. `{"intValue": "7"}` and
  `{"intValue": 7}` are the same `7` afterwards. The table row is now
  `normalized`, with a record to match.

The point worth carrying forward: **a guard derived from the models still
encodes a choice about what counts as an absence**, and that choice is not
checked by the guard. Deriving the field list from code removed one class of
blind spot and left another.

#### Open maintainer decisions

None of these is answered by the code, and each is a place where two readings
are semantically plausible. They are listed rather than decided, because
deciding one here would put the decision in the hub where every format inherits
it.

1. **Should the hub model causal structure?** OTLP's span tree is the first
   source with real parentage. Today it is preserved verbatim in
   `extensions.otel.span`; a `parent` field on `TraceEvent` would make it
   canonical, and would oblige every other edge to have a position on it.
2. **Which identifier encoding is canonical?** Hex or base64 — and should this
   edge normalize to it, given the two are not distinguishable with certainty?
3. **Is `gen_ai.conversation.id` a BenchFlow `session_id`?** The pinned package
   calls it "a conversation (session, thread)", which is close enough to be
   tempting and not close enough to be a fact.
4. **Should spans other than `execute_tool` map to typed kinds?** A `chat` span
   plausibly corresponds to an agent turn. Doing it needs a position on
   `gen_ai.input.messages` / `gen_ai.output.messages`, whose structure the
   pinned package defines by reference to a JSON schema it does not ship.
5. **Should a serialized `gen_ai.tool.call.arguments` string be deserialized?**
   The pinned text puts that obligation on the instrumentation. A reader that
   does it anyway is deciding that a string which happens to be JSON was meant
   as structure.
6. **Should a span status map onto `ToolStatus` or `OutcomeStatus`?** See above:
   plausible per span, unfounded for a run.
7. **Is depending on `_incubating` conventions acceptable?** Every `gen_ai.*`
   name is experimental at 0.62b1 and several are already deprecated. Copying
   the constants (as here) keeps the lock file untouched but freezes a snapshot;
   importing `opentelemetry-semantic-conventions` would track it and would make
   the package a real dependency.
8. **Is an `IR → OTel` emitter wanted at all?** Open question 5, still open for
   the outbound half. It needs answers to 1, 2 and 4 before it can be written
   without inventing a span tree for BenchFlow rollouts. §8.12 defers it in the
   meantime; deferring is not answering, and the question stays on this list.

#### Human verification

**HUMAN E2E VERIFIED, 2026-08-18.** The procedure is `e2e-f/PROCEDURE.md`, and
every step was carried out and judged by a person rather than by the suite:

| | what was verified |
|---|---|
| F1 | fixture provenance and encoding, regenerated against the pinned `opentelemetry-proto==1.41.1` wheel with a hash matching `uv.lock` |
| F2 | the mapping span by span — identity, parentage, ordering, tool calls, verbatim status retention, timestamp truncation, no trace-level aggregation, no conversation-to-session inference |
| F3 | all 31 loss records read individually; every per-event path resolved |
| F4 | the tri-state `AnyValue` behaviour |
| F4b | the `resourceSpans`/`scopeSpans` partition surviving as coordinates |
| F5 | every negative control biting as intended |
| F5b | the mutation harness: every defined mutation applied and caught, no `MISSED`, no `SKIP`, exit 0, followed by a green `tests/trajectories` run |
| F6 | the IR family still unwired — no external importers, no `opentelemetry` dependency, `uv.lock` unchanged |

**The verification found two defects, both in the procedure and neither in the
converter.** F3's scaling table and F5's resolvability check were still stating
the counts a pre-review version of the edge produced. Both are corrected, and
both criteria are now written as properties — a constant systemic half and a
constant per-span rate; zero unresolved paths in the canonical encoding and more
than zero under `exclude_none=True` — rather than as numbers that go stale the
next time a record is added. That is the reusable part: **a count in a procedure
is an expectation with a shelf life.**

Nothing in the converter changed as a result, and no test was weakened to make a
step pass.

#### Not claimed

The verification above is bounded by the following. None of them is a formality;
each one names something a reader could otherwise reasonably assume, and none is
narrowed by the fact that a person signed the steps off.

- **Human verification does not extend past those limits.** What was checked is
  that this edge does what §8.3 says it does, on the payloads named above. It is
  not evidence about payloads nobody has seen.
- **No OTLP payload emitted by a real agent has ever been read, including
  during the human verification.** Nothing in BenchFlow emits spans, so there is
  no rollout to check against — F1 verifies the fixture against the *library*,
  which is the strongest available substitute and is not the same thing. The
  fixture is
  the output of `opentelemetry-proto` 1.41.1 itself, which makes its **encoding**
  authoritative — base64 ids, `intValue` as a string, enums as member names,
  defaults written as absence — and its **content** a construction: the five
  spans, their attributes and their shape were chosen by hand, not observed.
- **There is no `IR → OTel` emitter.** This edge is inbound only. Nothing in
  this repository writes an OTLP span, and the outbound direction is not
  designed, sketched or estimated here.
- **There is no OTel round trip, and none is measured.** §8.10's
  `ACP → IR → ATIF → IR′` loop closes because both ATIF edges exist. No loop
  closes through OTel, so there is no preservation figure for this edge and none
  is implied by the record counts above.
- **`0 dropped` is a measurement of the fixture, not a property of the edge.**
  See the table above: two fully conformant inputs produce `DROPPED` records.
- **The mapping is checked against the pinned artifacts, not against the
  specification.** Where the OpenTelemetry specification says something those
  two packages do not express — the OTLP/JSON hex-identifier convention is the
  clearest case — this edge implements nothing and says so.
- **The GenAI semantic conventions are still `_incubating` at the pinned
  version**, and several attributes read here are already deprecated in it. A
  later version can move the ground under every `gen_ai.*` constant, and no
  amount of verification against 0.62b1 prevents that.
- **Nothing is wired.** `ir_from_otel.py` and `_otlp_anyvalue.py` join the
  closed family of §8.7; no run path imports either, and no artifact changes
  because they exist.

### 8.12 Scope decision: OpenTelemetry is ingest-only in this implementation

> **DECISION — ours, current, and reversible.** This section records a choice
> made by the author of this work. **It is not a maintainer ruling, and nothing
> here should be read as one.** No maintainer has stated that `IR → OTel` is out
> of scope, and none has stated that it is required.

**The decision.** For this implementation, OpenTelemetry is an **ingest
boundary**. The edge that exists is `OTLP/JSON → Canonical Trace IR` (§8.3).
`Canonical Trace IR → OTel` is **deferred**, not rejected.

#### Why defer rather than build

Five observations, each checkable in the tree today, and none of them a
statement about what the maintainers want:

1. **No outbound OTel consumer exists in the repository.** Nothing reads spans
   BenchFlow would produce.
2. **No OTel exporter or backend is integrated.** There is no OTLP endpoint, no
   collector configuration, and no observability backend referenced anywhere in
   `src/` or `docs/`.
3. **The OTel support that did exist was inbound.** The removed `OTelCollector`
   (§4.2) was an OTLP/HTTP *receiver*; it contained no emitter.
4. **The consumer named by the task's own issue body consumes ACP.** The Viewer
   work reads `trajectory/acp_trajectory.jsonl` and explicitly declines to
   become another trace schema.
5. **There is no concrete contract to build the emitter against.** Without a
   consumer, every question §8.11 lists — the span tree for a rollout, which ids
   to mint, which encoding to write them in, whether to commit to an
   `_incubating` vocabulary — would be answered by this implementation alone.
   Those answers would be **speculative policy baked into the hub**, and the hub
   is the one place where a wrong answer propagates to every other format.

Building an edge against no consumer is how a format acquires conventions nobody
chose. Deferring costs nothing that cannot be recovered; guessing does.

#### What this decision explicitly does not claim

- **It does not claim `IR → OTel` will never be needed.** It claims there is
  nothing to build it against *today*.
- **It does not reinterpret the `<->` in the task title.** That symbol is
  ambiguous — it can mean bidirectional conversion on every edge, or simply "among
  these three formats" — and this decision is **not** evidence that ingest-only
  is what it meant. The ambiguity is unchanged and is still a maintainer's to
  resolve (§6, question 5).
- **It does not narrow the task.** If the answer comes back "both", the work is
  an additional edge, not a redesign.

#### Reversal conditions, and what must not happen first

The deferral ends when **either** exists:

- a **consumer** that would read BenchFlow-emitted spans — a collector, a
  backend, a hosted viewer, a downstream tool; or
- a **contract** that fixes the open questions independently of this
  implementation — a maintainer decision, a vendored specification, or a
  convention this project agrees to follow.

Until then, **the IR must not be changed in anticipation of the emitter.** No
span-tree field, no id-minting policy, no OTel-shaped slot may be added to
`ir.py` on the argument that an emitter would want it. The hub is a superset of
what BenchFlow can *observe* (§8.2), and a field added for a converter that does
not exist is a field with no evidence behind it. When the emitter is written it
takes the IR as it finds it and declares whatever it cannot express — which is
the same contract every other outbound edge is held to (§8.3, `IR → ATIF`).

Adding the edge later is cheap by construction: the outbound direction is one
module and one loss report, exactly as `ir_to_atif.py` is, and the family test in
§8.7 is what keeps that addition from quietly wiring anything.

### 8.13 Loss-bounded conformance: is the observed divergence inside the declared contract?

§8.10 measures **D** — what actually changed across a round trip, read off the
two traces and never off the reports. Each converter separately produces **L** —
what it declared it could not carry, or had to invent. The two are produced
independently on purpose, and
[`test_the_harness_reads_traces_and_not_reports`](../tests/trajectories/test_ir_round_trip.py)
exists to keep the measurement from confirming the converters against
themselves.

[`ir_conformance.py`](../src/benchflow/trajectories/ir_conformance.py) is the
third thing: the **join**. It answers one question — *is every observed
divergence accounted for by something the contract declared?* — and it is
deliberately not folded into either half, because a measurement that consulted
the declarations would stop being a measurement.

#### The naive form does not hold, for two reasons worth stating

The obvious gate is `D(T, R(T)) ⊆ L(T)`. Implemented literally it fails on a
correct round trip, and both failures are real properties of the model rather
than bugs to paper over.

**The two sides address different spaces.** D speaks in canonical *hub* paths of
the IR. L speaks in three (`PathSpace`), and an outbound edge legitimately
declares a value in `TARGET` space using its target's vocabulary. `ir_to_atif`
writes `schema_version` into the ATIF document and declares it *there*, because
at that moment the value has no IR antecedent: a hub path would address a node
the trace being converted does not contain, and `ir.py`'s invariant that every
hub record resolves in the trace would — correctly — reject it. Then
`ir_from_atif` reads the value faithfully back into `extensions.schema_version`,
and the round trip observes a value the input never had. **Neither edge lied.
The fabrication is a property of the composition**, and the join is the only
place that can see it.

`TARGET_TO_HUB` is the bridge, three entries, each a fact about what
`ir_from_atif` does rather than a convention:

| declared at (target) | observed at (hub) | why |
|---|---|---|
| `schema_version` | `extensions.schema_version` | the inbound edge stores the document's dialect under the trace's extensions |
| `steps[].step_id` | `events[].extensions.step_id` | ...and each step's id under its event's |
| `steps[].message` | `events[].text` | an ATIF step message becomes the event's text |

**This is a bridge, not an exemption.** Remove the `SYNTHESIZED` record and the
gate fails; the map only lets a declaration made honestly in one space answer
for the path it occupies in the other. `test_every_bridge_entry_is_real` runs a
trip and checks each correspondence against actual values, so an entry cannot
rot into a lie while still reading as coverage.

**Not every divergence is a field-level fact.** When an event is fused away, the
values it held stop appearing at *every* path it populated. Those divergences
are caused by a change in the event sequence, not by a converter mishandling a
field. `structure_explained` is how the join says so **without** exempting the
paths involved: the values that went missing must be exactly values held by
events of a kind that lost instances. It is per value, not per path — a kind
vanishing does not license *any* change at a path those events touched. An edit
to a surviving event leaves a missing value nobody was holding, and fails.

#### The rules

1. **Undeclared fabrication is always a violation.** Values coming back with
   none going in requires a `SYNTHESIZED` record at that path, directly or
   through the bridge. There is no allowlist, and **structural metadata is not
   exempt**: `schema_version` and `steps[].step_id` are now declared by
   `ir_to_atif` like anything else, and clear the gate the ordinary way.
2. **A representable loss is a violation.** Values in, none out, at a path the
   target *does* have a slot for, is a gap in our own edge.
3. **A transformed path must account for both of its sides.** Values that
   disappeared: declared, or structure-explained. Values that appeared:
   declared. `TRANSFORMED` conflates the two, and a rule checking only the first
   would let an arbitrary insertion through — a mutation confirms it does not.

#### Why `schema_version` and `step_id` are `SYNTHESIZED` and not `NORMALIZED`

`NORMALIZED` says a source value was reshaped. Neither of these has a source
value: the ATIF dialect string is not an observation about the run, and the
step positions are not the IR's event index renumbered — that index is a
position in a different sequence, over events some of which never become steps.
Calling them normalizations would assert a correspondence that does not exist,
which is the same laundering §8.3 refuses for `name_semantics`.

#### Human verification of the conformance gate

**HUMAN E2E VERIFIED, 2026-08-18, H1–H8 all pass**, against
`e2e-h/PROCEDURE.md`. Every step was carried out and judged by a person rather
than by the suite.

| | what was verified |
|---|---|
| H1 | the working tree is exactly the six files under review, and the family/isolation tests pass — nothing outside the IR family imports any of it |
| H2 | `test_ir_conformance.py` green with **no skips**, so the two captured rollouts really ran rather than silently vanishing with the evidence tree |
| H3 | `D`, `L`, `D ∩ L` and `D \ L` read by hand on both rollouts: the bridge reduces `D \ L(hub)` to `D \ L`, every residual is structure-explained, and the gate reports no violation |
| H4 | each `TARGET_TO_HUB` entry corresponds to a real inbound/outbound behaviour, not to a convenience; the bridge tests pass |
| H5 | the gate can be made to fail: removing the `schema_version` record, removing `steps[].step_id`, or relabelling `SYNTHESIZED` as `NORMALIZED` each make the trip non-conformant |
| H6 | the structural rule is not a blanket exemption — an edit to an event that *survived* the trip is caught even though other events vanished |
| H7 | mutation harness: baseline green, every mutation applied and every one caught, no `ANCHOR NOT FOUND`, no `MISSED`, the file restored and `git status` unchanged |
| H8 | full suite on `HEAD` and on the working tree compared directly: the same pre-existing failures on both, empty diff, tree restored |

**The verification found one defect, in the procedure, not in the gate.** H8's
full-suite command could stop at an interactive GitHub credential prompt
part-way through a long run, which reads as a hang rather than as a failure.
`GIT_TERMINAL_PROMPT=0` is now on every H8 command so a credential problem
fails visibly instead of blocking the step.

#### What a green gate does not establish

- **Not that the contract is *right*.** An edge that declares `SYNTHESIZED` on
  a field and then invents it passes here. The gate checks that declaration and
  observation **agree**; it cannot check that the declaration is true. It is a
  consistency check between two independently produced artefacts, which is
  exactly as much as a join can be.
- **Not a universal property.** It was run on finite corpora — the two captured
  rollouts of §5.2 and Slice E's fixture. A clean result is evidence about
  those traces, and no corpus was built to make it come out that way.
- **Nothing about OpenTelemetry.** There is no OTel round trip here because
  there is no `IR → OTel` edge to close one with (§8.12). The OTel direction is
  ingest-only, and a gate cannot measure a loop that does not exist.
- **Little about ACP.** The `Representability` column travels from §8.10's ATIF
  capability table, so rule 2 is meaningful for the ATIF pair and for nothing
  else. The ACP loop is carried only as a **regression guard**, with that rule
  filtered out where it does not apply, and the test says so.
- **Nothing about a run.** No wiring was added. The gate reads two traces and
  two loss reports that a test constructed; it is not on any run path, writes
  no artifact, and `ir.py`, `ir_round_trip.py`, the viewer and every runtime
  module are untouched by it.

### 8.14 `Canonical Trace IR → viewer trace steps`

[`ir_to_view.py`](../src/benchflow/trajectories/ir_to_view.py) converts a trace
into the step list a BenchFlow viewer page renders. It is unwired: no run path
imports it, no page is built from it, and no artifact changes.

#### It produces steps, and deliberately not a payload

A viewer payload has five fields and only one of them is a function of the
trace. `rollout_name` is a directory name; `verifier` is four sidecar files;
`meta` is `result.json` and `timing.json`. The IR has no slot for `task_name`,
`skill_mode`, `reward`, `partial_trajectory` or `trajectory_source` **at all**,
so an edge that returned a whole payload would be **synthesizing run metadata
to fill a shape** rather than converting a trace. The trace-level fields the IR
*does* carry — `session_id`, `agent`, `usage`, `outcome`, the trace's own
start and finish — are declared `UNSUPPORTED`, which says they are outside this
edge's codomain rather than lost by it: the viewer shows them, from artifacts
this edge never sees. Assembly belongs to the wiring slice, which has the
directory.

#### Where the shape comes from, and how provisional that is

The wire shape is read from the viewer package proposed in
`benchflow-ai/benchflow#1034`, at the commit recorded in `VIEW_SCHEMA_ORIGIN`.
**Nothing is imported, vendored or fetched from that branch** — it is unmerged,
and the family rule is the one §8.3 already follows for ATIF: read the target
format as data, pin what matters by test, never reach into the module that
handles it. `test_the_vocabularies_are_frozen` freezes our copy.

Be clear about what that buys: it protects **our** contract from drifting. It
cannot notice #1034 changing. That correspondence was checked by a person once,
against the recorded commit, and re-checking it is a human step.

#### The one addition

`tool.name_semantics` is **ours**. #1034's `ToolCall` has six fields; this is a
seventh, and it carries :attr:`ToolCall.name_semantics` through unchanged.
Without it the viewer boundary cannot tell an ACP *category* from a *function
name* from a *span name*, and that is exactly the gap #1034 fills by inference:

```python
_HUE_INFER = (("read", ("read", "cat", "view", "ls", "list")), ...)
def tool_hue(kind: str, title: str) -> ToolHue:
    if kind in _KNOWN_HUES: return kind
    hay = f"{kind} {title}".lower()
    for hue, needles in _HUE_INFER:
        if any(needle in hay for needle in needles): return hue
```

On the three corpora that reads: an ACP `read` is a read (correct); an ATIF
`function_name` of `execute` becomes the `execute` category (a name read as a
category); an OTel `gen_ai.tool.name` of `read_file` becomes `read` (a category
inferred from a substring). This edge does none of it. A hue is emitted only
when the semantics say `acp_kind` **and** the value already **is** a member of
the display vocabulary — membership, never inference — and otherwise the
neutral `other`, which the renderer maps to the secondary/border tokens and
which therefore asserts nothing. Emitting an extra key is additive, but it is a
divergence from #1034's contract and not an agreed extension to it.

#### Declare, don't refuse

Unlike §8.3's ACP edge, this one never raises. A viewer is a display: an event
it cannot type must still reach the page. **Every event produces exactly one
step** — pinned by test on both captured rollouts and on hand-built traces —
and what the shape cannot hold goes to the loss report.

| IR event kind | step kind | how identity survives |
|---|---|---|
| `USER_MESSAGE` | `prompt` | observed `text` |
| `AGENT_MESSAGE` | `message` | observed `text` |
| `AGENT_REASONING` | `thought` | `reasoning`, not `text` |
| `TOOL_CALL` | `tool` | the tool object, `name_semantics` included |
| `TIMEOUT` | `timeout` | stays **typed**; `reason` from the event's terminal signal, the rest from `extensions` |
| `ORACLE` | `unknown` | `type: "oracle"` — the vocabulary has no oracle member |
| `UNKNOWN` | `unknown` | `type` = the source's own type string |

`source_type` is carried for **every** kind, not only the untyped ones: a
normalized kind whose source string was discarded is a lossy rename.

For the two kinds with no typed slot the step's `text` is a serialization of
the **canonical IR event**. #1034's own unknown branch serializes the raw ACP
dict it read off disk; this edge has no such document, so what it renders
carries IR field names, and the loss record says so. A page showing it must
label it the same way — it is not the producer's payload.

#### Absent is not observed-empty

Six viewer slots have no null to write. Each substitution is declared on its
own, and **only** when the source value was absent: an observed `""` title and
a missing one both render as `""`, and the report is the only place that
difference survives. Blocks are the same principle: a block the IR holds with
no text contributes no string, declared and omitted, because rendering its
`raw` as JSON would put tool output on the page that no tool ever produced. An
observed empty string is kept — #1034 filters falsy strings out, and that is a
small silent loss this edge does not reproduce.

#### Measured on the three corpora

**FACT — machine measurement, not a human verification.**

| corpus | events → steps | tools (`kind`, `name_semantics`, `hue`) | loss records |
|---|---|---|---|
| ACP H1 | 5 → 5 | `execute`/`acp_kind`/**`execute`**, `read`/`acp_kind`/**`read`** | 17 |
| ACP H2 | 4 → 4 | `think`/`acp_kind`/**`think`** | 17 |
| ATIF (§8.10 fixture) | 3 → 3 | `execute`/`function_name`/**`other`** | 18 |
| OTel (§8.11 fixture) | 5 → 5 | `read_file`, `write_file` / `gen_ai.tool.name` / **`other`** | 23 |

The third and fourth rows are the point. `execute` is spelled identically in
rows one and three and gets a category in one and not the other, because the
difference is in the semantics and not in the string.

OTel is also the only corpus with timestamps — all five steps carry `t` and
`dur` — and the only one whose untyped events carry the whole span in
`extensions`, `parentSpanId` included. It is also the only one with real tool
arguments, which the viewer shape has no slot for and which are declared
`DROPPED`: an observed value reaching no field.

Counts here are snapshots of these fixtures, not thresholds.

#### Human verification of the viewer edge

**HUMAN E2E VERIFIED, 2026-08-19, V1–V10 all pass**, against
`e2e-view/PROCEDURE.md`. Every step was carried out and judged by a person
reading the values the edge produced, not by re-running the suite.

| | what was verified |
|---|---|
| V1 | the working tree is exactly the four files under review; `ir.py`, `ir_round_trip.py`, the existing `viewer.py` and every runtime module untouched; the family/import test green |
| V2 | steps are 1-based and dense, no `label` is ever written, `type` is preserved on the **known** kinds too, absent optionals are omitted rather than nulled, and the tool object carries seven keys including `name_semantics` |
| V3 | ACP H1 5→5, ACP H2 4→4, ATIF 3→3, OTel 5→5 — one step per event on every corpus, no silent drop |
| V4 | a real `acp_kind` can produce a semantic hue; `function_name` and `gen_ai.tool.name` stay `other`; no substring and no title inference, checked with titles deliberately stuffed with category words |
| V5 | `ORACLE` and `UNKNOWN` both reach the page; the diagnostic text is declared a serialization of the canonical IR event and not a raw source record; OTel parentage survives inside `extensions` |
| V6 | absent and observed-empty are distinguishable **only** through the loss report, and are; H2's real timeout is carried, not synthesized; an opaque block with no text is `DROPPED` rather than invented |
| V7 | `role`, `content[].kind`/`raw`, per-event usage, `reasoning_segments` and `arguments` are declared `DROPPED`; trace-level fields are `UNSUPPORTED` as outside this edge's codomain; no run metadata appears on any step |
| V8 | 13 valid mutations applied and caught, no `ANCHOR NOT FOUND`, no `MISSED`, no collection error, restore green |
| V9 | 752 collected = 699 baseline + 53 new; a node-id comparison against `HEAD` shows no pre-existing test missing; ruff, format and `ty` green |
| V10 | the same four files at the end; `e2e-view/` never wrote into the clone |

**The verification found three defects, all in the mutation harness, none in
the converter.** Two mutations were no-ops — one kept the declaration it was
meant to remove, the other was overwritten by a later dict entry. The third
rewrote a `losses.add(` call into a tuple display, which is a `SyntaxError`
once the call's `space=` keyword is inside it: the module stopped importing and
pytest reported a **collection error** rather than a failure, so the harness
counted as "caught" a run that only proved a broken file will not import. All
three were rewritten outside the repo, and the procedure now names `error`
instead of `failed` as a second way a mutation can fail to mean what it claims.

#### What this verification does not cover

- **No wiring, and no page.** Nothing imports this edge from a run path, no
  renderer was run, no HTML was produced, and no artifact on disk changes. That
  the steps are correct is not evidence that they *render* correctly.
- **The contract pin protects our side only.** `VIEW_STEP_KINDS`,
  `VIEW_TOOL_HUES` and the emitted key sets are frozen by test, which stops
  *this* edge from drifting. It cannot notice `benchflow-ai/benchflow#1034`
  changing — that branch is unmerged and moving, and re-checking the
  correspondence to `VIEW_SCHEMA_ORIGIN` is a human step.
- **`name_semantics` is ours.** It is a seventh key on an object #1034 defines
  with six: additive, and a divergence from that contract rather than a
  maintainer-approved extension to it.
- **`ORACLE` stays `unknown`.** It keeps its identity in `type`, but it will
  remain an untyped step until the viewer contract gains a member for it —
  which is a change to propose upstream, not one to assume.

### 8.15 `viewer trace steps → a page the current viewer renders`

[`ir_to_view_html.py`](../src/benchflow/trajectories/ir_to_view_html.py) closes
the chain §8.14 opened. It takes the step list and produces one HTML page,
using the card builders `benchflow/trajectories/viewer.py` already renders its
ACP page with:

```
<format> → ir_from_* → CanonicalTrace → ir_to_view_steps → ir_to_view_html → page
```

It is written against the viewer **on `main`**, not against the reviewer-grade
package proposed in `benchflow-ai/benchflow#1034`. That branch is a design
reference for the step vocabulary (§8.14) and nothing more: no code is
imported, vendored or fetched from it, and this edge composes with the current
renderer whether or not that PR ever lands.

#### The direction of the dependency

One way, and it is the point. `ir_to_view_html` imports `viewer`; `viewer`
imports no part of the IR family. What crosses the boundary is a step list — a
plain document of six kinds and named keys — so the renderer stays a renderer
and the hub stays format-neutral. `test_only_the_ir_family_imports_the_ir`
still passes unchanged: this module joins the closed family rather than opening
it.

#### What it refuses to do

It does not rebuild steps into ACP capture events and hand them to
`_render_acp_events`. That would be a lie in the middle of the chain: the IR
holds records ACP has no type for (`oracle`, and anything `unknown`), a status
ACP cannot spell, and a tool name whose *semantics* are the whole reason the
hub exists. Forging capture events would launder all three straight back into
the assumption the conversion was built to stop.

#### What the current page does not show, measured

Against `viewer._render_acp_events` on the two captured rollouts of §8.3:

| | legacy ACP page | this edge |
|---|---|---|
| H1 (5 events) | 5 cards | 5 cards |
| H2 (4 events, ends in a real timeout) | **3 cards**, and the word "timeout" appears nowhere in the page | 4 cards, one of them a typed timeout |
| H2 + one unrecognized record | **3 cards** | 5 cards, the last one a labelled diagnostic |
| tool output | not on the page at all | shown when the block carried text |
| ATIF-only rollout directory | `<p>No trajectory files found</p>` (32 bytes) | 5 cards |

None of that is a bug being fixed. `_render_acp_events` has four branches —
`user_message`, `tool_call`, `agent_message`, `agent_thought` — and an
`agent_timeout` or an unknown record simply reaches none of them. This edge has
six step kinds to place, so it places them, and declares the difference instead
of presenting it as a repair. Three tests in
`tests/trajectories/test_viewer_primitives.py` pin the legacy behaviour as a
fact, so the day a branch is added, this edge's reason for existing is
revisited with it.

#### Classification: a table, not a substring

`viewer._tool_accent_class` scans the tool kind for needles and, failing that,
the human title. Executed on `main`: `read_file → acc-read`,
`bash -lc 'ls -la' → acc-bash`. It is not imported here, and an AST test
asserts the name never appears in this module.

Instead the hue that `ir_to_view` already decided — by membership, only when
the source said the string *is* a category — is mapped through `HUE_ACCENT`,
a table over the eight display hues. Two of them resolve to the neutral accent
for opposite reasons, and only one of the two is a loss:

- `other` — no category was observed. Nothing is lost; neutral is the claim.
- `think` — a real ACP kind the current stylesheet has no accent for. Declared
  `DROPPED` at `steps[i].tool.hue`, with the note that the category is still
  legible because it is the card's own label. Adding an accent for it is a
  viewer decision, not one to take inside a converter.

The same run, converted two ways, is the whole slice on one screen: H1's
`execute` arrives as an ACP `kind` and gets `acc-bash`; the identical string in
that run's `trainer/atif.json` arrives as a `function_name` and stays neutral.
OTel's `read_file` and `write_file` stay neutral for the same reason.

#### `name_semantics` on the page

The page has no payload, so preserving `name_semantics` means rendering it. It
appears twice per tool card: in the metrics line (`completed ·
name_semantics: acp_kind`), so a reader sees what kind of name they are looking
at, and as `data-name-semantics`, so a browser-level check can assert it
without parsing prose. `data-hue`, `data-tool-id` and `data-source-type` ride
along the same way.

#### Diagnostics say which document they are showing

A step with no typed slot is rendered under the label **`Canonical IR
representation`**, followed by its source type and the serialized event. The
label is not decoration: §8.14 emits the body of those steps as a
serialization of the *canonical IR event*, because the IR holds no source
record — and the legacy path's unknown branch, when it grows one, will be
showing the raw capture entry instead. A page that displayed ours under the
source's own type would claim a document that does not exist.

#### Cuts are announced

The legacy card cuts message text at 500 characters silently, and escapes
before cutting — so a long prompt can lose an entity to the knife. This edge
keeps the same 500 for message-shaped text (2000 for tool output, 4000 for a
diagnostic body), cuts **before** escaping, appends
`… [truncated, N more characters]` to the page, and records the cut as
`NORMALIZED`. The legacy asymmetry is pinned by a test and left alone: changing
it is a behaviour change to that renderer and belongs to its own commit.

#### Two reports, not one

`render_trace` returns the page and **both** reports — `ir->view` from §8.14 and
`view->html` from here — without merging them. They address different
documents, and a merged report would have to name a space neither edge has.
This is also the one edge in the family with the IR on *neither* side, so
`PathSpace.HUB` is never used: `SOURCE` addresses the step list it was given,
`TARGET` the page it produced, and no record of this edge can be joined to a
hub path by mistake.

#### Wiring: one opt-in branch, and what it costs

`render_rollout` gains a single branch, guarded by `BENCHFLOW_VIEWER_TRACE_IR`:

```python
if not turn_files:
    page = _trace_ir_page(rollout_dir, prompts)   # None whenever the switch is off
    if page is not None:
        return page
```

`_trace_ir_page` is the whole wiring surface. The import of
`ir_to_view_html` is **lazy and inside the function**, so importing the viewer
never imports the IR and deleting the family cannot stop `viewer.py` from
importing; a conversion that raises falls back to the ACP page with a line on
stderr, because a viewer that stops showing a run when a converter breaks is
worse than one that says so and shows it the old way. Removing this branch
unwires the family completely.

With the switch set, an ACP rollout renders through the hub and an ATIF-only
rollout — which the ACP path answers with `<p>No trajectory files found</p>` —
gets a page. `bench eval view` needs no flag and no change; `serve` writes
whatever `render_rollout` returned into `trajectory.html` as it always has.

This is the point where the reversibility claim of §8.7 narrows rather than
ends. `test_only_the_ir_family_imports_the_ir` now carries a `WIRING_SITES`
allowlist naming `viewer.py` and nothing else, a second test asserts that every
listed site really does import the family, and a third asserts the viewer holds
**no module-level** import of it.

There is also an entry point that wires nothing:
`python -m benchflow.trajectories.ir_to_view_html <rollout_dir|acp.jsonl|atif.json|otlp.json> [out.html]`,
which is how an ATIF document or an OTLP payload gets looked at — neither has a
route through `bench eval view`, and OTLP has no rollout artifact at all.
- **The viewer refactor that precedes it is pure.** `_page`, `_prompt_block`,
  `_message_block`, `_thought_block` and `_result_block` were lifted out of
  `_render_acp_events` unchanged; six pages rendered from the captured
  rollouts and from raw ACP and Codex session files have the same SHA-256
  before and after.
- **Not claimed:** no human end-to-end verification of the rendered pages, and
  no browser-level check of any kind. The pages have been produced and read as
  text; that is not the same as looking at them.
