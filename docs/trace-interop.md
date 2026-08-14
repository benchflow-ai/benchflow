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

**FACT.** Two other sources can put records into `acp_trajectory.jsonl`. Neither
goes through `_events_to_trajectory`, and the schema models neither. Note that
the artifact *path* is the same in every mode, so the same filename can hold
different record shapes depending on how the rollout ran.

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

So the file is not a superset of the ACP-session emitter's output; it holds one
of several possible shapes, and no single site defines its full vocabulary.
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
   deliberately.

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

## 8. Ideas not yet agreed

> **PROPOSAL — none of this is approved, scheduled, or implemented.**

The reconnaissance that produced this document also sketched a canonical
intermediate representation, so that conversion between trace formats would be
`N ↔ 1` rather than `N²`, with each conversion returning an explicit report of
the fields it could not carry.

That idea is recorded here only so this document is not silently read as an
argument for it. **It has not been reviewed by a maintainer, no design has been
agreed, and nothing in this repository implements it.** Any such work depends on
the open questions in §6 being answered first — in particular whether the capture
path may change at all, since several of the losses in §5 cannot be closed
downstream of it.
