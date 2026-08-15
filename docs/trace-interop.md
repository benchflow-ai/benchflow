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

### 8.3 Planned mapping

**`ACP → IR`** (Slice C,
[`ir_from_acp.py`](../src/benchflow/trajectories/ir_from_acp.py)) **and
`IR → ATIF`** (Slice D,
[`ir_to_atif.py`](../src/benchflow/trajectories/ir_to_atif.py)) **are
implemented**, both unwired and still provisional. `ATIF → IR` and the OTel
direction are not.

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

**FACT — the hub reproduces the direct exporter.** For any conformant capture
input, `ir_to_atif(acp_events_to_ir(events), prompts=P)` produces the same
document as `trajectory_to_atif_record(events=events, prompts=P)`, with one
deliberate exception (oracle, below). Asserted by
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
| `usage.cache_creation_tokens`, `.reasoning_tokens`, `.total_tokens`, `.source` | *(no slot)* | dropped | `usage.<field>` |
| `outcome.*` | *(no slot)* | dropped | `outcome`, once |
| `events[].provenance`, `.source_type`, `.extensions` | *(no slot)* | dropped | once each |
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

**FACT — measured on the same two real rollouts.** H1 produces **12** outbound
records (6 synthesized, 4 dropped, 2 normalized; 9 hub, 3 target); H2, whose
trajectory contains a real wall-clock timeout, produces **14** (5 synthesized,
7 dropped, 2 normalized), the extra drops being the timeout event, its
`extensions` and the trace `outcome`.

**FACT — the two reports compose.** For H1, `events[2].tool_call.arguments`
carries `unsupported` in the `acp -> ir` report and `synthesized` in the
`ir -> atif` one: the source never had arguments and the target demanded them
anyway, joined on one hub path (§8.2, choice 5).

**Known gap: the IR cannot express cost.** `TraceUsage` has no cost field, so
`final_metrics.total_cost_usd` — which the direct exporter accepts as an
argument — cannot be produced through the hub. Neither real rollout carries one
(agent-native runs have no proxy cost), so parity is unaffected there, but this
is a hub gap rather than a converter one and closing it means adding a field to
the IR.

#### IR ↔ OpenTelemetry

**PROPOSAL, and unverified.** §4.2 is FACT — there is no OTel dependency and no
`gen_ai.*` handling in this repository — so the mapping below is a sketch
against the OpenTelemetry GenAI semantic conventions **as remembered, not as
vendored**: no version of that specification is pinned in this repo, nothing
validates against it, and the conventions have been changing. It is written down
to show the hub is reachable from OTel, not as a design to implement as-is.

| IR | OTel sketch |
|---|---|
| `CanonicalTrace` | one root span per rollout |
| `trace_id` | the OTel trace id — `None` until a source provides one; never invented |
| `events[]` ordering | child span order / span events |
| `tool_call` | a child span; `name` → `gen_ai.tool.name`, `call_id` → `gen_ai.tool.call.id` |
| `usage` | `gen_ai.usage.input_tokens` / `.output_tokens` |
| `agent.model` | `gen_ai.request.model` |
| `outcome.status` | span status |
| `extensions` | span attributes, verbatim |

Which direction is wanted — receiver, emitter, or both — is **open question 5**
and stays open. The IR is symmetric on purpose so the answer does not invalidate
it.

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
    "source": "llm_proxy_normalized"
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
  invariants, the validation suite, this section, the `ACP → IR` converter and
  the `IR → ATIF` converter, each with its loss report.
- **Not implemented, deliberately:** `ATIF → IR`, any OTel work, any wiring into
  a run path, any on-disk artifact, any capture-layer enrichment.
- **Unchanged:** every existing format, exporter, artifact and code path.
  `export_atif.py` in particular is untouched and remains the only writer of
  `trainer/atif.json`.

The isolation property is unchanged in substance and restated at a new boundary:
`ir.py`, `ir_from_acp.py` and `ir_to_atif.py` form a closed family that may
import each other, and `test_only_the_ir_family_imports_the_ir` asserts that
nothing else in `src/benchflow` imports any of them. Each converter imports one
benchflow module — the IR — and reads or writes its own format as data; in
particular `ir_to_atif` does not import `export_atif`, and pins the shared schema
version by test instead.

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
