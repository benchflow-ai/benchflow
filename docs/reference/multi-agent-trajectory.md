# Multi-agent trajectory tracking through the LiteLLM proxy

Status: **design / target** (runtime-deferred, see `task-standard.md` G7). This
document records the industry comparison that motivates the design and the
contract BenchFlow should adopt so that a multi-agent workflow hosted through the
provider proxy produces **one structured trace tree** — agent identity *and*
agent-to-agent relationships — instead of one undifferentiated
`llm_trajectory.jsonl`.

## Problem

BenchFlow routes an agent's raw LLM calls through a loopback LiteLLM proxy. The
callback (`src/benchflow/providers/litellm_logging.py`) records `model + messages`
per call and keeps only `model_group` from request `metadata`
(`litellm_logging.py:130,150`) — every other tag is dropped. So when a
multi-agent workflow (a LangGraph supervisor→specialists graph, an Omnigent
session with sub-agents, concurrent arena seats) shares one proxy, the result is a
**flat, undifferentiated** log: you can see *that* N calls happened, not *which
agent* made each, nor how the agents relate.

The earlier medical-bench workaround — **one proxy per agent → one file per agent**
(`out/medical-hosted/<agent>/trajectory/llm_trajectory.jsonl`) — preserves
identity *by filename* but **destroys the relationship structure** (which agent
spawned/handed-off-to which) that every observability tool treats as first-class.
It also doesn't scale (one subprocess proxy per agent) and can't represent a
dynamic spawn tree. It was a demonstration, not the design.

## What the industry does (surveyed, verified)

A deep multi-source survey (OpenTelemetry GenAI semconv, LangSmith/LangGraph,
Langfuse, OpenLLMetry/Traceloop, Langtrace, and framework SDKs) returns one
**unanimous** structural answer:

| System | Data model | Per-agent identity carrier | Relationship mechanism | Session/seat grouping | Capture |
|---|---|---|---|---|---|
| **OTel GenAI semconv** | one trace = tree of **typed** spans | `gen_ai.agent.id` / `gen_ai.agent.name` + span name `invoke_agent {name}` + `gen_ai.operation.name` | parent/child span **nesting** (`execute_tool` nests under `invoke_agent`; agents nest under `invoke_workflow`) | `gen_ai.conversation.id` (Conditionally Required; **never synthesize** a fallback) | in-process instrumentation; cross-process via `traceparent` |
| **LangSmith / LangGraph** | run tree | `run_type` + `name` (no `run_type=agent`; an agent is a `chain` run) | `parent_run_id` + `dotted_order` + `child_run_ids` | `thread_id`; `trace_id` = root run id | LangChain callbacks (`run_id`/`parent_run_id`) |
| **Langfuse** | trace + nested observations | typed observations (generation/span/event) + `name` | `parent_observation_id` | `session_id` | SDK / decorators / OTel |
| **OpenLLMetry / Traceloop** | OTel spans | `traceloop.span.kind ∈ {workflow,task,agent,tool}` + `entity.name` + `workflow.name` | OTel context nesting via `run_id`/`parent_run_id` | — | SDK monkey-patch + LangChain `BaseCallbackHandler`; injects `extra_headers` to **propagate** context |
| **Langtrace** | OTel spans ("adhere to OTEL") | OTel `gen_ai.*` attributes | OTel parent nesting | — | OTel instrumentation |
| **OpenAI Agents SDK** | trace + spans | agent span; **handoffs** are edges | root span via `Runner` | `group_id` | SDK tracing |
| **LiteLLM proxy** (BenchFlow today) | flat callback log | `metadata` body field + headers — but callback keeps only `model_group` | none natively (must add a parent pointer in metadata) | `metadata` | proxy callback (`StandardLoggingPayload`) |

Three load-bearing facts, each confirmed 3-0 across independent verifiers:

1. **A multi-agent run is ONE trace = a tree of typed spans joined by explicit
   parent pointers — never a flat event log.** Universal across OTel, LangSmith,
   Langfuse, Langtrace, OpenLLMetry.
2. **Per-agent identity is carried *on each call* (span attributes + span name),
   so a single shared stream is differentiated per-call, not per-stream.** Agent /
   tool / LLM-call / orchestration are first-class, **distinct span types**
   (`gen_ai.operation.name`: `chat`, `invoke_agent`, `execute_tool`,
   `invoke_workflow`, …; LangSmith `run_type`: chain/llm/tool/retriever/…).
3. **Relationships are captured at call time via parent/child nesting, not inferred
   post-hoc** — `execute_tool` under `invoke_agent`; LangChain `run_id` →
   `parent_run_id`. Session/seat grouping uses a single real conversation id; the
   OTel spec **forbids synthetic fallbacks** (no UUID / trace-id / content hash).

**Critical caveat for BenchFlow:** none of these tools solve the
"shared collector loses the tag" problem *with an HTTP proxy*. They instrument
**in-process** (framework callbacks, SDK monkey-patching) where the active
agent/parent context is already known, and propagate it across process boundaries
via OTel context (`traceparent`). BenchFlow's proxy sits *outside* the agent
process, so the agent context must be **explicitly attached to each request** — the
proxy cannot recover it otherwise. (Also: OTel GenAI agent conventions are
*experimental* / SHOULD-level; only `gen_ai.operation.name` and
`gen_ai.provider.name` are strictly Required. `gen_ai.agent.*` officially live on
agent-lifecycle spans, not every raw chat span — attaching them per raw call is a
deliberate, reasonable BenchFlow extension.)

## The design: one pooled proxy + per-call metadata → one trace tree

Adopt the dominant industry shape, adapted for an out-of-process proxy:

**1. Each LLM request carries agent context in `metadata`** (the LiteLLM request
body field, which the proxy forwards to the logging callback — see *Verification*
below). A minimal, OTel-aligned schema:

```jsonc
"metadata": {
  "bf.agent_id":        "answer",          // stable id of the calling agent/node  (~ gen_ai.agent.id)
  "bf.agent_name":      "answer",          // human label                          (~ gen_ai.agent.name)
  "bf.span_kind":       "chat",            // chat | invoke_agent | execute_tool | invoke_workflow  (~ gen_ai.operation.name)
  "bf.parent_agent_id": "supervisor",      // parent pointer → reconstructs the tree (~ parent_run_id)
  "bf.session_id":      "medical-run-1",   // real conversation/seat id, NEVER synthesized (~ gen_ai.conversation.id)
  "bf.run_id":          "answer#2"         // this call's id, so children can point at it
}
```

**2. The callback records a span row** instead of today's bare model+messages line:
`litellm_logging.py:_base_record` stops discarding `metadata` and persists
`bf.agent_id` / `bf.agent_name` / `bf.span_kind` / `bf.parent_agent_id` /
`bf.session_id` / `bf.run_id` alongside the existing `model_group`. Every proxied
call becomes one typed span with a parent pointer.

**3. The trajectory becomes a tree.** `trajectory_from_litellm_callback_log`
reconstructs parent/child structure from `bf.parent_agent_id` / `bf.run_id`
exactly as LangSmith does from `parent_run_id` / `dotted_order`. One
`llm_trajectory.jsonl` then holds the whole multi-agent run, splittable per agent
**and** navigable as a tree — no per-agent files, no lost relationships.

Why this over "one proxy per agent": separate files preserve identity by filename
only and throw away the parent/child + handoff structure that is the *point* of a
multi-agent trace. The pooled-proxy + per-call-tag model is what every surveyed
tool does.

### The uniform adapter

BenchFlow cannot adopt a single framework as "the" multi-agent host (Omnigent,
the closest candidate, explicitly does **not** host LangGraph/CrewAI/AutoGen — see
below). The uniform layer is instead a **thin BenchFlow-side contract**: each
framework's native per-agent + parent context is mapped onto the one `metadata`
schema by a small per-framework shim, before the call leaves the agent process.

| Framework | Native per-agent + parent context the shim maps from |
|---|---|
| LangChain / LangGraph | `BaseCallbackHandler` `run_id` / `parent_run_id` / node name → `bf.*` (already exposed) |
| OpenAI Agents SDK | `Runner` / root span + agent name + handoff edges |
| Omnigent | its internal conversation tree (`parent_conversation_id`/`root_conversation_id`/`agent_id`) → `bf.*` |
| custom (e.g. our medical slice) | the node passes its own name as `bf.agent_id` when it builds the `ChatOpenAI` call |
| AutoGen / CrewAI / Swarm | **under-evidenced** — per-agent + handoff exposure to an interceptor not yet confirmed; needs a per-framework spike before claiming support |

The shim's only job is the mapping. Identity and relationships already exist
inside every framework; BenchFlow just needs them attached to the request.

### Unified `bf.*` vocabulary (consolidated with the adapter proposal)

The adapter proposal (PR #847) independently specified a richer per-call
attribution set. To avoid forking two schemas, that vocabulary is folded into the
one `bf.*` namespace. The callback captures **any** `bf.*` key generically (it
strips the `bf.` prefix from every metadata key), so the extended dimensions flow
through with **no code change** (verified in `tests/test_litellm_logging.py`).

| `bf.*` key | status | meaning | OTel / #847 analogue |
|---|---|---|---|
| `agent_id` / `agent_name` | implemented | the calling agent/node | `gen_ai.agent.id` / `.name` |
| `parent_agent_id` | implemented | parent pointer (tree edge) | `parent_run_id` / `framework_parent_id` |
| `run_id` | implemented | this call's id | `llm.call_id` |
| `session_id` | implemented | conversation/seat/rollout id (never synthesized) | `gen_ai.conversation.id` / `rollout_id` |
| `span_kind` | implemented | `chat`/`invoke_agent`/`execute_tool`/`invoke_workflow` — the `relation` hook | `gen_ai.operation.name` |
| `role` | extended (#847) | declared role (planner/implementer/reviewer) | `role` |
| `scene` / `turn_index` | extended (#847) | scene + turn within a scene | `scene` / `turn` |
| `team_id` | extended (#847) | team / sub-graph grouping | `team_id` |
| `framework` / `framework_node_id` | extended (#847) | framework name + native node id | `framework` / `framework_node_id` |
| `trace_id` | extended (#847) | cross-process / cross-framework trace correlation | `trace_id` |

### Uniform adapter protocol (from #847)

A hosted framework is wrapped by a thin adapter — a *normalizer*, not a
reimplementation — with four operations:

- `detect(task_dir, spec)` — can this adapter host the workflow?
- `prepare(ctx)` — install deps, write the LiteLLM env, compile the launch command.
- `run(ctx)` — run the external workflow **inside the BenchFlow sandbox**.
- `collect(ctx)` — gather native logs, normalize to BenchFlow events + graph edges,
  return an `AdapterTraceBundle` (framework-native raw logs under
  `trajectory/raw/<adapter>/` + coverage diagnostics: attribution quality, missing
  metadata, unsupported relations, redaction state).

Declared under a `benchflow.multi_agent` block (target / `benchflow:`-namespaced):
`capture_raw_llm: required` **fails closed** when zero LLM calls are captured;
`relationship_graph: required` persists `trajectory/agent_graph.json` (agent / team /
sub-graph nodes + edges carrying a `relation` ∈ {delegates, supervises, reviews,
handoff, parallel_child, fan_in, …}). When a framework **cannot** inject per-call
metadata, fall back to **one LiteLLM virtual key / route alias per role** as the
attribution channel. Report multi-agent lift only against a **matched single-agent
baseline** (same task set, tools, answer contract, usage + logging).

The implemented `build_agent_tree()` is the minimal in-memory realization of this
graph (one parent pointer); `agent_graph.json` is its richer persisted form, with
`bf.span_kind` as the existing per-edge `relation` hook.

## Does Omnigent satisfy this? (assessed separately, verified)

Short answer: **not as wired today.** Omnigent the *framework* is a strong
multi-agent foundation, but the integration does not carry that across the
BenchFlow boundary. Per-requirement:

- **Multi-agent support** — framework yes (recursive `AgentSpec.sub_agents`,
  `sys_session_send` spawns children, parent-linked conversation tree); the BF
  adapter drives a **single one-shot harness** (`omnigent run -p`) and only `pi`
  is wired end-to-end. *Partial.*
- **Per-agent attribution through the proxy** — **fails.** Omnigent's HTTP adapter
  sends only `{model, messages, tools?, stream?, **extra}` with `Content-Type` +
  `Authorization` headers — **no agent id on the wire** — and BenchFlow's callback
  would drop it anyway. `agent_id` lives only in Omnigent's DB.
- **Relationships** — Omnigent models a real parent-linked tree + shared OTel trace
  internally (`db_models.py`), but it is **not populated by the one-shot CLI path**
  (each `omnigent run -p` is a fresh, unlinked conversation, no `traceparent`) and
  is never exported to BenchFlow. *Partial / not wired.*
- **Proxy routing of ALL traffic** — only `pi` (OpenAI-wire) is proven proxied;
  native CLI sub-harnesses (Claude Code/Codex) use a separate
  `HARNESS_*_GATEWAY_BASE_URL` / `claude_gateway_shim` channel the adapter never
  sets, and sub-agent spawns pass no gateway override → **bypass risk**.
- **Uniform host for many frameworks** — hosts coding-agent CLIs + `claude_sdk` +
  `agents_sdk`, but LangGraph/CrewAI/AutoGen/LangChain/DeepAgents are explicitly
  "Not natively supported." *Not a universal host.*

Two paths to make Omnigent conformant (complementary): **(A)** inject `agent_id`
into proxy metadata + have the callback record it (the contract above) — best for
per-call cost/attribution; **(B)** run Omnigent in server/session mode and export
its native conversation tree, joined to proxy usage by `agent_id` — best for the
relationship graph. `agent_id` from (A) is the clean join key for (B).

## Implementation phases

- **P0 ✅ DONE — callback records attribution.** `litellm_logging.py:_base_record`
  now extracts every `bf.*` key from request `metadata` into `request.body['bf']`
  (prefix stripped) instead of keeping only `model_group`. The trajectory importer
  carries it through unchanged. Unit-tested in `tests/test_litellm_logging.py`.
- **P1 ✅ DONE — medical bench as proof, one proxy.** `examples/medical`'s `_llm`
  tags each node's call with `bf.*` via `extra_body` (the verified client
  mechanism); `trajectories/build_agent_tree()` reconstructs the unmixed agent
  tree. Verified end-to-end on **three backends**, all producing the identical
  tree `supervisor(1) → answer(2) → guardrail(1)`, `unmixed_ok: true`:
  `run_agent_tree.py` (local, docker-proxy) and `run_in_sandbox.py` (agent running
  **inside** a docker container and a remote daytona sandbox, through the proxy).
  Unit-tested in `tests/trajectories/test_agent_tree.py`.
- **P2 — tree-shaped canonical trajectory.** Add a multi-agent trajectory kind that
  reconstructs + emits the tree (parent pointers / dotted-order); fix
  `n_tool_calls` to count real tool spans.
- **P3 — OTLP export option.** Optionally emit OTLP spans so existing backends
  (Langfuse, Arize Phoenix/OpenInference, OpenLLMetry collectors) can render the
  tree, instead of (or alongside) the bespoke JSONL.
- **Omnigent track.** Wire `HARNESS_*_GATEWAY_*` at the proxy to close the bypass;
  drive server/session mode; export the conversation tree.
- **Concurrent-seats track ✅ DONE — native multi-agent floor.** `benchflow arena run
  --agents agents.yaml` (`src/benchflow/arena/`) runs N agents on ONE shared task +
  service CONCURRENTLY in ONE shared sandbox, each in `/work/<seat>`, each with its
  own ACP trajectory and — for proxy seats — a separate raw `llm_trajectory.jsonl`
  from that seat's own proxy (`session_id=floor-<seat>`). This answers the "concurrent
  seats" open question below: **distinct per-seat files**, separated at the proxy by a
  per-seat `bf.session_id`, rather than sibling sub-trees under one trace (which remains
  the model for a single agent's supervisor→specialist calls). Agents resolve from all
  three benchflow-ai/agents paths (raw ACP / ai-sdk / omnigent) or a BYOA manifest, and
  carry a per-agent instruction file (`CLAUDE.md`/`GEMINI.md`/`AGENTS.md`). See
  `examples/arena/README.md`. Unit-tested in `tests/test_{agents_manifest,agent_driver,
  agent_instructions,concurrent_floor,arena_cli}.py`.

## Verification (the research's #1 open question)

The whole design rests on: *does a request-body `metadata` field survive to the
LiteLLM proxy logging callback?* This was the survey's top open question. It is
**confirmed empirically** against BenchFlow's own loopback proxy: a single chat
completion sent with `metadata: {bf.agent_id, bf.agent_name, bf.span_kind,
bf.parent_agent_id, bf.session_id, bf.run_id}` had **all six fields arrive intact**
at the callback (verdict PASS). The callback already *reads*
`litellm_params.metadata` (`litellm_logging.py:130`) — LiteLLM merges the request
body's `metadata` into it — so the only missing piece is *recording* the agent
fields instead of keeping just `model_group` (P0).

## Open questions

- AutoGen / CrewAI / OpenAI Swarm: how each exposes per-agent identity + explicit
  handoff **edges** to an interceptor (needs a per-framework spike).
- Bespoke JSONL vs native OTLP export (P3) — what is lost/gained by staying custom.
- Concurrent seats vs supervisor→specialist: sibling sub-trees under one trace, OTel
  span links, or distinct `bf.session_id` per seat?

## Sources

OpenTelemetry GenAI spans + agent spans
(`opentelemetry.io/docs/specs/semconv/gen-ai/`), LangSmith run-data-format
(`docs.langchain.com/langsmith/run-data-format`), Langfuse data model
(`langfuse.com/docs/observability/data-model`), OpenLLMetry/Traceloop semantic
conventions (`traceloop.com/docs/openllmetry`), Langtrace
(`github.com/Scale3-Labs/langtrace`). 27 sources fetched; 22 claims confirmed, 3
refuted, across 110 research agents.
