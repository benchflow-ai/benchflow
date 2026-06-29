# Multi-agent workflow adapters and tracing

Status: proposal.

BenchFlow already supports native Scene / Role / Turn execution. This document defines the next layer: a uniform adapter contract for external multi-agent workflow frameworks and relationship-aware trajectory artifacts backed by LiteLLM proxy logging.

The feature should preserve each framework's native workflow semantics while emitting one BenchFlow artifact set that can answer:

- which agent acted;
- which agent delegated, reviewed, supervised, or handed off to another agent;
- which LLM calls produced each visible message, tool call, or handoff;
- how much each agent, scene, team, branch, and external workflow node cost;
- how the multi-agent run compares with a matched single-agent baseline.

## Scope notes

I could not identify a public multi-agent benchmark or workflow project named `lingtai` under that spelling. Keep `lingtai` as a reserved adapter id until a repository, package, or workflow contract is supplied.

Comparable public systems considered here:

- LangGraph: graph nodes, edges, subgraphs, routing, parallel workers, persistence, and streamed state.
- AutoGen AgentChat: teams such as RoundRobin, Selector, MagenticOne, and Swarm handoff.
- CrewAI: crews of agents and tasks with sequential or hierarchical manager-driven execution.
- HARBOR: bounded stages executed by specialized agents through commands, gates, artifacts, and parallel trials.
- BenchAgent: protocol-aligned comparison of single-agent, fixed multi-agent, and evolving multi-agent workflows.

## Current BenchFlow baseline

BenchFlow-native multi-agent support is centered on declarative scenes, roles, and turns. A multi-role rollout executes in one sandbox, switches active roles, reconnects the selected agent session, and keeps the workspace as the explicit handoff medium.

The current trajectory path is ACP-first: `trajectory/acp_trajectory.jsonl` records ACP-visible user messages, agent messages, thoughts, tool calls, and timeouts. That is enough for linear BenchFlow-authored role handoff. It is not enough for external multi-agent workflows that contain nested graphs, parallel workers, supervisor delegation, manager validation, dynamic agent spawning, or framework-native state that never appears in ACP events.

The missing layer is a normalized multi-agent event graph plus proxy-level LLM-call capture.

## Comparison

| System | Native primitive | Relationship semantics | Trace surface | BenchFlow implication |
|---|---|---|---|---|
| BenchFlow native | `agents.roles`, `scenes`, `turns`, `benchflow.teams` | Linear role turns and sequential shared-workspace handoff | ACP trajectory and rollout tree | Keep as canonical simple path; enrich every event with scene, turn, role, and handoff metadata. |
| LangGraph | Graph nodes, edges, subgraphs, dynamic workers | Routing, parallel fan-out/fan-in, orchestrator-worker, nested subgraphs | Framework state, streams, and LangSmith traces | Adapter must preserve node id, subgraph id, parent graph id, and dynamic worker edges. |
| AutoGen AgentChat | Team presets and participant messages | Shared context, selected speaker, round-robin order, handoff messages | Streamed team messages and `TaskResult.messages` | Adapter maps participant `source` to agent node and message order to speaker/handoff edges. |
| CrewAI | Crews of agents and tasks | Task sequence, manager delegation, validation before proceeding | Crew output, task outputs, usage metrics, execution hooks/logs | Adapter maps tasks to scene spans and manager delegation to `supervises` / `delegates` edges. |
| HARBOR | Bounded stages with specialized agents | Stage gates, persistent artifacts, reusable knowledge, parallel trials | Stage logs, commands, gates, artifacts | Adapter maps stages to scenes, gates to verifier/checkpoint events, artifacts to evidence. |
| BenchAgent | Normalized workflow protocol | Fixed and evolving MAS under matched loader, tools, answer contract, usage, and trajectory logging | Protocol-aligned execution and logs | BenchFlow reports multi-agent lift only against matched single-agent baselines and cost. |
| Generic OpenAI-compatible workflow | Any command that calls an OpenAI-compatible endpoint | Unknown unless app supplies metadata | LiteLLM proxy request/response logs | Best-effort adapter: reliable LLM calls, weaker relationship graph. |

## Target artifacts

Existing artifacts remain stable:

- `trajectory/acp_trajectory.jsonl`
- `result.json`
- trainer artifacts such as ATIF / ADP

Add multi-agent artifacts:

- `trajectory/llm_raw.jsonl` — one record per LiteLLM proxy request/response, redacted according to policy.
- `trajectory/multiagent_events.jsonl` — normalized relationship-aware events from ACP, LiteLLM, and framework-native logs.
- `trajectory/agent_graph.json` — nodes and edges for the run-level agent/workflow DAG.
- `trajectory/index.json` — schema versions, counts, checksums, coverage diagnostics, and artifact pointers.
- `trajectory/redaction_report.json` — whether message bodies, tool arguments, and model responses were persisted, hashed, or omitted.
- `viewer/multiagent_timeline.json` — denormalized UI view for swimlanes, fan-out/fan-in, handoffs, nested calls, and cost overlays.

## Normalized event fields

Every `multiagent_events.jsonl` record should be stable, addressable, and relationship-aware.

Required in M0:

- `schema_version`
- `event_id`
- `rollout_id`
- `timestamp`
- `source`: `acp`, `litellm_proxy`, or `framework_native`
- `framework`: `benchflow-native`, `langgraph`, `autogen`, `crewai`, `harbor`, `lingtai`, or `generic-openai-compatible`
- `adapter`
- `scene`
- `turn_index`
- `team_id`
- `agent_id`
- `role`
- `relation`
- one of `llm.call_id`, `framework_run_id`, or `acp_event_index`

Suggested optional fields:

- `framework_node_id`
- `framework_parent_id`
- `parent_event_id`
- `root_event_id`
- `related_agent_id`
- `handoff_from`
- `handoff_to`
- `branch_id`
- `subgraph_id`
- `span_id`
- `llm.model`
- `llm.provider`
- `llm.status`
- `llm.prompt_tokens`
- `llm.completion_tokens`
- `llm.total_tokens`
- `llm.cost_usd`
- `llm.prompt_sha256`
- `llm.response_sha256`
- `llm.raw_request_path`
- `llm.raw_response_path`

Suggested `relation` enum:

- `starts`
- `responds_to`
- `delegates`
- `supervises`
- `handoff`
- `reviews`
- `critiques`
- `revises`
- `parallel_child`
- `fan_in`
- `tool_subagent`
- `llm_call`
- `verifies`
- `terminates`

`agent_graph.json` should summarize the run-level structure: agent nodes, team nodes, subgraph nodes, tool-subagent nodes, and edges with `relation`, `first_event_id`, `last_event_id`, and per-edge metrics.

## LiteLLM proxy tracing

Use LiteLLM as the raw LLM-call capture layer because most external workflows can be pointed at an OpenAI-compatible endpoint even when their native event model differs.

Implementation requirements:

1. Start one LiteLLM proxy per rollout or per job, depending on isolation and concurrency.
2. Attach a BenchFlow custom callback to persist LiteLLM's standard logging payload for every successful and failed call.
3. Pass BenchFlow metadata in every request when the adapter controls the client: `rollout_id`, `scene`, `turn`, `team_id`, `agent_id`, `role`, `framework`, `framework_node_id`, `parent_event_id`, and `trace_id`.
4. Pass tags for quick filtering, such as `benchflow:rollout:<id>`, `agent:planner`, and `framework:langgraph`.
5. If a framework cannot pass per-request metadata, use one LiteLLM virtual key or route alias per role as a fallback attribution channel.
6. Persist `x-litellm-call-id` and map it to normalized events through `llm.call_id`.
7. Redact or hash prompt/response bodies when task policy disables body persistence.
8. Fail closed when `capture_raw_llm: required` and no LiteLLM calls are captured.

## Uniform adapter protocol

Adapters should be thin normalizers around external workflow runtimes. They should not reimplement the framework.

Required adapter operations:

- `detect(task_dir, spec)`: decide whether the adapter can host the workflow.
- `prepare(ctx)`: install dependencies, write LiteLLM env, and compile the launch command.
- `run(ctx)`: run the external workflow inside the BenchFlow sandbox.
- `collect(ctx)`: collect native logs and normalize them into BenchFlow events and graph edges.

`AdapterTraceBundle` should contain:

- normalized events for `multiagent_events.jsonl`;
- graph nodes and edges for `agent_graph.json`;
- framework-native raw logs under `trajectory/raw/<adapter>/`;
- coverage diagnostics: attribution quality, missing metadata, unsupported relationship semantics, and redaction state.

Built-in adapters:

| Adapter | First behavior |
|---|---|
| `benchflow-native` | Enrich current Scene / Role / Turn events with scene, turn, role, agent, and handoff edges. |
| `generic-openai-compatible` | Run any declared command through the LiteLLM proxy and attribute calls from metadata, tags, virtual keys, and process env. |
| `autogen` | Capture streamed team messages and final task result messages; map message `source` to agents. |
| `crewai` | Capture crew task outputs, usage metrics, process mode, manager/delegation relationships, and available hooks. |
| `langgraph` | Capture node, edge, subgraph, checkpoint/thread, and dynamic worker mappings. |
| `harbor` | Preserve stage, command, gate, artifact, specialized-agent, and parallel-trial metadata. |
| `lingtai` | Reserved until a concrete workflow contract is supplied. |

## `task.md` authoring surface

Keep stable orchestration fields at the existing root keys: `agents`, `scenes`, and `user`. Put adapter-specific and experimental fields under the reserved `benchflow` namespace until they are promoted into typed task-standard models.

```yaml
agents:
  roles:
    orchestrator:
      agent: python-workflow
      model: litellm/gpt-5.5
    planner:
      agent: external
      model: litellm/gpt-5.5
    implementer:
      agent: external
      model: litellm/gpt-5.5
    reviewer:
      agent: external
      model: litellm/gpt-5.5

scenes:
  - name: external-workflow
    turns:
      - role: orchestrator

benchflow:
  multi_agent:
    adapter: langgraph
    mode: external-workflow
    entrypoint: workflows.design_review:run
    workflow_root: workflow/
    trace:
      llm_proxy: litellm
      capture_raw_llm: required      # required | best-effort | disabled
      capture_framework_events: best-effort
      relationship_graph: required   # required | best-effort | disabled
      redact_messages: false
    agents:
      mapping:
        planner: {role: planner, framework_node: plan_node}
        implementer: {role: implementer, framework_node: implement_node}
        reviewer: {role: reviewer, framework_node: review_node}
    relationships:
      allowed: [delegates, reviews, handoff, parallel_child, fan_in]
```

Validation rules:

- `benchflow.multi_agent.adapter` must be known or explicitly set to `generic-openai-compatible`.
- `capture_raw_llm: required` requires a LiteLLM runtime and at least one captured LLM call.
- Every mapped role must reference a declared `agents.roles` role.
- `relationship_graph: required` must fail closed if the adapter cannot emit parent/child or handoff edges.
- Message persistence must obey task-level privacy and redaction policy.

## Viewer requirements

The viewer should expose three synchronized views:

1. Swimlane timeline — one lane per agent/role with LLM calls, tool calls, messages, handoffs, and verifier checkpoints.
2. Relationship DAG — supervisor, delegation, handoff, review, parallel child, fan-in, and subgraph edges.
3. Cost/outcome overlay — tokens, cost, latency, tool calls, and reward by agent, scene, team, branch, and whole rollout.

Clicking an event should show normalized event JSON, raw LiteLLM payload pointer if allowed, framework-native raw event pointer, ACP event pointer when available, and parent/child links.

## Benchmark comparison policy

Multi-agent support should not assume that more agents are better. Every report should include matched baselines:

- same task set;
- same tool access;
- same answer contract;
- same usage accounting;
- same trajectory logging coverage;
- single-agent baseline, fixed multi-agent workflow, and dynamic/evolving workflow when available.

Report reward/pass rate, token cost, wall-clock latency, LLM calls, tool calls, active agents, relationship graph coverage, raw LLM capture coverage, and attribution failures.

## Milestones

### M0 — native enrichment and schema

- Add `benchflow.multi_agent` task.md parser surface under the raw `benchflow` namespace.
- Enrich BenchFlow-native Scene / Role / Turn ACP events with `scene`, `turn_index`, `role`, `agent_id`, and sequential handoff edges.
- Add artifact writers for `multiagent_events.jsonl`, `agent_graph.json`, `index.json`, and `redaction_report.json`.
- Add a schema-only task.md fixture and docs.

### M1 — LiteLLM raw trajectory capture

- Add a BenchFlow LiteLLM custom callback that writes `llm_raw.jsonl`.
- Add request metadata/tag injection for native roles and adapter launches.
- Add required/best-effort/disabled policy and fail-closed validation.
- Add per-agent usage rollups in `result.json` and `summary.json`.

### M2 — first external adapters

- Implement `generic-openai-compatible`, `autogen`, and `crewai` adapters.
- Preserve framework-native logs under `trajectory/raw/<adapter>/`.
- Map native messages/tasks to normalized events and graph edges.

### M3 — graph-native adapter

- Implement `langgraph` adapter with node, edge, subgraph, checkpoint/thread, and dynamic worker mapping.
- Add fan-out/fan-in and subgraph visualization.

### M4 — benchmark-platform parity and arena concurrency

- Implement `harbor` adapter once the concrete Harbor task/workflow contract is selected.
- Add `lingtai` adapter once a concrete workflow contract is supplied.
- Promote stable fields from `benchflow.multi_agent` into typed task-standard models.
- Integrate with future arena-concurrent / A2A support.

## References

- AutoGen AgentChat teams: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/teams.html
- CrewAI crews: https://docs.crewai.com/en/concepts/crews
- LangGraph workflow/agent patterns: https://docs.langchain.com/oss/python/langgraph/workflows-agents
- LangGraph subgraphs: https://docs.langchain.com/oss/python/langgraph/use-subgraphs
- LiteLLM proxy logging: https://docs.litellm.ai/docs/proxy/logging
- LiteLLM custom callbacks: https://docs.litellm.ai/docs/observability/custom_callback
- LiteLLM StandardLoggingPayload: https://docs.litellm.ai/docs/proxy/logging_spec
- HARBOR paper: https://arxiv.org/abs/2606.08610
- BenchAgent paper: https://arxiv.org/abs/2606.05670
