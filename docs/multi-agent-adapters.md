# Multi-agent real-agent adapters and tracing

Status: proposal plus M0 implementation slice.

This design is about **real agent runs**, not graph-framework LLM calls. A BenchFlow multi-agent run should launch, supervise, and trace independent agent sessions such as Claude Code, Codex, Gemini CLI, OpenHands, or any ACP / CLI / A2A-compatible agent. LangGraph-style LLM workflows can be supported later as a low-priority adapter, but they are not the target primitive.

The primary trace source is the agent session transcript: ACP events when the agent speaks ACP, native transcript scraping when the agent has its own session log, or a session-factory wrapper when the agent is only available as a CLI or SDK. LiteLLM proxy capture is optional auxiliary evidence for agents that expose model calls through an OpenAI-compatible endpoint. It must not be the main multi-agent abstraction because subscription agents and opaque coding agents may not expose raw provider calls.

## Implemented in this PR

The first executable slice is implemented for BenchFlow-native Scene / Role / Turn execution:

- `src/benchflow/trajectories/multiagent.py` adds `RealAgentTraceRecorder` and session/handoff records.
- `src/benchflow/rollout/_user_loop.py` now records every real `connect_as(role)` session in `_run_steps()` and user-loop rounds.
- Existing `trajectory/acp_trajectory.jsonl` remains the merged compatibility view.
- New real-agent artifacts are emitted when a rollout directory exists:
  - `trajectory/sessions.jsonl`
  - `trajectory/handoffs.jsonl`
  - `trajectory/multiagent_events.jsonl`
  - `trajectory/agent_graph.json`
  - `trajectory/agents/<role>/<session>/acp.jsonl`
- Tests cover the pure recorder and scene integration.

Still not implemented here: MCP delegation, A2A remote agents, native team import, and workspace diff capture.

## Desired behavior

BenchFlow should be able to run a real multi-agent evaluation like this:

1. start Claude Code as `planner`;
2. stop or pause that session and save its isolated transcript;
3. start Codex as `implementer` using only the declared handoff artifact;
4. start Gemini CLI as `reviewer` with read-only policy;
5. optionally let a supervisor delegate to worker agents through a BenchFlow MCP tool or A2A bridge;
6. show a single relationship graph over the isolated agent trajectories;
7. score the final environment with the normal verifier.

The core rule: **each agent gets its own session and own trajectory; BenchFlow adds the relationships.** Shared context is explicit through files, diffs, messages, task ids, or declared handoff artifacts, never by silently merging conversation histories.

## Existing work to learn from

| Work | What it gives us | BenchFlow adapter implication |
|---|---|---|
| BenchFlow current native scenes | Real ACP-backed roles run sequentially in one sandbox. Each role can use a different `agent` and `model`; context is reset by reconnecting. | This PR implements the first M0 trace layer there: per-role session artifacts plus relationship metadata. |
| Agent Client Protocol (ACP) | Client-to-coding-agent protocol for local or remote coding agents. Local agents can run as subprocesses over JSON-RPC stdio. | BenchFlow is the ACP client/orchestrator. Use ACP sessions as the first-class real-agent driver. |
| Claude Code subagents and agent teams | Real Claude Code subagents/teammates have separate context windows. Agent teams have a lead, teammates, task list, mailbox, and direct teammate messaging. | Borrow the data model: lead, teammate, task, mailbox, isolated context, task claiming, cross-agent messages. Do not assume all agents are Claude. |
| A2A Protocol | Agent-to-agent task protocol with Agent Cards, task ids, context ids, streaming task updates, artifacts, and terminal task states. | Add an `a2a-remote-agent` adapter. Map A2A tasks/artifacts/status updates into session and handoff events. |
| MCP | Tool/resource/prompt protocol, not agent-to-agent by itself. | Add a `benchflow_delegate_to_agent` MCP server. A supervisor agent calls this tool; BenchFlow starts a real worker session and links the worker transcript to the supervisor tool call. |
| HARBOR | Staged automation with specialized agents, standardized commands, persistent artifacts, gates, and parallel trials. | Model stages as scenes, gates as verifier/checkpoint events, artifacts as handoff/evidence, and parallel trials as forked session groups. |
| BenchAgent | Normalized evaluation of single-agent, fixed MAS, and evolving MAS workflows under common loader, tools, answer contract, usage accounting, and trajectory logging. | Require matched baselines and report cost/accuracy tradeoffs. Multi-agent is not assumed better. |
| SWE-Interact | Converts SWE tasks into vague, progressive user-driven tasks; the LLM user simulator reveals requirements and inspects the workspace. | Useful as a stress test, but BenchFlow should add an explicit intent/reveal/drift audit layer before treating simulator behavior as benchmark-stable. |
| SWE-Together | Reconstructs 109 real user-agent coding sessions from 11,260 raw sessions; uses anchored, state-conditional replay with decomposed intents, trigger conditions, Intent Coverage, and User Correction. | Prefer SWE-Together's anchored replay discipline for BenchFlow interactive-user tasks: immutable intent objects, trigger-gated feedback, simulator-fidelity metrics, and correction-cost reporting. |
| AaaS-AN / Agent Network | Agents and agent groups are vertices; routes are edges; the scheduler maintains an execution graph and context isolation. | Use this as the long-term shape for dynamic networks: session nodes plus route/handoff edges. |
| `lingtai` | No public repo/package/spec found under that spelling during this pass. | Keep as a reserved adapter id until an actual contract is provided. |

## Interactive-user benchmark comparison

SWE-Interact and SWE-Together are both interactive coding-agent benchmarks, but they make different choices about simulator control.

| Axis | SWE-Interact | SWE-Together | BenchFlow target |
|---|---|---|---|
| Source tasks | Converts existing SWE-style tasks into multi-turn user-driven variants. | Reconstructs tasks from real user-agent coding sessions with recoverable repositories, clear goals, and observable outcomes. | Support both converted tasks and recorded-session imports, but preserve provenance and validation evidence separately. |
| User simulator | Model-backed user starts vague, reveals requirements, inspects workspace, and gives feedback. | Reactive LLM simulator anchored to original session analysis; speaks only when trigger conditions arise in the evaluated trajectory. | `BaseUser` plus `IntentSpec`: deterministic/scripted users first, model users second, both audited against immutable intent. |
| Drift handling | Drift is mainly visible as a post-hoc failure mode such as missing requirements. | The paper explicitly names interaction drift as a threat and uses Intent Coverage to audit recall and scope precision. | First-class drift audit: no new intent, no contradiction, no hidden-test leak, requirement reveal ledger, and simulator-failure channel. |
| Turn timing | Progressive reveal across rounds. | State-conditional replay: no-op unless feedback is warranted by the live trajectory. | User loop should allow no-op turns and trigger-gated feedback, not only fixed scripted rounds. |
| Metrics | Final correctness plus trajectory/failure analysis. | Final correctness plus User Correction and Intent Coverage. | Final reward plus user effort, correction/nudge counts, ask-user call quality, intent coverage, drift violations, and simulator failure rate. |
| Runtime exposure | Public repo primarily provides task data and Harbor configs. | Public repo exposes tasks, launcher, eval code, agent harness choices, and task images. | BenchFlow should keep simulator runtime, user tool calls, intent ledger, and drift audit as first-class artifacts. |

SWE-Together is closer to the benchmark contract BenchFlow should adopt for user simulation because it separates task correctness from simulator fidelity. Its Intent Coverage is especially important: it measures whether replayed simulator messages preserve original-session intents and stay within scope, so low intent coverage can be treated as simulator instability rather than agent failure. BenchFlow should generalize that into a typed `IntentSpec` and `intent_reveal_ledger.jsonl` rather than relying only on post-hoc LLM judging.

The key BenchFlow delta is an explicit artifact contract:

```text
trajectory/user/ask_user_calls.jsonl
trajectory/user/intent_reveal_ledger.jsonl
trajectory/user/drift_audit.jsonl
trajectory/user/user_tool_calls.jsonl
trajectory/user/simulator_decisions.jsonl
```

These artifacts should exist for both SWE-Interact-style progressive tasks and SWE-Together-style recorded-session replay. Publication-grade interactive tasks should fail closed when the intent ledger or drift audit is missing.

## Architecture

### 1. Real agent session is the primitive

The new `RealAgentSession` record tracks every launched agent instance:

```json
{
  "session_id": "sess_planner_001",
  "agent_id": "planner",
  "agent_type": "claude-agent-acp",
  "model": "claude-sonnet-4-6",
  "driver": "acp",
  "workspace_mode": "shared",
  "trajectory_path": "trajectory/agents/planner/sess_planner_001/acp.jsonl",
  "native_transcript_path": null,
  "workspace_diff_path": null,
  "handoff_out": ["handoff_001"]
}
```

Supported drivers target:

- `acp`: existing BenchFlow ACP connection path.
- `session-factory`: Python session wrapper for SDK-only agents.
- `cli`: prompt-in, transcript/log-out wrapper for CLI agents without ACP.
- `a2a`: remote agent task execution through A2A.
- `mcp-delegate`: supervisor tool call that launches another real agent through BenchFlow.
- `native-team-import`: ingest an agent's own team transcript when BenchFlow did not supervise every spawned member directly.

### 2. Multi-agent run is a graph of sessions

`trajectory/agent_graph.json` has nodes for agent sessions and handoff artifacts, and edges such as `handoff_to`, `delegates_to`, `supervises`, `reviews`, `critiques`, `revises_after`, `messages`, `produces`, `consumed_by`, `branches_to`, `merges_into`, and `verifies`.

### 3. Per-agent isolated trajectories

Do not collapse all agent events into one flat ACP file. Preserve both a merged compatibility view and the isolated source views:

```text
trajectory/
  acp_trajectory.jsonl
  multiagent_events.jsonl
  agent_graph.json
  sessions.jsonl
  handoffs.jsonl
  agents/
    planner/sess_planner_001/acp.jsonl
    planner/sess_planner_001/native.jsonl
    planner/sess_planner_001/workspace.diff
    implementer/sess_impl_001/acp.jsonl
    reviewer/sess_reviewer_001/acp.jsonl
```

`multiagent_events.jsonl` should contain normalized event pointers and relationship metadata. Heavy transcript content stays under per-session directories.

### 4. Handoff modes

| Mode | Meaning | Implementation |
|---|---|---|
| `sequential-shared-workspace` | Agents run one after another in the same sandbox, with fresh context and shared files. | Current scenes plus per-session trajectories and handoff records. |
| `sequential-artifact-handoff` | Agents run one after another but can only consume declared artifacts, not full prior transcript. | Copy handoff files/diffs into next prompt; do not inject prior conversation. |
| `parallel-worktree` | Multiple agents run concurrently on isolated worktrees or sandbox snapshots. | Fork workspace, run sessions, capture diffs, then merge or judge. |
| `delegate-tool` | A supervisor agent calls a tool to start a worker agent. | BenchFlow MCP server exposes `delegate_to_agent`; worker is a real session. |
| `a2a-remote` | BenchFlow sends a task to a remote agent service. | A2A client creates/subscribes to tasks and collects artifacts/status updates. |
| `native-team-import` | A real agent spawns its own internal team. | Ingest native team logs and map members/tasks/mailbox to sessions and edges. |

## Uniform adapter contract

The adapter should host real agent sessions. It is not a wrapper around a multi-agent LLM library.

```python
class RealAgentAdapter(Protocol):
    name: str

    async def start_session(self, ctx: StartSessionContext) -> RealAgentSession:
        """Launch or attach to one real agent session."""

    async def send_prompt(self, session: RealAgentSession, prompt: str) -> None:
        """Send the prompt/task to that real agent."""

    async def wait(self, session: RealAgentSession) -> SessionResult:
        """Wait until the agent reaches done/failed/timeout."""

    async def collect_trajectory(self, session: RealAgentSession) -> TrajectoryBundle:
        """Collect ACP/native transcript, tool calls, usage, artifacts, and diff."""

    async def stop_session(self, session: RealAgentSession) -> None:
        """Close the session and clean up leaked processes."""
```

A separate `MultiAgentOrchestrator` composes these sessions according to the task spec and writes `sessions.jsonl`, `handoffs.jsonl`, `multiagent_events.jsonl`, and `agent_graph.json`.

## `task.md` authoring surface

Keep root `agents` and `scenes` for stable BenchFlow roles. Put experimental real-agent tracing under `benchflow.multi_agent`.

```yaml
agents:
  roles:
    planner:
      agent: claude-agent-acp
      model: claude-sonnet-4-6
    implementer:
      agent: codex-acp
      model: gpt-5.5
    reviewer:
      agent: gemini
      model: gemini-3.1-flash-lite-preview

scenes:
  - name: plan
    turns:
      - role: planner
  - name: implement
    turns:
      - role: implementer
  - name: review
    turns:
      - role: reviewer

benchflow:
  multi_agent:
    runtime: real-agent-sessions
    mode: sequential-artifact-handoff
    trace:
      per_agent_trajectories: required
      merged_compat_trajectory: true
      native_transcripts: best-effort
      workspace_diffs: required
      raw_llm_proxy: optional
    handoffs:
      - id: plan-to-implementer
        from: planner
        to: implementer
        artifacts: [/app/plan.md]
        inject: prompt-summary
      - id: implementation-to-reviewer
        from: implementer
        to: reviewer
        artifacts: [/app, /logs/artifacts/patch.diff]
        relation: reviews
    isolation:
      agent_context: fresh-per-session
      workspace: shared
      transcript_sharing: none
```

## Implementation plan

### M0: direct real-agent sessions

1. Reuse current role/scene execution to run different real agents sequentially. **Implemented.**
2. On every real role session, create a `RealAgentSession` record and per-session trajectory directory. **Implemented.**
3. Write ACP events to both the compatibility merged file and the per-agent file. **Implemented.**
4. On disconnect, collect native transcript if the agent has a known scraper. **Pending.**
5. Capture workspace diff and declared handoff artifacts after each role. **Pending.**
6. Emit `sessions.jsonl`, `handoffs.jsonl`, `multiagent_events.jsonl`, and `agent_graph.json`. **Implemented.**
7. Add viewer data for swimlanes: one lane per real agent session. **Pending.**

### M1: supervisor starts workers

1. Add `delegate-tool` mode with a BenchFlow MCP server exposing `delegate_to_agent(role, prompt, workspace_mode, artifacts)`.
2. When the supervisor calls the tool, launch the requested real agent as a child session and return only declared summary/artifacts.
3. Link supervisor tool call id to child session id in `agent_graph.json`.
4. Support parallel forked worktrees/sandbox snapshots for independent workers.
5. Add per-agent usage and cost rollups from ACP usage, native usage metadata, or LiteLLM when available.

### M2: remote and native-team import

1. Add A2A remote-agent adapter using Agent Cards for discovery and A2A tasks for execution.
2. Map A2A task id/context id/status updates/artifacts to `RealAgentSession`, handoff, and relationship events.
3. Add native-team import for Claude Code agent teams: team lead, teammates, task list, mailbox messages, session ids, and transcript pointers.
4. Only after these real-agent paths exist, consider low-priority adapters for LangGraph/CrewAI/AutoGen LLM-workflow traces.

### M3: interactive-user validity

1. Add `IntentSpec` and requirement ids to native `task.md` or verifier-private sidecars.
2. Record `ask_user_calls.jsonl`, `intent_reveal_ledger.jsonl`, `drift_audit.jsonl`, and `simulator_decisions.jsonl` for user-driven runs.
3. Support state-conditional no-op user turns so SWE-Together-style replay can wait until trigger conditions arise.
4. Report User Correction-style intervention counts and Intent Coverage-style simulator-fidelity metrics separately from task reward.
5. Add a publication-grade interactive validity gate that rejects tasks without intent, reveal, and drift-audit evidence.

## Viewer requirements

The viewer should show session swimlanes by real agent instance, handoff artifacts and message edges, supervisor delegation calls linked to worker transcripts, per-agent tokens/cost/latency/tool calls, workspace diffs by agent, verifier/reward events anchored to final state, and coverage warnings when an agent only provided a partial/native-scraped trace.

## Benchmark reporting policy

Multi-agent support must report whether it helped under matched conditions: single-agent baseline, same task set, same sandbox and verifier, same tool permissions or explicitly reported differences, same answer contract, per-agent and total usage, trajectory coverage by agent, relationship graph coverage, success/cost tradeoff, user-correction effort, intent coverage, and simulator drift/audit failures.

## References

- Agent Client Protocol: https://agentclientprotocol.com/get-started/introduction
- A2A Protocol: https://a2a-protocol.org/latest/specification/
- MCP specification: https://modelcontextprotocol.io/specification/2025-06-18
- Claude Code subagents: https://code.claude.com/docs/en/sub-agents
- Claude Code agent teams: https://code.claude.com/docs/en/agent-teams
- HARBOR paper: https://arxiv.org/abs/2606.08610
- BenchAgent paper: https://arxiv.org/abs/2606.05670
- SWE-Interact paper: https://arxiv.org/abs/2606.30573
- SWE-Together paper: https://arxiv.org/abs/2606.29957
- SWE-Together repo: https://github.com/Togetherbench/SWE-Together
- Agent-as-a-Service based on Agent Network: https://arxiv.org/abs/2505.08446
