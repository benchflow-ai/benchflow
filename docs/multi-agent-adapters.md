# Multi-agent real-agent adapters and tracing

Status: proposal.

This design is about **real agent runs**, not graph-framework LLM calls. A BenchFlow multi-agent run should launch, supervise, and trace independent agent sessions such as Claude Code, Codex, Gemini CLI, OpenHands, or any ACP / CLI / A2A-compatible agent. LangGraph-style LLM workflows can be supported later as a low-priority adapter, but they are not the target primitive.

The primary trace source is the agent session transcript: ACP events when the agent speaks ACP, native transcript scraping when the agent has its own session log, or a session-factory wrapper when the agent is only available as a CLI or SDK. LiteLLM proxy capture is optional auxiliary evidence for agents that expose model calls through an OpenAI-compatible endpoint. It must not be the main multi-agent abstraction because subscription agents and opaque coding agents may not expose raw provider calls.

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
| BenchFlow current native scenes | Real ACP-backed roles run sequentially in one sandbox. Each role can use a different `agent` and `model`; context is reset by reconnecting. | Treat this as M0. Add per-role session directories and relationship metadata. |
| Agent Client Protocol (ACP) | Client-to-coding-agent protocol for local or remote coding agents. Local agents can run as subprocesses over JSON-RPC stdio. | BenchFlow is the ACP client/orchestrator. Use ACP sessions as the first-class real-agent driver. |
| Claude Code subagents and agent teams | Real Claude Code subagents/teammates have separate context windows. Agent teams have a lead, teammates, task list, mailbox, and direct teammate messaging. | Borrow the data model: lead, teammate, task, mailbox, isolated context, task claiming, cross-agent messages. Do not assume all agents are Claude. |
| A2A Protocol | Agent-to-agent task protocol with Agent Cards, task ids, context ids, streaming task updates, artifacts, and terminal task states. | Add an `a2a-remote-agent` adapter. Map A2A tasks/artifacts/status updates into session and handoff events. |
| MCP | Tool/resource/prompt protocol, not agent-to-agent by itself. | Add a `benchflow_delegate_to_agent` MCP server. A supervisor agent calls this tool; BenchFlow starts a real worker session and links the worker transcript to the supervisor tool call. |
| HARBOR | Staged automation with specialized agents, standardized commands, persistent artifacts, gates, and parallel trials. | Model stages as scenes, gates as verifier/checkpoint events, artifacts as handoff/evidence, and parallel trials as forked session groups. |
| BenchAgent | Normalized evaluation of single-agent, fixed MAS, and evolving MAS workflows under common loader, tools, answer contract, usage accounting, and trajectory logging. | Require matched baselines and report cost/accuracy tradeoffs. Multi-agent is not assumed better. |
| AaaS-AN / Agent Network | Agents and agent groups are vertices; routes are edges; the scheduler maintains an execution graph and context isolation. | Use this as the long-term shape for dynamic networks: session nodes plus route/handoff edges. |
| `lingtai` | No public repo/package/spec found under that spelling during this pass. | Keep as a reserved adapter id until an actual contract is provided. |

## Architecture

### 1. Real agent session is the primitive

Add a `RealAgentSession` runtime record for every launched agent instance:

```json
{
  "session_id": "sess_planner_001",
  "agent_id": "planner",
  "agent_type": "claude-agent-acp",
  "model": "claude-sonnet-4-6",
  "driver": "acp",
  "workspace_mode": "shared",
  "trajectory_path": "trajectory/agents/planner/sess_planner_001/acp.jsonl",
  "native_transcript_path": "trajectory/agents/planner/sess_planner_001/native.jsonl",
  "workspace_diff_path": "trajectory/agents/planner/sess_planner_001/workspace.diff",
  "handoff_out": ["handoff_plan_to_implementer"]
}
```

Supported drivers:

- `acp`: existing BenchFlow ACP connection path.
- `session-factory`: Python session wrapper for SDK-only agents.
- `cli`: prompt-in, transcript/log-out wrapper for CLI agents without ACP.
- `a2a`: remote agent task execution through A2A.
- `mcp-delegate`: supervisor tool call that launches another real agent through BenchFlow.
- `native-team-import`: ingest an agent's own team transcript when BenchFlow did not supervise every spawned member directly.

### 2. Multi-agent run is a graph of sessions

Add `trajectory/agent_graph.json` with nodes for agent sessions and handoff artifacts, and edges such as `handoff_to`, `delegates_to`, `supervises`, `reviews`, `critiques`, `revises_after`, `messages`, `produces`, `consumed_by`, `branches_to`, `merges_into`, and `verifies`.

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

For supervisor/delegation mode:

```yaml
benchflow:
  multi_agent:
    runtime: real-agent-sessions
    mode: delegate-tool
    supervisor: planner
    delegate_tool:
      type: mcp
      name: benchflow_delegate_to_agent
      allowed_workers: [implementer, reviewer]
      worker_workspace: forked-worktree
      return_to_supervisor: summary-and-artifacts
```

Validation rules:

- Every role in `handoffs.from`, `handoffs.to`, `supervisor`, and `allowed_workers` must exist in `agents.roles`.
- `per_agent_trajectories: required` must fail closed if no transcript can be collected for any launched session.
- `workspace_diffs: required` must fail closed when a role with write access has no before/after diff or declared write artifact.
- `transcript_sharing: none` means downstream agents may receive summaries or artifacts, but not upstream raw conversation history.
- `raw_llm_proxy` may only enrich the trace; absence of raw LLM calls cannot make an otherwise valid real-agent transcript incomplete unless explicitly set to `required`.

## Implementation plan

### M0: direct real-agent sessions

1. Reuse current role/scene execution to run different real agents sequentially.
2. On every `connect_as(role)`, create a `RealAgentSession` record and per-session trajectory directory.
3. Write ACP events to both the compatibility merged file and the per-agent file.
4. On disconnect, collect native transcript if the agent has a known scraper.
5. Capture workspace diff and declared handoff artifacts after each role.
6. Emit `sessions.jsonl`, `handoffs.jsonl`, `multiagent_events.jsonl`, and `agent_graph.json`.
7. Add viewer data for swimlanes: one lane per real agent session.

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

## Viewer requirements

The viewer should show:

- session swimlanes by real agent instance;
- handoff artifacts and message edges between sessions;
- supervisor `delegate_to_agent` tool calls linked to worker transcripts;
- per-agent tokens/cost/latency/tool calls;
- workspace diffs by agent;
- verifier and reward events anchored to the final state;
- coverage warnings when an agent only provided a partial/native-scraped trace.

## Benchmark reporting policy

Multi-agent support must report whether it helped under matched conditions: single-agent baseline, same task set, same sandbox and verifier, same tool permissions or explicitly reported differences, same answer contract, per-agent and total usage, trajectory coverage by agent, relationship graph coverage, and success/cost tradeoff.

## References

- Agent Client Protocol: https://agentclientprotocol.com/get-started/introduction
- A2A Protocol: https://a2a-protocol.org/latest/specification/
- MCP specification: https://modelcontextprotocol.io/specification/2025-06-18
- Claude Code subagents: https://code.claude.com/docs/en/sub-agents
- Claude Code agent teams: https://code.claude.com/docs/en/agent-teams
- HARBOR paper: https://arxiv.org/abs/2606.08610
- BenchAgent paper: https://arxiv.org/abs/2606.05670
- Agent-as-a-Service based on Agent Network: https://arxiv.org/abs/2505.08446
