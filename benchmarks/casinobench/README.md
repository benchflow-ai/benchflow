# CasinoBench — shared-casino real-agent benchmark

CasinoBench is a stateful casino and poker tournament benchmark. It differs
from planner/reviewer workflows: multiple real agents must sit in the **same**
casino service, act through the frozen casino tool surface, and be scored from
the dealer's event log and chip ledger.

`environment.toml` follows the ClawsBench/env-0 manifest pattern:

| Manifest field | Why |
|---|---|
| `base_image` | CasinoBench tasks build `FROM ghcr.io/benchflow-ai/casinobench-base:latest`, which carries the engine, casino service, and `casino` CLI |
| `owns_lifecycle = false` | the task image does not start the service itself; BenchFlow starts `casino-service` before agents run |
| `task_selection.mechanism = "image"` | each generated task image bakes the selected game, seed data, and verifier package |
| `[[environment.services]]` | the single `casino` service is shared by every real agent session in the rollout |

## Single-agent compatibility

The existing CasinoBench generated task packages still run like a normal env-0
mock-service benchmark:

```bash
bench eval run \
  --source-repo benchflow-ai/casinobench --source-path tasks \
  --environment-manifest benchmarks/casinobench/environment.toml \
  --agent claude-agent-acp --model claude-sonnet-4-6
```

## Multi-agent shared-casino shape

For multi-agent play, BenchFlow must not create one sandbox or one casino per
agent. It should run one rollout sandbox, provision one `casino-service`, then
start separate real agent sessions against that same `CASINO_URL`.

The trace artifacts added by this PR make that visible:

```text
trajectory/
  sessions.jsonl                         one record per real agent session
  handoffs.jsonl                         explicit relationships between sessions
  multiagent_events.jsonl                normalized event pointers
  agent_graph.json                       includes the shared casino environment
  agents/<role>/<session>/acp.jsonl      isolated ACP transcript per player agent
```

A CasinoBench multi-agent task should declare roles as player seats and use role
env to pass the intended identity once the casino service exposes multi-player
identity on the tool surface:

```yaml
agents:
  roles:
    seat0:
      agent: claude-agent-acp
      model: claude-sonnet-4-6
      env:
        CASINO_PLAYER_ID: seat0
        CASINO_URL: http://localhost:9001
    seat1:
      agent: codex-acp
      model: gpt-5.5
      env:
        CASINO_PLAYER_ID: seat1
        CASINO_URL: http://localhost:9001
scenes:
  - name: casino-floor
    turns:
      - role: seat0
        prompt: "Join the table as seat0 and play only through the casino CLI."
      - role: seat1
        prompt: "Join the same table as seat1 and play only through the casino CLI."
benchflow:
  multi_agent:
    runtime: real-agent-sessions
    mode: shared-environment
    environment: casinobench
    trace:
      per_agent_trajectories: required
      shared_environment_graph: required
      raw_llm_proxy: optional
```

Current M0 limitation: BenchFlow now records the shared-environment graph and
per-agent session transcripts, but true CasinoBench tournament scheduling still
needs a dealer-driven player loop that prompts only the actor whose
`casino_observe` response contains an active `request_id`. The next runtime
slice should add that scheduler rather than using a fixed planner/reviewer-style
turn list.
